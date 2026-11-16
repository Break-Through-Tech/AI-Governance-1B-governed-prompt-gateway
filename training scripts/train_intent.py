"""
train_intent.py
Fine-tunes roberta-base to classify customer-service intents for node 1 (classify_intent),
the same way the team trained the risk models. Node 1 covers insurance and retail banking.

What each dataset does:
  - Bitext Insurance + Retail Banking: TRAIN the model on our 23 gateway intents
    (GROUPS in build_intent_labels.py, so both scripts always use the same labels).
  - CLINC150, train split: TRAINS a 24th class, "unknown", on questions that are not about insurance
    or banking (out-of-scope queries, weather, flights, recipes, ...). All of them get the one label
    "unknown"; CLINC150's own labels are not used. Turn this off with --no-unknown-class.
  - CLINC150, validation + test splits: tune min_confidence and test that non-insurance/banking
    questions come back "unknown". A question is "unknown" when the model predicts the unknown class
    OR its top probability is below min_confidence.

What it does, in order:
  1. Loads Bitext (both industries) and CLINC150.
  2. Splits Bitext 80 / 10 / 10 into train / validation / test, stratified by intent, and adds the
     CLINC150 "unknown" questions (train split to train, validation split to validate).
  3. Fine-tunes the model on train, with early stopping on validation macro-F1.
  4. Chooses min_confidence on the TUNE set (Bitext validation + CLINC150 validation split).
  5. Reports results on the TEST set (Bitext test + CLINC150 test split), which nothing was tuned on.
  6. Saves the model, optionally uploads it to your Hugging Face account, and with --write-config
     switches node 1 to it in gateway/config.yaml.

Colab setup (Runtime -> Change runtime type -> T4 GPU; needed, training on CPU takes hours):
  !git clone -b node/intent <your repo url>
  %cd <repo folder>
  !pip install -q -r gateway/requirements.txt transformers datasets accelerate scikit-learn pandas

  # Log in so the script can upload the model (paste a token with WRITE access from
  # huggingface.co/settings/tokens):
  from huggingface_hub import notebook_login
  notebook_login()

  !python "training scripts/train_intent.py" --push-to-hub YOUR-HF-USERNAME/customer-intent-roberta --write-config
  !python -m pytest gateway/tests

Faster trial run first (fewer training questions, 1 epoch), to check everything works:
  !python "training scripts/train_intent.py" --max-train-per-intent 300 --max-epochs 1

To compare with a model that has no "unknown" class (it relies on min_confidence alone):
  !python "training scripts/train_intent.py" --no-unknown-class --output-dir "training scripts/results/intent_model_no_unknown"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_intent_labels as shared  # noqa: E402  (GROUPS, data loading, CLINC150, metrics)

REPO = shared.REPO
CONFIG_PATH = shared.CONFIG_PATH
RESULTS = shared.RESULTS
UNKNOWN = shared.UNKNOWN


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune an intent classifier for node 1 on Bitext.")
    p.add_argument("--base-model", default="roberta-base", help="Pre-trained model to fine-tune.")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    p.add_argument("--max-epochs", type=int, default=4, help="Upper limit; early stopping usually ends sooner.")
    p.add_argument("--patience", type=int, default=1,
                   help="Stop after this many epochs without validation improvement.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=64, help="Max tokens per question (Bitext questions are short).")
    p.add_argument("--max-train-per-intent", type=int, default=0,
                   help="Cap training questions per gateway intent (0 = use all). Use ~300 for a quick trial.")
    p.add_argument("--clinc-other-per-intent", type=int, default=3,
                   help="Questions per other-domain CLINC150 intent in the tune and test sets.")
    p.add_argument("--no-unknown-class", action="store_true",
                   help="Don't train an 'unknown' class on CLINC150; rely on min_confidence alone.")
    p.add_argument("--unknown-per-intent", type=int, default=15,
                   help="Training questions per other-domain CLINC150 intent for the 'unknown' class "
                        "(all of CLINC150's out-of-scope training questions are used too).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bitext-csv", help="CSV export instead of downloading (columns: instruction, intent, industry).")
    p.add_argument("--clinc-csv", help="CSV export instead of downloading (columns: text, intent, split).")
    p.add_argument("--output-dir", default=str(RESULTS / "intent_model"), help="Where to save the trained model.")
    p.add_argument("--push-to-hub", default="",
                   help="Upload the model to this Hugging Face repo, e.g. YOUR-HF-USERNAME/customer-intent-roberta.")
    p.add_argument("--write-config", action="store_true",
                   help="Switch node 1 to this model in gateway/config.yaml (backend, model, threshold).")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# 1–2. Data
# ----------------------------------------------------------------------------------------------
def split_bitext(df: pd.DataFrame, seed: int, cap: int):
    to_group = {b: g for g, spec in shared.GROUPS.items() for b in spec["bitext"]}
    df = df.assign(group=df["intent"].map(to_group))
    train, rest = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["intent"], random_state=seed)
    if cap:
        train = train.sample(frac=1, random_state=seed).groupby("group", group_keys=False).head(cap)
    return train, val, test


def unknown_rows(clinc: pd.DataFrame, split: str, per_intent: int, seed: int) -> pd.DataFrame:
    """CLINC150 questions for the "unknown" class, shaped like the Bitext rows (instruction, group)."""
    neg = shared.clinc_negatives(clinc, split, per_intent, seed)
    return pd.DataFrame({"instruction": neg["instruction"].values, "group": UNKNOWN,
                         "intent": "clinc/" + neg["original_intent"].astype(str).values})


def as_eval_frame(bitext_part: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"instruction": bitext_part["instruction"].values, "true": bitext_part["group"].values,
                         "source": "bitext", "original_intent": bitext_part["intent"].values})


class TextDataset:
    """Tokenized questions + label ids, in the shape the Hugging Face Trainer expects."""

    def __init__(self, texts, labels, label2id, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = [label2id[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


# ----------------------------------------------------------------------------------------------
# 3. Train
# ----------------------------------------------------------------------------------------------
def train(args, labels, train_df, val_df):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    label2id = {label: i for i, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label2id,
        ignore_mismatched_sizes=True)   # replace any existing classifier head with our 12 intents

    train_ds = TextDataset(train_df["instruction"], train_df["group"], label2id, tokenizer, args.max_length)
    val_ds = TextDataset(val_df["instruction"], val_df["group"], label2id, tokenizer, args.max_length)

    def compute_metrics(pred):
        y_pred = pred.predictions.argmax(-1)
        return {"macro_f1": f1_score(pred.label_ids, y_pred, average="macro"),
                "accuracy": accuracy_score(pred.label_ids, y_pred)}

    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(REPO / "checkpoints" / "intent"),
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=int(0.1 * steps_per_epoch * args.max_epochs),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    print(f"Training {args.base_model} on {len(train_ds):,} questions "
          f"({'GPU' if torch.cuda.is_available() else 'CPU: this will be slow'})…")
    trainer.train()
    return trainer.model, tokenizer


def make_predictor(model, tokenizer, max_length, batch_size=128):
    """predict(texts) -> (top label per text, top probability per text)."""
    import torch
    device = next(model.parameters()).device
    model.eval()

    @torch.no_grad()
    def predict(texts):
        tops, confs = [], []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            tops += [model.config.id2label[int(j)] for j in probs.argmax(-1)]
            confs += probs.max(-1).tolist()
        return tops, confs
    return predict


def score(frame: pd.DataFrame, predict) -> pd.DataFrame:
    tops, confs = predict(frame["instruction"].tolist())
    return frame.assign(top=tops, confidence=confs)


# ----------------------------------------------------------------------------------------------
# 6. Point node 1 at the model in config.yaml (only three lines of the intent: section change)
# ----------------------------------------------------------------------------------------------
def write_config(model_ref: str, threshold: float):
    text = CONFIG_PATH.read_text()
    match = re.search(r"(?ms)^intent:.*?(?=^\S)", text)
    if not match:
        raise SystemExit("Couldn't find the intent: section in config.yaml; set backend, "
                         "finetuned_model and min_confidence by hand.")
    block = match.group(0)
    for key, value in (("backend", "finetuned"), ("finetuned_model", model_ref), ("min_confidence", threshold)):
        block, n = re.subn(rf"(?m)^  {key}:.*$", f"  {key}: {value}", block, count=1)
        if n == 0:   # key missing (older config): add it under "intent:"
            block = block.replace("\n", f"\n  {key}: {value}\n", 1)
    CONFIG_PATH.write_text(text[:match.start()] + block + text[match.end():])


# ----------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading Bitext and CLINC150…")
    bitext = shared.load_bitext(args.bitext_csv)
    shared.check_groups(set(bitext["intent"]))
    clinc = shared.load_clinc(args.clinc_csv)
    train_df, val_df, test_df = split_bitext(bitext, args.seed, args.max_train_per_intent)
    print(f"  Bitext (insurance + banking): {len(train_df):,} train, {len(val_df):,} validation, "
          f"{len(test_df):,} test")

    labels = list(shared.GROUPS)
    fit_train, fit_val = train_df, val_df
    if not args.no_unknown_class:
        labels.append(UNKNOWN)
        unk_train = unknown_rows(clinc, "train", args.unknown_per_intent, args.seed)
        unk_val = unknown_rows(clinc, "validation", args.clinc_other_per_intent, args.seed)
        fit_train = pd.concat([train_df, unk_train], ignore_index=True)
        fit_val = pd.concat([val_df, unk_val], ignore_index=True)
        print(f"  'unknown' class: {len(unk_train):,} CLINC150 training questions, {len(unk_val):,} validation")
    print(f"  {len(labels)} classes")

    model, tokenizer = train(args, labels, fit_train, fit_val)
    predict = make_predictor(model, tokenizer, args.max_length)

    tune = pd.concat([as_eval_frame(val_df),
                      shared.clinc_negatives(clinc, "validation", args.clinc_other_per_intent, args.seed)],
                     ignore_index=True)
    test = pd.concat([as_eval_frame(test_df),
                      shared.clinc_negatives(clinc, "test", args.clinc_other_per_intent, args.seed)],
                     ignore_index=True)
    print("Scoring the tune and test sets…")
    threshold, sweep = shared.choose_threshold(score(tune, predict))
    test_scored = score(test, predict)
    test_scored["predicted"] = shared.predict(test_scored, threshold)
    r = shared.rates(test_scored, test_scored["predicted"])

    labels = list(shared.GROUPS) + [UNKNOWN]
    report = classification_report(test_scored["true"], test_scored["predicted"], labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(test_scored["true"], test_scored["predicted"], labels=labels),
                      index=[f"true: {l}" for l in labels], columns=labels)
    unk_note = "no 'unknown' class" if args.no_unknown_class else "with an 'unknown' class trained on CLINC150"
    summary = (f"Backend: finetuned ({args.base_model}, {unk_note})   "
               f"min_confidence (chosen on the tune set): {threshold}\n"
               f"Trained on {len(fit_train):,} questions ({len(train_df):,} Bitext insurance + banking)\n"
               f"TEST SET RESULTS\n"
               f"  Macro-F1: {r['macro_f1']:.3f}   Accuracy: {r['accuracy']:.3f}\n"
               f"  Insurance + banking questions labelled right: {r['insurance_correct']:.1%}\n"
               f"  Insurance + banking wrongly marked unknown:   {r['insurance_marked_unknown']:.1%}\n"
               f"  CLINC150 out-of-scope correctly unknown:      {r['clinc_oos_caught']:.1%}\n"
               f"  CLINC150 other-domain correctly unknown:      {r['clinc_other_domain_caught']:.1%}\n\n"
               f"{report}")
    print("\n" + summary)

    (RESULTS / "intent_report_finetuned.txt").write_text(summary)
    sweep.to_csv(RESULTS / "intent_threshold_sweep_finetuned.csv", index=False)
    cm.to_csv(RESULTS / "intent_confusion_finetuned.csv")
    test_scored[test_scored["true"] != test_scored["predicted"]].to_csv(
        RESULTS / "intent_errors_finetuned.csv", index=False)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved the model to {args.output_dir} and the results to {RESULTS}")
    model_ref = args.output_dir
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        model_ref = args.push_to_hub
        print(f"Uploaded the model to https://huggingface.co/{args.push_to_hub}")

    if args.write_config:
        if not args.push_to_hub:
            print("Note: config.yaml will point at a folder on THIS machine. Use --push-to-hub so "
                  "teammates (and the website) can download the model.")
        write_config(model_ref, threshold)
        print(f"Node 1 now uses the fine-tuned model ({model_ref}). Next: python -m pytest gateway/tests")
    else:
        print("config.yaml not changed. Rerun with --write-config to switch node 1 to this model.")


if __name__ == "__main__":
    main()
