"""
train_intent.py
Fine-tunes roberta-base to classify customer-service intents for node 1 (classify_intent),
the same way the team trained the risk models. Node 1 covers insurance and retail banking.

What each dataset does:
  - Bitext Insurance + Retail Banking: TRAIN the model on our 23 gateway intents
    (GROUPS below).
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
     points node 1 at it in gateway/config.yaml (finetuned_model and min_confidence).

Colab setup (Runtime -> Change runtime type -> T4 GPU; needed, training on CPU takes hours):
  !git clone -b node/intent <your repo url>
  %cd <repo folder>
  !pip install -q -r gateway/requirements.txt torch transformers datasets accelerate scikit-learn pandas

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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "gateway" / "config.yaml"
RESULTS = REPO / "training scripts" / "results"
UNKNOWN = "unknown"

# One Bitext dataset per industry. Intents are named "<industry>/<bitext intent>" because the two
# datasets share some intent names (customer_service, human_agent).
BITEXT = {
    "insurance": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
    "banking": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
}

CLINC = "clinc/clinc_oos"
CLINC_CONFIG = "plus"                             # the version with the most out-of-scope examples

# ----------------------------------------------------------------------------------------------
# 1. Our gateway intents and which Bitext intents each one covers ("<industry>/<bitext intent>").
#    This is a TEAM DECISION. Change it here, rerun, and the config + evaluation follow.
#    Every Bitext intent must appear exactly once (the script checks).
#    "Talk to a person" is ONE shared label: it means the same thing in both industries, and the
#    two datasets' versions of it are almost word for word the same.
# ----------------------------------------------------------------------------------------------
GROUPS = {
    # --- insurance
    "insurance.claims.coverage_question": {"bitext": ["insurance/check_coverage"]},
    "insurance.claims.file_claim":        {"bitext": ["insurance/file_claim", "insurance/report_incident"]},
    "insurance.claims.status":            {"bitext": ["insurance/track_claim", "insurance/receive_payment"]},
    "insurance.claims.settlement":        {"bitext": ["insurance/accept_settlement", "insurance/negotiate_settlement",
                                                      "insurance/reject_settlement",
                                                      "insurance/appeal_denied_insurance_claim"]},
    "insurance.policy.question":          {"bitext": ["insurance/information_auto_insurance",
                                                      "insurance/information_home_insurance",
                                                      "insurance/information_health_insurance",
                                                      "insurance/information_life_insurance",
                                                      "insurance/information_pet_insurance",
                                                      "insurance/information_travel_insurance"]},
    "insurance.policy.change":            {"bitext": ["insurance/change_coverage", "insurance/upgrade_coverage",
                                                      "insurance/downgrade_coverage",
                                                      "insurance/change_personal_details",
                                                      "insurance/renew_insurance_policy"]},
    "insurance.policy.cancel":            {"bitext": ["insurance/cancel_insurance_policy",
                                                      "insurance/cancellation_fees"]},
    "insurance.quote.new_policy":         {"bitext": ["insurance/buy_insurance_policy",
                                                      "insurance/compare_insurance_policies",
                                                      "insurance/calculate_insurance_quote", "insurance/check_rates"]},
    "insurance.billing.payment":          {"bitext": ["insurance/pay", "insurance/payment_methods",
                                                      "insurance/check_payments", "insurance/schedule_payments"]},
    "insurance.billing.dispute":          {"bitext": ["insurance/dispute_invoice", "insurance/report_payment_issue"]},
    "insurance.support.complaint":        {"bitext": ["insurance/file_complaint"]},
    "insurance.support.info":             {"bitext": ["insurance/general_information",
                                                      "insurance/schedule_appointment"]},
    # --- retail banking
    "banking.account.open_close":         {"bitext": ["banking/create_account", "banking/close_account"]},
    "banking.account.balance_activity":   {"bitext": ["banking/check_recent_transactions",
                                                      "banking/check_current_balance_on_card"]},
    "banking.card.manage":                {"bitext": ["banking/activate_card", "banking/activate_card_international_usage",
                                                      "banking/block_card", "banking/cancel_card"]},
    "banking.fees":                       {"bitext": ["banking/check_fees", "banking/check_card_annual_fee"]},
    "banking.atm.problem":                {"bitext": ["banking/dispute_ATM_withdrawal", "banking/recover_swallowed_card"]},
    "banking.locations":                  {"bitext": ["banking/find_ATM", "banking/find_branch"]},
    "banking.loan.apply":                 {"bitext": ["banking/apply_for_loan", "banking/apply_for_mortgage"]},
    "banking.loan.manage":                {"bitext": ["banking/cancel_loan", "banking/cancel_mortgage",
                                                      "banking/check_loan_payments", "banking/check_mortgage_payments"]},
    "banking.transfer":                   {"bitext": ["banking/make_transfer", "banking/cancel_transfer"]},
    "banking.password":                   {"bitext": ["banking/get_password", "banking/set_up_password"]},
    # --- shared by both industries
    "support.contact":                    {"bitext": ["insurance/agent", "insurance/customer_service",
                                                      "insurance/human_agent", "insurance/insurance_representative",
                                                      "banking/customer_service", "banking/human_agent"]},
}

# CLINC150 intents that a customer of our two industries COULD legitimately ask the gateway about.
# They are left out of "unknown" (for training and for testing), because "unknown" would be wrong.
CLINC_EXCLUDE = {
    # insurance
    "insurance", "insurance_change",
    # CLINC150's whole "banking" domain
    "freeze_account", "routing", "pin_change", "bill_due", "pay_bill", "account_blocked", "interest_rate",
    "min_payment", "bill_balance", "transfer", "order_checks", "balance", "spending_history", "transactions",
    "report_fraud",
    # CLINC150's whole "credit cards" domain
    "replacement_card_duration", "expiration_date", "damaged_card", "improve_credit_score", "report_lost_card",
    "card_declined", "credit_limit_change", "apr", "redeem_rewards", "credit_limit", "rewards_balance",
    "application_status", "credit_score", "new_card", "international_fees",
    # money questions a bank customer might ask
    "direct_deposit", "rollover_401k",
}


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
                   help="Point node 1 at this model in gateway/config.yaml (finetuned_model, min_confidence).")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# 1–2. Data
# ----------------------------------------------------------------------------------------------
def load_bitext(csv_path=None) -> pd.DataFrame:
    """Both Bitext datasets as columns instruction, intent ("<industry>/<bitext intent>")."""
    if csv_path:
        df = pd.read_csv(csv_path)                     # columns: instruction, intent, industry
    else:
        from datasets import load_dataset
        df = pd.concat([load_dataset(repo, split="train").to_pandas().assign(industry=industry)
                        for industry, repo in BITEXT.items()], ignore_index=True)
    df = df[["instruction", "intent", "industry"]].dropna()
    df["intent"] = df["industry"] + "/" + df["intent"]
    # Bitext writes slots like {{Policy Number}}; turn them into plain words a customer might type.
    df["instruction"] = (df["instruction"]
                         .str.replace(r"\{\{\s*(.+?)\s*\}\}", lambda m: m.group(1).lower(), regex=True)
                         .str.replace(r"\s+", " ", regex=True).str.strip())
    return df.drop_duplicates("instruction").reset_index(drop=True)[["instruction", "intent"]]


def load_clinc(csv_path=None) -> pd.DataFrame:
    """CLINC150 train, validation and test splits as columns text, clinc_intent, split."""
    if csv_path:
        df = pd.read_csv(csv_path).rename(columns={"intent": "clinc_intent"})
        return df[["text", "clinc_intent", "split"]]
    from datasets import load_dataset
    ds = load_dataset(CLINC, CLINC_CONFIG)
    names = ds["test"].features["intent"].names          # label ids -> names ("oos" is one of them)
    parts = []
    for split in ("train", "validation", "test"):
        part = ds[split].to_pandas()
        parts.append(pd.DataFrame({"text": part["text"], "clinc_intent": part["intent"].map(names.__getitem__),
                                   "split": split}))
    return pd.concat(parts, ignore_index=True)


def clinc_negatives(clinc: pd.DataFrame, split: str, per_intent: int, seed: int) -> pd.DataFrame:
    """Everything that should come back "unknown": all out-of-scope queries + a few per other intent."""
    part = clinc[(clinc["split"] == split) & ~clinc["clinc_intent"].isin(CLINC_EXCLUDE)]
    oos = part[part["clinc_intent"] == "oos"]
    other = (part[part["clinc_intent"] != "oos"].sample(frac=1, random_state=seed)
             .groupby("clinc_intent", group_keys=False).head(per_intent))
    rows = pd.concat([oos, other])
    return pd.DataFrame({"instruction": rows["text"].values, "true": UNKNOWN,
                         "source": ["clinc_oos" if c == "oos" else "clinc_other_domain" for c in rows["clinc_intent"]],
                         "original_intent": rows["clinc_intent"].values})


def check_groups(bitext_intents: set[str]):
    mapped = Counter(b for g in GROUPS.values() for b in g["bitext"])
    twice = [b for b, n in mapped.items() if n > 1]
    missing = sorted(bitext_intents - set(mapped))
    extra = sorted(set(mapped) - bitext_intents)
    if twice or missing or extra:
        raise SystemExit(f"Fix GROUPS first.\n  mapped twice: {twice}\n  not mapped: {missing}\n"
                         f"  not in the dataset: {extra}")


def split_bitext(df: pd.DataFrame, seed: int, cap: int):
    to_group = {b: g for g, spec in GROUPS.items() for b in spec["bitext"]}
    df = df.assign(group=df["intent"].map(to_group))
    train, rest = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["intent"], random_state=seed)
    if cap:
        train = train.sample(frac=1, random_state=seed).groupby("group", group_keys=False).head(cap)
    return train, val, test


def unknown_rows(clinc: pd.DataFrame, split: str, per_intent: int, seed: int) -> pd.DataFrame:
    """CLINC150 questions for the "unknown" class, shaped like the Bitext rows (instruction, group)."""
    neg = clinc_negatives(clinc, split, per_intent, seed)
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


def apply_threshold(scored: pd.DataFrame, threshold: float) -> pd.Series:
    """The top label, or "unknown" when its probability is below the threshold."""
    return scored["top"].where(scored["confidence"] >= threshold, UNKNOWN)


def rates(scored: pd.DataFrame, pred: pd.Series) -> dict:
    ins = scored["true"] != UNKNOWN
    return {
        "macro_f1": f1_score(scored["true"], pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(scored["true"], pred),
        "in_scope_correct": float((pred[ins] == scored["true"][ins]).mean()),
        "in_scope_marked_unknown": float((pred[ins] == UNKNOWN).mean()),
        "clinc_oos_caught": float((pred[scored["source"] == "clinc_oos"] == UNKNOWN).mean()),
        "clinc_other_domain_caught": float((pred[scored["source"] == "clinc_other_domain"] == UNKNOWN).mean()),
    }


def choose_threshold(tune: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = [{"threshold": t, **rates(tune, apply_threshold(tune, t))}
            for t in [round(x * 0.025, 3) for x in range(0, 37)]]           # 0.00 .. 0.90
    sweep = pd.DataFrame(rows)
    return float(sweep.loc[sweep["macro_f1"].idxmax(), "threshold"]), sweep


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
        raise SystemExit("Couldn't find the intent: section in config.yaml; set "
                         "finetuned_model and min_confidence by hand.")
    block = match.group(0)
    for key, value in (("finetuned_model", model_ref), ("min_confidence", threshold)):
        block, n = re.subn(rf"(?m)^  {key}:.*$", f"  {key}: {value}", block, count=1)
        if n == 0:   # key missing (older config): add it under "intent:"
            block = block.replace("\n", f"\n  {key}: {value}\n", 1)
    CONFIG_PATH.write_text(text[:match.start()] + block + text[match.end():])


# ----------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading Bitext and CLINC150…")
    bitext = load_bitext(args.bitext_csv)
    check_groups(set(bitext["intent"]))
    clinc = load_clinc(args.clinc_csv)
    train_df, val_df, test_df = split_bitext(bitext, args.seed, args.max_train_per_intent)
    print(f"  Bitext (insurance + banking): {len(train_df):,} train, {len(val_df):,} validation, "
          f"{len(test_df):,} test")

    labels = list(GROUPS)
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
                      clinc_negatives(clinc, "validation", args.clinc_other_per_intent, args.seed)],
                     ignore_index=True)
    test = pd.concat([as_eval_frame(test_df),
                      clinc_negatives(clinc, "test", args.clinc_other_per_intent, args.seed)],
                     ignore_index=True)
    print("Scoring the tune and test sets…")
    threshold, sweep = choose_threshold(score(tune, predict))
    test_scored = score(test, predict)
    test_scored["predicted"] = apply_threshold(test_scored, threshold)
    r = rates(test_scored, test_scored["predicted"])

    labels = list(GROUPS) + [UNKNOWN]
    report = classification_report(test_scored["true"], test_scored["predicted"], labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(test_scored["true"], test_scored["predicted"], labels=labels),
                      index=[f"true: {l}" for l in labels], columns=labels)
    unk_note = "no 'unknown' class" if args.no_unknown_class else "with an 'unknown' class trained on CLINC150"
    summary = (f"Model: {args.base_model}, {unk_note}   "
               f"min_confidence (chosen on the tune set): {threshold}\n"
               f"Trained on {len(fit_train):,} questions ({len(train_df):,} Bitext insurance + banking)\n"
               f"TEST SET RESULTS\n"
               f"  Macro-F1: {r['macro_f1']:.3f}   Accuracy: {r['accuracy']:.3f}\n"
               f"  Insurance + banking questions labelled right: {r['in_scope_correct']:.1%}\n"
               f"  Insurance + banking wrongly marked unknown:   {r['in_scope_marked_unknown']:.1%}\n"
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
        print(f"Node 1 now points at {model_ref} (min_confidence {threshold}). Next: python -m pytest gateway/tests")
    else:
        print("config.yaml not changed. Rerun with --write-config to point node 1 at this model.")


if __name__ == "__main__":
    main()
