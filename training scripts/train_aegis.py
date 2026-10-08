"""
train_aegis.py
Multi-label risk-category classifier on NVIDIA Aegis 2.0.

What it does, in order:
  1. Loads Aegis from Hugging Face and runs prepare_aegis.sql (DuckDB) on it.
  2. Turns each prompt's category list into a row of 0s and 1s (multi-hot).
  3. Fine-tunes each pre-trained model you name, with early stopping.
  4. Scores on validation (for choosing) and test (for reporting).
  5. Appends one results row per model to results/aegis_results.csv,
     and saves per-category scores and predictions.

The model scores EVERY category (yes/no + probability). The gateway returns one
category: by default the highest-scoring one. With --priority-file, it instead
returns the highest-priority category above the threshold (stretch goal).

Colab setup (Runtime -> Change runtime type -> T4 GPU), then:
  !pip install -q transformers datasets duckdb scikit-learn accelerate sentencepiece
  !python train_aegis.py --models roberta-base microsoft/deberta-v3-base \
        textdetox/bert-multilingual-toxicity-classifier unitary/unbiased-toxic-roberta

Check the data only (no GPU needed, e.g. in a VS Code terminal):
  python train_aegis.py --prep-only

Optimizing the winner afterwards (one model, other settings):
  !python train_aegis.py --models roberta-base --lr 3e-5 --tune-thresholds
"""

import argparse
import math
import os
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

HF_DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"


# ----------------------------------------------------------------------------
# Settings you can change from the command line
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Aegis risk-category classifiers.")
    p.add_argument("--models", nargs="+", default=["roberta-base"],
                   help="One or more Hugging Face model names, trained one after another.")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    p.add_argument("--max-epochs", type=int, default=5,
                   help="Upper limit; early stopping usually ends sooner.")
    p.add_argument("--patience", type=int, default=2,
                   help="Stop after this many epochs without validation improvement.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=0,
                   help="Max tokens per prompt. 0 = set automatically (95th percentile, <= 512).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="A category counts as present when its probability is at least this.")
    p.add_argument("--tune-thresholds", action="store_true",
                   help="Pick a separate threshold per category on VALIDATION, then apply to test.")
    p.add_argument("--priority-file", default="",
                   help="Stretch goal: text file, one category per line, most severe first.")
    p.add_argument("--sql", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_aegis.sql"))
    p.add_argument("--data-dir", default="",
                   help="Optional folder with train.json / validation.json / test.json instead of Hugging Face.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--save-models", action="store_true", help="Save each fine-tuned model to disk.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prep-only", action="store_true",
                   help="Run the SQL preparation, print the summary and categories, then stop (no GPU needed).")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 1. Data: load raw Aegis, run the SQL preparation
# ----------------------------------------------------------------------------
def load_raw_splits(data_dir):
    """Return {'train': df, 'validation': df, 'test': df} with Aegis's original columns."""
    if data_dir:
        def read(path):
            try:
                return pd.read_json(path, lines=True)   # one JSON object per line
            except ValueError:
                return pd.read_json(path)               # one JSON array
        return {s: read(os.path.join(data_dir, f"{s}.json")) for s in ["train", "validation", "test"]}
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET)
    return {s: ds[s].to_pandas() for s in ["train", "validation", "test"]}


def prepare_data(raw, sql_path):
    """Register the raw splits in DuckDB, run prepare_aegis.sql, return the prepared tables."""
    con = duckdb.connect()
    for split, df in raw.items():
        con.register(f"raw_{split}_df", df)
        con.execute(f"CREATE OR REPLACE TABLE raw_{split} AS SELECT * FROM raw_{split}_df")

    with open(sql_path) as f:
        con.execute(f.read())

    print("\nPrepared data:")
    print(con.sql("SELECT * FROM prep_summary").df().to_string(index=False))

    categories = con.sql("SELECT cat, n_train FROM categories ORDER BY n_train DESC").df()
    print("\nCategories (training counts):")
    print(categories.to_string(index=False))

    splits = {s: con.sql(f"SELECT prompt, cats FROM {s}_prep").df()
              for s in ["train", "validation", "test"]}
    return splits, categories["cat"].tolist(), categories["n_train"].to_numpy()


def to_multi_hot(cat_lists, categories):
    """['Violence', 'Harassment'] -> [0, 1, 0, 1, ...] in the order of `categories`."""
    index = {c: i for i, c in enumerate(categories)}
    y = np.zeros((len(cat_lists), len(categories)), dtype=np.float32)
    for row, cats in enumerate(cat_lists):
        for c in cats:
            y[row, index[c]] = 1.0
    return y


# ----------------------------------------------------------------------------
# 2. Choosing one category from the model's scores
# ----------------------------------------------------------------------------
def pick_top(probs):
    """Default: the highest-scoring category for each prompt."""
    return probs.argmax(axis=1)


def pick_priority(probs, thresholds, priority_rank):
    """Stretch goal: among categories above threshold, the highest-priority one.
    If none pass, fall back to the highest score."""
    picks = probs.argmax(axis=1)
    present = probs >= thresholds
    for i in range(len(probs)):
        candidates = np.flatnonzero(present[i])
        if len(candidates):
            picks[i] = candidates[np.argmin(priority_rank[candidates])]
    return picks


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def score(y_true, probs, thresholds, categories, priority_rank=None):
    """All metrics for one split. y_true and probs are (prompts x categories)."""
    y_pred = (probs >= thresholds).astype(int)

    # Multi-label metrics: each category scored as its own yes/no question.
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_cat_p, per_cat_r, per_cat_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    worst = int(np.argmin(per_cat_r))

    # The one category the gateway returns: correct if it is any of the true ones.
    top = pick_top(probs)
    out = {
        "macro_f1": f1,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": p,
        "macro_recall": r,
        "min_recall": float(per_cat_r[worst]),
        "min_recall_category": categories[worst],
        "top1_hit_rate": float(y_true[np.arange(len(top)), top].mean()),
    }
    if priority_rank is not None:
        pr = pick_priority(probs, thresholds, priority_rank)
        out["priority_hit_rate"] = float(y_true[np.arange(len(pr)), pr].mean())

    per_category = pd.DataFrame({
        "category": categories, "precision": per_cat_p, "recall": per_cat_r,
        "f1": per_cat_f1, "support": support, "threshold": thresholds,
    })
    return out, per_category


def tune_thresholds(y_true, probs, grid=np.arange(0.1, 0.91, 0.05)):
    """For each category, the threshold that maximizes its F1 on VALIDATION."""
    best = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        f1s = [f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        best[j] = grid[int(np.argmax(f1s))]
    return best


# ----------------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------------
class WeightedBCETrainer(Trainer):
    """Multi-label loss (one yes/no per category) with extra weight on rare categories."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def choose_max_length(tokenizer, prompts):
    lengths = [len(ids) for ids in tokenizer(list(prompts), truncation=False)["input_ids"]]
    p95 = int(np.percentile(lengths, 95))
    print(f"Token length: mean {np.mean(lengths):.0f}, 95th pct {p95}, max {max(lengths)}")
    return min(512, max(32, p95))


def train_one(model_name, splits, categories, n_train, args, priority_rank):
    print(f"\n{'=' * 70}\nModel: {model_name}\n{'=' * 70}")
    tag = model_name.replace("/", "_")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = args.max_length or choose_max_length(tokenizer, splits["train"]["prompt"])

    def build(df):
        ds = Dataset.from_dict({
            "prompt": df["prompt"].tolist(),
            "labels": to_multi_hot(df["cats"], categories).tolist(),
        })
        return ds.map(lambda b: tokenizer(b["prompt"], truncation=True, max_length=max_length),
                      batched=True, remove_columns=["prompt"])

    data = {s: build(df) for s, df in splits.items()}
    y = {s: to_multi_hot(df["cats"], categories) for s, df in splits.items()}

    # Weight each category by how rare it is: (negatives / positives), capped at 50.
    n_total = len(splits["train"])
    pos_weight = torch.tensor(np.clip((n_total - n_train) / n_train, 1.0, 50.0), dtype=torch.float)

    # Swap the pre-trained head for a fresh one with one output per category.
    # The "MISMATCH ... Reinit" message this prints for some models is expected.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(categories)),
        label2id={c: i for i, c in enumerate(categories)},
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        m, _ = score(labels, sigmoid(logits), np.full(len(categories), args.threshold), categories)
        return {k: v for k, v in m.items() if not isinstance(v, str)}

    steps_per_epoch = math.ceil(len(data["train"]) / args.batch_size)
    use_fp16 = torch.cuda.is_available() and "deberta" not in model_name.lower()  # DeBERTa-v3 can go NaN in fp16

    training_args = TrainingArguments(
        output_dir=os.path.join("checkpoints", tag),
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
        fp16=use_fp16,
        seed=args.seed,
        report_to="none",
    )

    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        pos_weight=pos_weight,
    )
    trainer.train()
    evals = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    best_epoch = max(evals, key=lambda h: h["eval_macro_f1"])["epoch"] if evals else None
    print(f"Best epoch: {best_epoch} (of {trainer.state.epoch:.0f} run)")

    # Probabilities on validation and test
    probs = {s: sigmoid(trainer.predict(data[s]).predictions) for s in ["validation", "test"]}

    # Thresholds: one shared value, or tuned per category on VALIDATION only
    thresholds = np.full(len(categories), args.threshold)
    if args.tune_thresholds:
        thresholds = tune_thresholds(y["validation"], probs["validation"])
        print("Tuned thresholds:", {c: round(float(t), 2) for c, t in zip(categories, thresholds)})

    val_m, _ = score(y["validation"], probs["validation"], thresholds, categories, priority_rank)
    test_m, test_per_cat = score(y["test"], probs["test"], thresholds, categories, priority_rank)

    print("\nValidation:", {k: (round(v, 3) if not isinstance(v, str) else v) for k, v in val_m.items()})
    print("Test:      ", {k: (round(v, 3) if not isinstance(v, str) else v) for k, v in test_m.items()})

    save_results(model_name, tag, args, max_length, best_epoch, thresholds,
                 val_m, test_m, test_per_cat, splits["test"], probs["test"], categories, priority_rank)

    if args.save_models:
        out_dir = os.path.join("saved_models", tag)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        print("Model saved to", out_dir)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# 5. Saving results
# ----------------------------------------------------------------------------
def save_results(model_name, tag, args, max_length, best_epoch, thresholds,
                 val_m, test_m, test_per_cat, test_df, test_probs, categories, priority_rank):
    os.makedirs(args.results_dir, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_name,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "thresholds": "tuned" if args.tune_thresholds else args.threshold,
        "seed": args.seed,
        "n_categories": len(categories),
    }
    row.update({f"val_{k}": v for k, v in val_m.items()})
    row.update({f"test_{k}": v for k, v in test_m.items()})

    summary_path = os.path.join(args.results_dir, "aegis_results.csv")
    pd.DataFrame([row]).to_csv(summary_path, mode="a", index=False,
                               header=not os.path.exists(summary_path))

    stamp = datetime.now().strftime("%m%d_%H%M")
    test_per_cat.to_csv(os.path.join(args.results_dir, f"per_category_{tag}_{stamp}.csv"), index=False)

    top = pick_top(test_probs)
    preds = pd.DataFrame({
        "prompt": test_df["prompt"],
        "true_categories": test_df["cats"].apply(lambda c: ", ".join(c)),
        "predicted_top": [categories[i] for i in top],
        "top_score": test_probs[np.arange(len(top)), top].round(3),
    })
    if priority_rank is not None:
        pr = pick_priority(test_probs, thresholds, priority_rank)
        preds["predicted_priority"] = [categories[i] for i in pr]
    preds.to_csv(os.path.join(args.results_dir, f"predictions_{tag}_{stamp}.csv"), index=False)
    print(f"Results appended to {summary_path}")


# ----------------------------------------------------------------------------
def load_priority(path, categories):
    """Rank each category by its line in the file (0 = most severe).
    Categories missing from the file go last."""
    if not path:
        return None
    with open(path) as f:
        order = [line.strip() for line in f if line.strip()]
    missing = [c for c in categories if c not in order]
    if missing:
        print("Not in priority file (ranked last):", missing)
    order += missing
    return np.array([order.index(c) for c in categories])


def main():
    args = parse_args()
    raw = load_raw_splits(args.data_dir)
    splits, categories, n_train = prepare_data(raw, args.sql)
    if args.prep_only:
        os.makedirs(args.results_dir, exist_ok=True)
        for name, df in splits.items():
            df.assign(cats=df["cats"].apply(", ".join)).to_csv(
                os.path.join(args.results_dir, f"{name}_prep.csv"), index=False)
        print(f"\nPrepared splits saved to {args.results_dir}/ (train_prep.csv, ...). Stopping before training.")
        return
    priority_rank = load_priority(args.priority_file, categories)
    for model_name in args.models:
        train_one(model_name, splits, categories, n_train, args, priority_rank)


if __name__ == "__main__":
    main()