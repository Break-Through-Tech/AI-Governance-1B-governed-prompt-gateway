"""
build_intent_labels.py
Builds node 1's intent labels from Bitext and tests them with Bitext + CLINC150.
This is the no-training BASELINE; train_intent.py fine-tunes roberta-base on the same labels.

Node 1 covers two regulated industries: insurance and retail banking.

Two kinds of data, two jobs:
  - Bitext Insurance (39 intents) + Bitext Retail Banking (26 intents): WHAT kind of customer-service
    question it is. Give our 23 intent labels, the example questions in config.yaml, and test questions.
  - CLINC150 (150 general-assistant intents + "out-of-scope"): what is NOT an insurance or banking
    question. Its out-of-scope queries and its other-domain intents (weather, flights, recipes, ...)
    should all come back "unknown".

What it does, in order:
  1. Loads both datasets from Hugging Face (or from CSV exports with --bitext-csv / --clinc-csv).
  2. Merges the 65 Bitext intents (39 insurance + 26 banking) into our 23 gateway intents (GROUPS below).
  3. Picks ~20 example questions per gateway intent for config.yaml.
  4. Builds two sets of held-out questions, each with insurance questions AND CLINC150 non-insurance ones:
       tune  (Bitext held-out half + CLINC150 validation split): used to choose min_confidence
       test  (other Bitext held-out half + CLINC150 test split):  used ONLY for the reported numbers
  5. Runs the REAL node (gateway/nodes/intent.py) on both sets.
  6. Writes results to "training scripts/results/intent_*" and, with --write-config,
     puts the new labels, examples and threshold into gateway/config.yaml.

Colab setup (Runtime -> Change runtime type -> T4 GPU helps for the embedding backend):
  !git clone <your repo url>
  %cd <repo folder>
  !pip install -q -r gateway/requirements.txt datasets sentence-transformers scikit-learn pandas
  !python "training scripts/build_intent_labels.py"

When the numbers look right, write them into config.yaml and run the tests:
  !python "training scripts/build_intent_labels.py" --write-config
  !pytest gateway/tests

Datasets (credit all three in the write-up):
  bitext/Bitext-insurance-llm-chatbot-training-dataset       CDLA-Sharing 1.0, synthetic
  bitext/Bitext-retail-banking-llm-chatbot-training-dataset  CDLA-Sharing 1.0, synthetic
  clinc/clinc_oos ("plus" config)                             CC BY 3.0, crowdsourced
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gateway import settings                      # noqa: E402  (needs the repo on the path first)
from gateway.nodes import intent as intent_node  # noqa: E402

# One Bitext dataset per industry. Intents are named "<industry>/<bitext intent>" because the two
# datasets share some intent names (customer_service, human_agent).
BITEXT = {
    "insurance": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
    "banking": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
}
CLINC = "clinc/clinc_oos"
CLINC_CONFIG = "plus"                             # the version with the most out-of-scope examples
CONFIG_PATH = REPO / "gateway" / "config.yaml"
RESULTS = REPO / "training scripts" / "results"
UNKNOWN = intent_node.UNKNOWN

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
    p = argparse.ArgumentParser(description="Build node 1's intent labels from Bitext; test with Bitext + CLINC150.")
    p.add_argument("--backend", choices=["lexical", "embedding"], default="embedding")
    p.add_argument("--examples-per-intent", type=int, default=20,
                   help="Example questions per gateway intent written into config.yaml.")
    p.add_argument("--heldout-per-intent", type=int, default=400,
                   help="Held-out Bitext questions per gateway intent, split half tune / half test.")
    p.add_argument("--clinc-other-per-intent", type=int, default=3,
                   help="Questions taken from each non-insurance CLINC150 intent, per split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bitext-csv", help="CSV export instead of downloading (columns: instruction, intent, "
                                        "industry = insurance or banking).")
    p.add_argument("--clinc-csv", help="CSV export instead of downloading (columns: text, intent, split; "
                                       "intent as a name such as 'oos', split = train, validation or test).")
    p.add_argument("--write-config", action="store_true",
                   help="Write the new intents, examples, backend and threshold into gateway/config.yaml.")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# 2. Load and clean
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


# ----------------------------------------------------------------------------------------------
# 3–4. Split Bitext: examples for config.yaml, then held-out questions split into tune / test
# ----------------------------------------------------------------------------------------------
def split_bitext(df: pd.DataFrame, n_examples: int, n_heldout: int, seed: int):
    to_group = {b: g for g, spec in GROUPS.items() for b in spec["bitext"]}
    df = df.assign(group=df["intent"].map(to_group)).sample(frac=1, random_state=seed)
    examples, tune, test = {}, [], []
    for group, spec in GROUPS.items():
        rows = df[df["group"] == group]
        # Spread the examples across the group's Bitext intents, so every sub-intent is represented.
        k = -(-n_examples // len(spec["bitext"]))           # ceiling division
        picked = rows.groupby("intent", group_keys=False).head(k).head(n_examples)
        examples[group] = picked["instruction"].tolist()
        rest = rows.drop(picked.index).head(n_heldout)
        half = len(rest) // 2
        tune.append(rest.iloc[:half])
        test.append(rest.iloc[half:])

    def frame(parts):
        d = pd.concat(parts)
        return pd.DataFrame({"instruction": d["instruction"].values, "true": d["group"].values,
                             "source": "bitext", "original_intent": d["intent"].values})
    return examples, frame(tune), frame(test)


# ----------------------------------------------------------------------------------------------
# 5. Run the real node's scorer, choose the threshold on TUNE, report on TEST
# ----------------------------------------------------------------------------------------------
def score_all(qs: pd.DataFrame, intents_cfg: dict, backend: str, name: str) -> pd.DataFrame:
    with settings.override({"intent": {"intents": intents_cfg, "backend": backend}}):
        tops, confs = [], []
        for i, q in enumerate(qs["instruction"], 1):
            scores = intent_node.score_intents(intent_node.extract_question(q))
            top = max(scores, key=scores.get)
            tops.append(top)
            confs.append(scores[top])
            if i % 500 == 0:
                print(f"  {name}: scored {i}/{len(qs)}")
    return qs.assign(top=tops, confidence=confs)


def predict(scored: pd.DataFrame, threshold: float) -> pd.Series:
    return scored["top"].where(scored["confidence"] >= threshold, UNKNOWN)


def rates(scored: pd.DataFrame, pred: pd.Series) -> dict:
    ins = scored["true"] != UNKNOWN
    return {
        "macro_f1": f1_score(scored["true"], pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(scored["true"], pred),
        "insurance_correct": float((pred[ins] == scored["true"][ins]).mean()),
        "insurance_marked_unknown": float((pred[ins] == UNKNOWN).mean()),
        "clinc_oos_caught": float((pred[scored["source"] == "clinc_oos"] == UNKNOWN).mean()),
        "clinc_other_domain_caught": float((pred[scored["source"] == "clinc_other_domain"] == UNKNOWN).mean()),
    }


def choose_threshold(tune: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = [{"threshold": t, **rates(tune, predict(tune, t))}
            for t in [round(x * 0.025, 3) for x in range(0, 37)]]           # 0.00 .. 0.90
    sweep = pd.DataFrame(rows)
    return float(sweep.loc[sweep["macro_f1"].idxmax(), "threshold"]), sweep


# ----------------------------------------------------------------------------------------------
# 6. Write config.yaml (only the intent: section; every comment elsewhere is kept)
# ----------------------------------------------------------------------------------------------
def intent_section(examples: dict, backend: str, threshold: float) -> str:
    old = settings.get("intent")
    body = {
        "backend": backend,
        "finetuned_model": old.get("finetuned_model", "YOUR-HF-USERNAME/customer-intent-roberta"),
        "finetuned_max_length": old.get("finetuned_max_length", 64),
        "embedding_model": old.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        "min_confidence": threshold,
        "intents": {g: {"bitext_intents": GROUPS[g]["bitext"],
                        "examples": examples[g]} for g in GROUPS},
    }
    text = yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=120)
    indented = "\n".join(("  " + line) if line else line for line in text.splitlines())
    header = ("intent:                       # owner: TBD\n"
              f"  # Built by \"training scripts/build_intent_labels.py\": labels and examples from Bitext Insurance + Retail\n"
              f"  # Banking (synthetic, CDLA-Sharing 1.0); min_confidence tuned with {CLINC} (CC BY 3.0).\n"
              "  # Edit GROUPS in that script and rerun, rather than editing this section by hand.\n")
    return header + indented + "\n"


def write_config(section: str):
    text = CONFIG_PATH.read_text()
    # The intent: block runs from "intent:" to the next top-level key.
    new, n = re.subn(r"(?ms)^intent:.*?(?=^\S)", lambda _: section + "\n", text, count=1)
    if n != 1:
        raise SystemExit("Couldn't find the intent: section in config.yaml; paste intent_section.yaml by hand.")
    CONFIG_PATH.write_text(new)


# ----------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading Bitext and CLINC150…")
    bitext = load_bitext(args.bitext_csv)
    check_groups(set(bitext["intent"]))
    clinc = load_clinc(args.clinc_csv)
    print(f"  Bitext (insurance + banking): {len(bitext):,} unique questions, {bitext['intent'].nunique()} intents "
          f"-> {len(GROUPS)} gateway intents")

    examples, bt_tune, bt_test = split_bitext(bitext, args.examples_per_intent, args.heldout_per_intent, args.seed)
    tune = pd.concat([bt_tune, clinc_negatives(clinc, "validation", args.clinc_other_per_intent, args.seed)],
                     ignore_index=True)
    test = pd.concat([bt_test, clinc_negatives(clinc, "test", args.clinc_other_per_intent, args.seed)],
                     ignore_index=True)
    intents_cfg = {g: {"examples": examples[g]} for g in GROUPS}
    for name, d in (("tune", tune), ("test", test)):
        print(f"  {name}: {(d['source'] == 'bitext').sum():,} insurance questions, "
              f"{(d['source'] == 'clinc_oos').sum():,} CLINC150 out-of-scope, "
              f"{(d['source'] == 'clinc_other_domain').sum():,} CLINC150 other-domain")

    print(f"Scoring with the {args.backend} backend…")
    tune_scored = score_all(tune, intents_cfg, args.backend, "tune")
    threshold, sweep = choose_threshold(tune_scored)
    test_scored = score_all(test, intents_cfg, args.backend, "test")
    test_scored["predicted"] = predict(test_scored, threshold)
    r = rates(test_scored, test_scored["predicted"])

    labels = list(GROUPS) + [UNKNOWN]
    report = classification_report(test_scored["true"], test_scored["predicted"], labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(test_scored["true"], test_scored["predicted"], labels=labels),
                      index=[f"true: {l}" for l in labels], columns=labels)

    summary = (f"Backend: {args.backend}   min_confidence (chosen on the tune set): {threshold}\n"
               f"TEST SET RESULTS\n"
               f"  Macro-F1: {r['macro_f1']:.3f}   Accuracy: {r['accuracy']:.3f}\n"
               f"  Insurance + banking questions labelled right: {r['insurance_correct']:.1%}\n"
               f"  Insurance + banking wrongly marked unknown:   {r['insurance_marked_unknown']:.1%}\n"
               f"  CLINC150 out-of-scope correctly unknown:      {r['clinc_oos_caught']:.1%}\n"
               f"  CLINC150 other-domain correctly unknown:      {r['clinc_other_domain_caught']:.1%}\n\n"
               f"{report}")
    print("\n" + summary)

    tag = args.backend
    (RESULTS / f"intent_report_{tag}.txt").write_text(summary)
    sweep.to_csv(RESULTS / f"intent_threshold_sweep_{tag}.csv", index=False)
    cm.to_csv(RESULTS / f"intent_confusion_{tag}.csv")
    test_scored[test_scored["true"] != test_scored["predicted"]].to_csv(
        RESULTS / f"intent_errors_{tag}.csv", index=False)
    section = intent_section(examples, args.backend, threshold)
    (RESULTS / "intent_section.yaml").write_text(section)
    print(f"Saved the report, threshold sweep, confusion matrix and errors to {RESULTS}")

    if args.write_config:
        write_config(section)
        print(f"Updated the intent: section of {CONFIG_PATH}. Next: pytest gateway/tests")
    else:
        print("config.yaml not changed. Rerun with --write-config to apply.")


if __name__ == "__main__":
    main()
