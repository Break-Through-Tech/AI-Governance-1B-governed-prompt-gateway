"""Measures how well screen_risk separates harmful prompts from harmless ones.

Run from the repo root (needs the risk model dependencies):
    python -m gateway.evals.risk_eval
    python -m gateway.evals.risk_eval --dataset jbb
    python -m gateway.evals.risk_eval --per-class 50

Datasets, read straight from the zips in data/:
- jbb: JailbreakBench, 100 harmful requests and 100 harmless look-alikes. Polite wording the detector
  never trained on. The classifier did train on it, so category accuracy here is optimistic.
- toxicchat: a balanced sample of the ToxicChat test split. Held out, but the same kind of data the
  detector learned from.
"""
import argparse
import csv
import io
import json
import random
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from gateway import settings
from gateway.nodes.risk import screen_risk

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"
SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _read_zip_csv(zip_name: str, member: str) -> list[dict]:
    """Read one CSV out of a zip in data/, so nobody has to unzip anything first."""
    csv.field_size_limit(10**9)
    with zipfile.ZipFile(DATA / zip_name) as z, z.open(member) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


def load_jbb() -> list[dict]:
    """JailbreakBench harmful requests (harmful=1) and their harmless look-alikes (harmful=0)."""
    harmful = _read_zip_csv("jailbreak.zip", "jailbreak/behaviors_harmful.csv")
    benign = _read_zip_csv("jailbreak.zip", "jailbreak/behaviors_benign.csv")
    return ([{"text": r["Goal"], "harmful": 1, "category": r["Category"]} for r in harmful]
            + [{"text": r["Goal"], "harmful": 0, "category": None} for r in benign])


def load_toxicchat(per_class: int, seed: int) -> list[dict]:
    """A balanced random sample of the ToxicChat test split, so a run takes minutes instead of an hour."""
    rows = [{"text": r["user_input"], "harmful": int(r["toxicity"]), "category": None}
            for r in _read_zip_csv("lmsys_toxicchat.zip", "lmsys_toxicchat/test.csv")]
    rng = random.Random(seed)
    toxic = [r for r in rows if r["harmful"]]
    clean = [r for r in rows if not r["harmful"]]
    return rng.sample(toxic, min(per_class, len(toxic))) + rng.sample(clean, min(per_class, len(clean)))


def screen_all(rows: list[dict]) -> list[dict]:
    """Run screen_risk on every prompt and keep what it decided and how long it took."""
    screen_risk({"original_prompt": "warm-up so model loading isn't counted as latency"})
    results = []
    for i, row in enumerate(rows, 1):
        start = time.perf_counter()
        out = screen_risk({"original_prompt": row["text"]})
        results.append({**row,
                        "label": out["risk_label"],
                        "score": out.get("risk_score"),
                        "predicted_categories": out.get("risk_categories", []),
                        "ms": (time.perf_counter() - start) * 1000,
                        "crashed": bool(out.get("errors"))})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")
    return results


def stopped(r: dict) -> bool:
    """Anything not labeled safe never reaches a model, so it counts as stopped."""
    return r["label"] != "safe"


def _rate(rows: list[dict], test) -> float:
    """Share of rows that pass the test, rounded for the report."""
    return round(sum(1 for r in rows if test(r)) / len(rows), 3) if rows else 0.0


def summarize(results: list[dict]) -> dict:
    """Turn per-prompt results into the numbers we report: catch rate, false alarms, sweep, latency."""
    harmful = [r for r in results if r["harmful"]]
    benign = [r for r in results if not r["harmful"]]
    caught = sum(1 for r in harmful if stopped(r))
    false_alarms = sum(1 for r in benign if stopped(r))
    precision = caught / (caught + false_alarms) if caught + false_alarms else 0.0
    recall = caught / len(harmful) if harmful else 0.0
    scored = [r for r in results if r["score"] is not None]
    categorized = [r for r in harmful if r["category"] and stopped(r) and not r["crashed"]]
    latency = sorted(r["ms"] for r in results)
    return {
        "n": {"harmful": len(harmful), "benign": len(benign), "crashed": sum(r["crashed"] for r in results)},
        "catch_rate": round(recall, 3),
        "block_rate": _rate(harmful, lambda r: r["label"] == "unsafe"),
        "false_alarm_rate": _rate(benign, stopped),
        "precision": round(precision, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "confusion": {f"{'harmful' if h else 'benign'} -> {label}": n
                      for (h, label), n in sorted(Counter((r["harmful"], r["label"]) for r in results).items())},
        "threshold_sweep": [
            {"threshold": t,
             "catch_rate": _rate([r for r in scored if r["harmful"]], lambda r: r["score"] >= t),
             "false_alarm_rate": _rate([r for r in scored if not r["harmful"]], lambda r: r["score"] >= t)}
            for t in SWEEP],
        "category_accuracy": (_rate(categorized, lambda r: r["category"] in r["predicted_categories"])
                              if categorized else None),
        "latency_ms": ({"p50": round(latency[len(latency) // 2], 1),
                        "p95": round(latency[max(int(len(latency) * 0.95) - 1, 0)], 1)} if latency else {}),
        "missed_examples": [r["text"][:120] for r in harmful if not stopped(r)][:15],
        "false_alarm_examples": [r["text"][:120] for r in benign if stopped(r)][:15],
    }


def print_summary(name: str, s: dict):
    """Short readable version of one dataset's results for the terminal."""
    print(f"\n=== {name}  ({s['n']['harmful']} harmful, {s['n']['benign']} harmless, {s['n']['crashed']} crashed) ===")
    print(f"catch rate {s['catch_rate']:.0%}  (blocked outright {s['block_rate']:.0%})  ·  "
          f"false alarms {s['false_alarm_rate']:.0%}  ·  precision {s['precision']:.0%}  ·  F1 {s['f1']:.2f}")
    if s["category_accuracy"] is not None:
        print(f"category accuracy on caught prompts: {s['category_accuracy']:.0%}")
    print(f"latency p50 {s['latency_ms'].get('p50')} ms, p95 {s['latency_ms'].get('p95')} ms")
    print("confusion:", s["confusion"])
    print("threshold   catch   false alarms")
    for row in s["threshold_sweep"]:
        print(f"  {row['threshold']:.1f}      {row['catch_rate']:>4.0%}   {row['false_alarm_rate']:>4.0%}")
    for text in s["missed_examples"][:5]:
        print(f"  missed: {text}")


def main():
    """Evaluate screen_risk on the chosen datasets, print a summary, and save it to gateway/evals/results/."""
    parser = argparse.ArgumentParser(description="Evaluate the screen_risk node on labeled prompts.")
    parser.add_argument("--dataset", choices=["all", "jbb", "toxicchat"], default="all")
    parser.add_argument("--per-class", type=int, default=150, help="ToxicChat prompts per class")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = {}
    if args.dataset in ("all", "jbb"):
        datasets["jbb"] = load_jbb()
    if args.dataset in ("all", "toxicchat"):
        datasets["toxicchat"] = load_toxicchat(args.per_class, args.seed)

    cfg = settings.get("risk")
    report = {"run_at": datetime.now().isoformat(timespec="seconds"),
              "thresholds": {"block": cfg["block_threshold"], "redirect": cfg["redirect_threshold"]},
              "seed": args.seed, "per_class": args.per_class, "datasets": {}}
    with settings.override({"risk": {"backend": "model"}}):
        for name, rows in datasets.items():
            print(f"\n{name}: screening {len(rows)} prompts")
            report["datasets"][name] = summarize(screen_all(rows))

    for name, s in report["datasets"].items():
        print_summary(name, s)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"risk_eval_{datetime.now():%Y%m%d_%H%M}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()