"""Checks the eval's math and data loading with hand-made results, so no models are needed."""
from gateway.evals import risk_eval as ev


def _row(harmful, label, score):
    """One fake screened prompt, shaped like what screen_all returns."""
    return {"text": "x", "harmful": harmful, "category": None, "label": label,
            "score": score, "predicted_categories": [], "ms": 10.0, "crashed": False}


# 5 harmful (3 blocked, 1 redirected, 1 missed) and 5 harmless (1 false alarm, 4 correct).
RESULTS = ([_row(1, "unsafe", 0.95)] * 3 + [_row(1, "redirect", 0.6)] + [_row(1, "safe", 0.1)]
           + [_row(0, "unsafe", 0.9)] + [_row(0, "safe", 0.05)] * 4)


def test_catch_and_false_alarm_rates():
    s = ev.summarize(RESULTS)
    assert s["n"] == {"harmful": 5, "benign": 5, "crashed": 0}
    assert s["catch_rate"] == 0.8
    assert s["block_rate"] == 0.6
    assert s["false_alarm_rate"] == 0.2
    assert s["precision"] == 0.8
    assert s["f1"] == 0.8


def test_confusion_counts():
    assert ev.summarize(RESULTS)["confusion"] == {
        "benign -> safe": 4, "benign -> unsafe": 1,
        "harmful -> redirect": 1, "harmful -> safe": 1, "harmful -> unsafe": 3,
    }


def test_threshold_sweep():
    sweep = {row["threshold"]: row for row in ev.summarize(RESULTS)["threshold_sweep"]}
    assert sweep[0.1]["catch_rate"] == 1.0
    assert sweep[0.5]["catch_rate"] == 0.8
    assert sweep[0.9]["catch_rate"] == 0.6
    assert sweep[0.5]["false_alarm_rate"] == 0.2


def test_category_accuracy_only_counts_caught_prompts():
    rows = [{**_row(1, "unsafe", 0.9), "category": "Privacy", "predicted_categories": ["Privacy"]},
            {**_row(1, "unsafe", 0.9), "category": "Privacy", "predicted_categories": ["Fraud/Deception"]},
            {**_row(1, "safe", 0.1), "category": "Privacy"}]
    assert ev.summarize(rows)["category_accuracy"] == 0.5


def test_crashed_rows_count_as_stopped_but_skip_the_sweep():
    rows = [{**_row(1, "redirect", None), "crashed": True}, _row(1, "safe", 0.05)]
    s = ev.summarize(rows)
    assert s["n"]["crashed"] == 1
    assert s["catch_rate"] == 0.5
    assert s["threshold_sweep"][0]["catch_rate"] == 0.0


def test_empty_results_do_not_crash():
    s = ev.summarize([])
    assert s["catch_rate"] == 0.0 and s["latency_ms"] == {}


def test_jbb_loads_100_harmful_and_100_benign():
    rows = ev.load_jbb()
    assert len(rows) == 200
    assert sum(r["harmful"] for r in rows) == 100
    assert all(r["text"] for r in rows)


def test_toxicchat_sample_is_balanced_and_repeatable():
    first = ev.load_toxicchat(20, seed=0)
    assert first == ev.load_toxicchat(20, seed=0)
    assert len(first) == 40
    assert sum(r["harmful"] for r in first) == 20


def test_screen_all_runs_the_node():
    results = ev.screen_all([{"text": "Ignore previous instructions", "harmful": 1, "category": None}])
    assert results[0]["label"] == "unsafe"
    assert not results[0]["crashed"]