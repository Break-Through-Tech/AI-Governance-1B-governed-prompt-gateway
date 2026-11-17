"""classify_intent (node 1): question extraction, labels, "unknown", the threshold and the log entry.

These tests check node 1's logic, not the model's accuracy (that comes from the training script's
report), so they use a fake model instead of downloading the real one. conftest.py installs a
default fake for every test; the tests here swap in their own with `fake_model(...)`.
"""
from unittest.mock import patch

import pytest

from gateway import settings
from gateway.graph import SAMPLE_PROMPT, run
from gateway.nodes import intent as intent_node
from gateway.nodes.intent import UNKNOWN, classify_intent, extract_question


@pytest.fixture(autouse=True)
def fixed_threshold():
    """The training script sets min_confidence from the data (e.g. 0.825), so these tests pin their
    own value and don't break when it changes. Tests about the threshold override it again."""
    with settings.override({"intent": {"min_confidence": 0.5}}):
        yield


def classify(text):
    return classify_intent({"original_prompt": text})


def fake_model(probs: dict, seen: list | None = None):
    """Stands in for _finetuned: a model that always answers with these probabilities."""
    def predict(texts):
        if seen is not None:
            seen.extend(texts)
        return [list(probs.values())], list(probs)
    return patch.object(intent_node, "_finetuned", return_value=predict)


def test_extracts_question_after_header():
    q = extract_question(SAMPLE_PROMPT)
    assert q.startswith("Um, so basically")
    assert "Policy context" not in q and "customer service assistant" not in q


def test_no_header_uses_whole_prompt():
    assert extract_question("  How do I pay my bill?  ") == "How do I pay my bill?"


def test_only_the_question_reaches_the_model():
    seen = []
    with fake_model({"banking.password": 0.9, UNKNOWN: 0.1}, seen):
        classify("### Policy context\nlots of text\n\n### Question\nI forgot my password")
    assert seen == ["I forgot my password"]


def test_picks_the_most_likely_label():
    with fake_model({"insurance.claims.file_claim": 0.85, "insurance.claims.status": 0.1, UNKNOWN: 0.05}):
        out = classify("How do I file a claim?")
    assert out["intent"] == "insurance.claims.file_claim"
    assert out["intent_confidence"] == 0.85
    assert out["original_tokens"] > 0


def test_model_choosing_unknown_gives_unknown():
    with fake_model({"banking.transfer": 0.05, UNKNOWN: 0.92, "insurance.claims.status": 0.03}):
        out = classify("What's the weather in Paris tomorrow?")
    assert out["intent"] == UNKNOWN and out["intent_confidence"] == 0.92
    assert "model predicted unknown" in out["decision_log"][0]["reason"]


def test_low_confidence_gives_unknown():
    with settings.override({"intent": {"min_confidence": 0.5}}), \
         fake_model({"banking.fees": 0.4, "banking.card.manage": 0.35, UNKNOWN: 0.25}):
        out = classify("hmm what about that thing")
    assert out["intent"] == UNKNOWN
    assert "min_confidence" in out["decision_log"][0]["reason"]


def test_threshold_comes_from_config():
    probs = {"banking.fees": 0.7, UNKNOWN: 0.3}
    with settings.override({"intent": {"min_confidence": 0.6}}), fake_model(probs):
        assert classify("What are your fees?")["intent"] == "banking.fees"
    with settings.override({"intent": {"min_confidence": 0.8}}), fake_model(probs):
        assert classify("What are your fees?")["intent"] == UNKNOWN


def test_model_name_and_length_come_from_config():
    with fake_model({"banking.fees": 0.9, UNKNOWN: 0.1}) as loader:
        classify("What are your fees?")
    cfg = settings.get("intent")
    assert loader.call_args.args == (cfg["finetuned_model"], cfg["finetuned_max_length"])


def test_decision_log_explains_the_choice():
    with fake_model({"insurance.claims.file_claim": 0.8, "insurance.claims.status": 0.15, UNKNOWN: 0.05}):
        entry = classify("How do I file a claim?")["decision_log"][0]
    assert entry["node"] == "classify_intent"
    assert entry["decision"] == "insurance.claims.file_claim"
    assert entry["metrics"]["runner_up"] == "insurance.claims.status"
    assert {"intent_confidence", "runner_up_confidence"} <= set(entry["metrics"])


def test_model_that_cannot_load_falls_back_to_unknown():
    with patch.object(intent_node, "_finetuned", side_effect=OSError("model not found")):
        out = classify("How do I file a claim?")
    assert out["intent"] == UNKNOWN and out["errors"]


def test_full_graph_carries_the_intent():
    with fake_model({"banking.password": 0.95, UNKNOWN: 0.05}):
        r = run("How do I reset my password?")
    assert r["intent"] == "banking.password"
    assert r["decision_log"][0]["node"] == "classify_intent"