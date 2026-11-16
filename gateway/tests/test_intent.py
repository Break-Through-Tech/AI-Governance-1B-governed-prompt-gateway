"""classify_intent (node 1): question extraction, labels, the unknown threshold and both backends.

The label tests read the intents from config.yaml, so they keep passing when the label set changes
(e.g. after rebuilding it from Bitext with "training scripts/build_intent_labels.py").
"""
from unittest.mock import patch

import numpy as np
import pytest

from gateway import settings
from gateway.graph import SAMPLE_PROMPT, run
from gateway.nodes import intent as intent_node
from gateway.nodes.intent import UNKNOWN, classify_intent, extract_question

INTENTS = settings.get("intent")["intents"]


@pytest.fixture(autouse=True)
def lexical_backend():
    """These tests check node 1's logic, not a model's accuracy, so they use the dependency-free
    backend whatever config.yaml says. (Accuracy comes from the training scripts' reports.)"""
    with settings.override({"intent": {"backend": "lexical", "min_confidence": 0.40}}):
        yield


def classify(text):
    return classify_intent({"original_prompt": text})


def test_extracts_question_after_header():
    q = extract_question(SAMPLE_PROMPT)
    assert q.startswith("Um, so basically")
    assert "Policy context" not in q and "customer service assistant" not in q


def test_no_header_uses_whole_prompt():
    assert extract_question("  How do I pay my bill?  ") == "How do I pay my bill?"


def test_every_intent_has_examples():
    assert len(INTENTS) >= 2
    for name, spec in INTENTS.items():
        assert len(spec["examples"]) >= 3, name


@pytest.mark.parametrize("name", list(INTENTS))
def test_each_intents_own_example_maps_back_to_it(name):
    # A config example should be its own best match. If this fails, two intents share an example
    # or are too similar to tell apart.
    out = classify(INTENTS[name]["examples"][0])
    assert out["intent"] == name
    assert 0.0 < out["intent_confidence"] <= 1.0


@pytest.mark.parametrize("text", ["write me a poem about the ocean", "asdf qwer", ""])
def test_off_topic_is_unknown(text):
    out = classify(text)
    assert out["intent"] == UNKNOWN
    assert out["intent_confidence"] < settings.get("intent")["min_confidence"]


def test_decision_log_explains_the_choice():
    name = next(iter(INTENTS))
    entry = classify(INTENTS[name]["examples"][0])["decision_log"][0]
    assert entry["decision"] == name
    assert {"backend", "intent_confidence", "runner_up", "runner_up_confidence"} <= set(entry["metrics"])


def test_threshold_comes_from_config():
    example = next(iter(INTENTS.values()))["examples"][0]
    with settings.override({"intent": {"min_confidence": 1.01}}):
        assert classify(example)["intent"] == UNKNOWN


TINY_INTENTS = {
    "billing.payment": {"examples": ["How do I pay my bill?", "Pay my premium"]},
    "account.access": {"examples": ["I forgot my password", "Reset my password"]},
}


class FakeEncoder:
    """Stands in for sentence-transformers: one dimension per word in a tiny vocabulary."""
    VOCAB = ["pay", "password"]

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for t in texts:
            v = np.array([float(w in t.lower()) for w in self.VOCAB]) + 1e-6
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


def test_embedding_backend_loads_lazily_and_scores():
    # Scored directly with TINY_INTENTS: settings.override merges dicts, so it can't remove the
    # config's own intents.
    intent_node._encoder.cache_clear()
    intent_node._example_matrix.cache_clear()
    with patch.object(intent_node, "_encoder", return_value=FakeEncoder()) as enc:
        scores = intent_node._embedding_scores("help, I lost my password", TINY_INTENTS, "fake-model")
    assert enc.called
    assert max(scores, key=scores.get) == "account.access"
    assert set(scores) == set(TINY_INTENTS)


def test_embedding_backend_is_used_by_the_node():
    intent_node._encoder.cache_clear()
    intent_node._example_matrix.cache_clear()
    with settings.override({"intent": {"backend": "embedding"}}), \
         patch.object(intent_node, "_encoder", return_value=FakeEncoder()):
        out = classify("How do I pay my bill?")
    assert out["decision_log"][0]["metrics"]["backend"] == "embedding"
    assert not out.get("errors")


def test_finetuned_backend_uses_model_probabilities():
    calls = []

    def fake_predict(texts):
        calls.append(texts)
        return np.array([[0.1, 0.9]]), ["billing.payment", "account.access"]

    intent_node._finetuned.cache_clear()
    with settings.override({"intent": {"backend": "finetuned", "min_confidence": 0.5}}), \
         patch.object(intent_node, "_finetuned", return_value=fake_predict) as loader:
        out = classify("### Question\nI lost my password")
    assert loader.call_args.args == (settings.get("intent")["finetuned_model"],
                                     settings.get("intent")["finetuned_max_length"])
    assert calls == [["I lost my password"]]              # only the question reaches the model
    assert out["intent"] == "account.access" and out["intent_confidence"] == 0.9
    with settings.override({"intent": {"backend": "finetuned", "min_confidence": 0.95}}), \
         patch.object(intent_node, "_finetuned", return_value=fake_predict):
        assert classify("I lost my password")["intent"] == UNKNOWN


def test_finetuned_unknown_class_wins_even_when_confident():
    def fake_predict(texts):
        return np.array([[0.05, 0.92, 0.03]]), ["banking.transfer", UNKNOWN, "insurance.claims.status"]

    with settings.override({"intent": {"backend": "finetuned", "min_confidence": 0.5}}), \
         patch.object(intent_node, "_finetuned", return_value=fake_predict):
        out = classify("What's the weather in Paris tomorrow?")
    assert out["intent"] == UNKNOWN and out["intent_confidence"] == 0.92
    assert "model predicted unknown" in out["decision_log"][0]["reason"]


def test_unknown_backend_fails_into_the_fallback():
    with settings.override({"intent": {"backend": "nope"}}):
        out = classify("How do I pay my bill?")
    assert out["intent"] == UNKNOWN and out["errors"]


def test_full_graph_carries_the_intent():
    name, spec = next(iter(INTENTS.items()))
    r = run(spec["examples"][0])
    assert r["intent"] == name
    assert r["decision_log"][0]["node"] == "classify_intent"
