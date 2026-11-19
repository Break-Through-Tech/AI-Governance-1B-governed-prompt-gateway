import pytest

from gateway import settings
from gateway.nodes import cache
from gateway.nodes import intent


@pytest.fixture(autouse=True)
def empty_cache():
    """Every test starts with an empty stub cache."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def fake_intent_model(monkeypatch):
    """Node 1 normally downloads its fine-tuned model from Hugging Face (needs torch + transformers).
    Tests use this stand-in instead, so they run offline in seconds. The real model's accuracy comes
    from the report that "training scripts/train_intent.py" writes."""
    def predict(texts):
        return [[0.9, 0.1]], ["insurance.claims.coverage_question", intent.UNKNOWN]
    monkeypatch.setattr(intent, "_finetuned", lambda *args: predict)

@pytest.fixture(autouse=True)
def risk_stub_backend():
    """Tests use phrase-match screening so they run offline without loading the risk models."""
    with settings.override({"risk": {"backend": "stub"}}):
        yield
