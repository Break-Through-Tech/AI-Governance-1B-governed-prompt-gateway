import pytest

from gateway import settings
from gateway.nodes import cache, intent


@pytest.fixture(autouse=True)
def empty_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def fake_intent_model(monkeypatch):
    """Use a stand-in intent model so tests run offline."""
    def predict(texts):
        return [[0.9, 0.1]], [
            "insurance.claims.coverage_question",
            intent.UNKNOWN,
        ]

    monkeypatch.setattr(intent, "_finetuned", lambda *args: predict)


@pytest.fixture(autouse=True)
def offline_backends():
    """Use deterministic risk, cache, and generation backends during tests."""
    with settings.override({"risk": {"backend": "stub"}, "cache": {"backend": "stub"},
                            "generation": {"simulate": True}}):
        yield
