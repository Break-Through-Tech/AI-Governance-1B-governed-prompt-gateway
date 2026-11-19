import pytest

from gateway import settings
from gateway.nodes import cache


@pytest.fixture(autouse=True)
def empty_cache():
    """Every test starts with an empty stub cache."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def risk_stub_backend():
    """Tests use phrase-match screening so they run offline without loading the risk models."""
    with settings.override({"risk": {"backend": "stub"}}):
        yield
