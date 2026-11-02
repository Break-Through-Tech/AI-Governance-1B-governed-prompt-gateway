import pytest

from gateway.nodes import cache


@pytest.fixture(autouse=True)
def empty_cache():
    """Every test starts with an empty stub cache."""
    cache.clear()
    yield
    cache.clear()
