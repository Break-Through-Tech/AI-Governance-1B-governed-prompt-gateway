"""Nodes 6 and 9: cache_lookup and cache_store. Owner: Vanessa.

cache_lookup: checks whether a semantically similar prompt has been answered before.
cache_store:  saves a new response so future similar prompts can get a cache hit.

When cache.backend is "stub" (default) an in-memory exact-match dict is used so the graph
runs without any dependencies. Set cache.backend to "semantic" in config.yaml to use the
real FAISS-backed SemanticCache from toxicity-detection-api/cache.py.

The cache only ever sees prompts that passed risk screening (the graph runs it after screen_risk),
so a cached safe answer is never served to an unsafe prompt.
"""
import functools
import sys
from pathlib import Path

from gateway import settings
from gateway.nodes._base import decision, safe_node

_STORE: dict[str, str] = {}


def _key(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def clear():
    """Empty the stub cache (used by tests)."""
    _STORE.clear()


@functools.lru_cache(maxsize=1)
def _get_semantic():
    """Return the shared SemanticCache instance (created on first call)."""
    _root = Path(__file__).resolve().parents[2] / "toxicity-detection-api"
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from cache import SemanticCache  # noqa: PLC0415
    cfg = settings.get("cache")
    return SemanticCache(threshold=cfg["min_similarity"])


# Fail open: a broken cache only costs money, so treat it as a miss.
@safe_node("cache_lookup", fallback=lambda s: {"cache_hit": False})
def cache_lookup(state):
    prompt = state["prompt_for_model"]
    backend = settings.get("cache")["backend"]

    if backend == "semantic":
        response, score = _get_semantic().lookup(prompt)
        if response is not None:
            return {"cache_hit": True, "cache_similarity": score, "response": response,
                    "decision_log": [decision("hit", "semantic cache hit; skipping the model",
                                              cache_similarity=score)]}
        return {"cache_hit": False, "cache_similarity": score,
                "decision_log": [decision("miss", f"best similarity {score:.3f} below threshold",
                                          cache_similarity=score)]}

    # stub backend
    hit = _STORE.get(_key(prompt))
    if hit is None:
        return {"cache_hit": False, "cache_similarity": 0.0,
                "decision_log": [decision("miss", "STUB: no exact match")]}
    return {"cache_hit": True, "cache_similarity": 1.0, "response": hit,
            "decision_log": [decision("hit", "STUB: exact match; skipping the model", cache_similarity=1.0)]}


@safe_node("cache_store", fallback=lambda s: {})
def cache_store(state):
    if state.get("generation_error") or state.get("risk_label") != "safe" or not state.get("response"):
        return {"decision_log": [decision("skipped", "only successful, safe responses are cached")]}
    prompt, response = state["prompt_for_model"], state["response"]
    backend = settings.get("cache")["backend"]

    if backend == "semantic":
        _get_semantic().store(prompt, response)
        return {"decision_log": [decision("stored", "saved to semantic cache")]}

    # stub backend
    _STORE[_key(prompt)] = response
    return {"decision_log": [decision("stored", "STUB: saved response for exact-match reuse",
                                      cache_size=len(_STORE))]}
