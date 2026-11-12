"""Nodes 6 and 9: cache_lookup and cache_store. Owner: Vanessa.

STUB: an in-memory exact-match cache (after normalizing case and whitespace). The real version
should wrap SemanticCache from toxicity-detection-api/cache.py, with two small changes:
`lookup` should also return the similarity score (the UI shows it), and the encoder should load
lazily. Set cache.backend to "semantic" in config.yaml once it's wired in.

The cache only ever sees prompts that passed risk screening (the graph runs it after screen_risk),
so a cached safe answer is never served to an unsafe prompt.
"""
import importlib.util
from pathlib import Path

from gateway import settings
from gateway.nodes._base import decision, safe_node

_STORE: dict[str, str] = {}
_SEMANTIC = None


def _key(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _get_semantic():
    """Create the SemanticCache on first use, so the encoder doesn't load at import time."""
    global _SEMANTIC
    if _SEMANTIC is None:
        path = Path(__file__).resolve().parents[2] / "toxicity-detection-api" / "cache.py"
        spec = importlib.util.spec_from_file_location("semantic_cache_impl", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SEMANTIC = module.SemanticCache(threshold=settings.get("cache")["min_similarity"])
    return _SEMANTIC


def clear():
    """Empty the stub cache (used by tests)."""
    _STORE.clear()


def _check_backend():
    backend = settings.get("cache")["backend"]
    if backend not in ("stub", "semantic"):
        raise NotImplementedError(f"cache backend {backend!r} is not supported")
    return backend


# Fail open: a broken cache only costs money, so treat it as a miss.
@safe_node("cache_lookup", fallback=lambda s: {"cache_hit": False})
def cache_lookup(state):
    _check_backend()
    hit = _STORE.get(_key(state["prompt_for_model"]))
    if hit is None:
        return {"cache_hit": False, "cache_similarity": 0.0,
                "decision_log": [decision("miss", "STUB: no exact match")]}
    return {"cache_hit": True, "cache_similarity": 1.0, "response": hit,
            "decision_log": [decision("hit", "STUB: exact match; skipping the model", cache_similarity=1.0)]}


@safe_node("cache_store", fallback=lambda s: {})
def cache_store(state):
    backend = _check_backend()
    if state.get("generation_error") or state.get("risk_label") != "safe" or not state.get("response"):
        return {"decision_log": [decision("skipped", "only successful, safe responses are cached")]}
    prompt = state["prompt_for_model"]
    if backend == "semantic":
        cache = _get_semantic()
        cache.store(prompt, state["response"])
        size = len(cache.results)
        reason = "saved response in the semantic cache"
    else:
        _STORE[_key(prompt)] = state["response"]
        size = len(_STORE)
        reason = "STUB: saved response for exact-match reuse"
    return {"decision_log": [decision("stored", reason, cache_size=size, backend=backend)]}