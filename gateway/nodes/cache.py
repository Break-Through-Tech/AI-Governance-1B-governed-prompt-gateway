"""Nodes 6 and 9: cache_lookup and cache_store. Owner: Vanessa."""
import importlib.util
import re
from pathlib import Path
from threading import Lock

from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.nodes.intent import UNKNOWN, extract_question
from gateway.nodes.optimizer import entities

_STORE: dict[str, str] = {}
_SEMANTIC = None
_INIT_LOCK = Lock()
_NEGATIONS = re.compile(r"\b(?:no|not|never|without|cannot|except)\b|n['’]t\b", re.I)


def _key(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _get_semantic():
    """Create the SemanticCache on first use, so the encoder doesn't load at import time."""
    global _SEMANTIC
    with _INIT_LOCK:
        if _SEMANTIC is None:
            path = Path(__file__).resolve().parents[2] / "toxicity-detection-api" / "cache.py"
            spec = importlib.util.spec_from_file_location("semantic_cache_impl", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _SEMANTIC = module.SemanticCache()
        return _SEMANTIC


def clear():
    """Empty both caches without loading the embedding model."""
    with _INIT_LOCK:
        _STORE.clear()
        if _SEMANTIC is not None:
            _SEMANTIC.clear()


def _semantic_query(state):
    original = state["original_prompt"].strip()
    question = extract_question(original)
    context = original[:-len(question)] if question else original
    # Match only within the same original context and intent. Use the original
    # question so compression cannot erase a material difference before lookup.
    intent = state.get("intent", UNKNOWN)
    scope = (context, intent, question if intent == UNKNOWN else None,
             frozenset(entities(question)), tuple(match.lower() for match in _NEGATIONS.findall(question)),
             settings.get("risk")["backend"], settings.generation_simulated())
    return question, scope


def _check_backend():
    backend = settings.get("cache")["backend"]
    if backend not in ("stub", "semantic"):
        raise NotImplementedError(f"cache backend {backend!r} is not supported")
    return backend


# Fail open: a broken cache only costs money, so treat it as a miss.
@safe_node("cache_lookup", fallback=lambda s: {"cache_hit": False})
def cache_lookup(state):
    backend = _check_backend()
    if state.get("risk_label") != "safe":
        return {"cache_hit": False, "decision_log": [decision("skipped", "requires safe screening")]}
    if backend == "semantic":
        prompt, scope = _semantic_query(state)
        hit, score = _get_semantic().lookup_with_score(
            prompt, scope=scope, threshold=settings.get("cache")["min_similarity"]
        )
    else:
        hit = _STORE.get(_key(state["prompt_for_model"]))
        score = 1.0 if hit is not None else 0.0
    result = {"cache_hit": hit is not None}
    if score is not None:
        result["cache_similarity"] = score
    if hit is not None:
        result["response"] = hit
    outcome = "hit" if hit is not None else "miss"
    return {**result, "decision_log": [decision(
        outcome, f"{backend} cache {outcome}", backend=backend, cache_similarity=score
    )]}


@safe_node("cache_store", fallback=lambda s: {})
def cache_store(state):
    backend = _check_backend()
    if state.get("generation_error") or state.get("risk_label") != "safe" or not state.get("response"):
        return {"decision_log": [decision("skipped", "only successful, safe responses are cached")]}
    if backend == "semantic":
        cache = _get_semantic()
        question, scope = _semantic_query(state)
        cache.store(question, state["response"], scope=scope)
        size = len(cache.results)
        reason = "saved response in the semantic cache"
    else:
        _STORE[_key(state["prompt_for_model"])] = state["response"]
        size = len(_STORE)
        reason = "STUB: saved response for exact-match reuse"
    return {"decision_log": [decision("stored", reason, cache_size=size, backend=backend)]}
