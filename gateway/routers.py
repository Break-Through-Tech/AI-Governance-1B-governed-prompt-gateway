"""Conditional-edge functions (README §5).

A router only reads state and returns a branch label; graph.py maps each label to a node.
The labels are what appear on the edges in the diagram.
"""
from gateway import settings


def after_risk(state) -> str:
    # Anything other than an explicit "safe" goes to handle_unsafe (fail closed).
    return "safe" if state.get("risk_label") == "safe" else "unsafe / redirect"


def after_fidelity(state) -> str:
    if state.get("fidelity_passed"):
        return "pass"
    if state.get("optimization_attempts", 0) < settings.get("optimizer")["max_attempts"]:
        return "fail, retry left"
    return "fail, no retries"


def after_cache(state) -> str:
    return "hit" if state.get("cache_hit") else "miss"


def after_generate(state) -> str:
    if not state.get("generation_error"):
        return "ok"
    if len(state.get("failed_models", [])) <= settings.get("routing")["max_model_fallbacks"]:
        return "error, fallback left"
    return "error, no fallback"
