"""Node 1: classify_intent. Owner: TBD.

STUB: returns the default intent. Replace the body of `classify_intent` with a real intent
classifier (claims, billing, policy question, ...). Keep the same returned fields.
"""
from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.tokens import count_tokens, estimate_tokens


def _fallback(state):
    return {"intent": "unknown", "intent_confidence": 0.0,
            "original_tokens": estimate_tokens(state.get("original_prompt", ""))}


@safe_node("classify_intent", fallback=_fallback)
def classify_intent(state):
    n = count_tokens(state["original_prompt"])
    intent = settings.get("intent")["default_intent"]
    return {
        "intent": intent,
        "intent_confidence": 0.0,
        "original_tokens": n,
        "decision_log": [decision(intent, "STUB: intent classifier not built yet", original_tokens=n)],
    }
