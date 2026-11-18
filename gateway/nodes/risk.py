"""Real ML-backed risk screening for the LangGraph gateway.

Uses the same in-process toxicity pipeline as the FastAPI /check endpoint.
A classifier failure is handled by @safe_node as a human redirect (fail closed).
The detector provides a binary decision, not a calibrated probability.
"""
from importlib import import_module
from gateway import settings
from gateway.nodes._base import decision, safe_node

GENERIC_REFUSAL = "I can't help with that request."


def label_for(score: float) -> str:
    cfg = settings.get("risk")
    if score >= cfg["block_threshold"]:
        return "unsafe"
    if score >= cfg["redirect_threshold"]:
        return "redirect"
    return "safe"


@safe_node("screen_risk", fallback=lambda s: {
    "risk_label": "redirect", "risk_categories": ["screening_unavailable"]
})
def screen_risk(state):
    pipeline = import_module("pipeline")
    result = pipeline.run_pipeline(state["original_prompt"])
    if not isinstance(result, dict) or result.get("label") not in ("safe", "unsafe"):
        raise ValueError("Unexpected toxicity model response")

    unsafe = result["label"] == "unsafe"
    label = "unsafe" if unsafe else "safe"
    # The detection model returns a class (0/1), not a probability.
    # Use 0.0/1.0 as binary policy scores; do NOT present these as confidence.
    score = 1.0 if unsafe else 0.0
    category = result.get("category") if unsafe else None
    categories = [category] if isinstance(category, str) and category else []
    category_confidence = result.get("confidence") if unsafe else None
    return {
        "risk_label": label,
        "risk_score": score,
        "risk_categories": categories,
        "decision_log": [decision(label, "ML toxicity detector decision (binary, not calibrated risk probability)",
                                  risk_score=score, score_type="binary_detector_label",
                                  category=category, category_confidence=category_confidence)],
    }


@safe_node("handle_unsafe", fallback=lambda s: {"response": GENERIC_REFUSAL, "final_status": "blocked"})
def handle_unsafe(state):
    label = state.get("risk_label", "unsafe")
    messages = settings.get("risk")["messages"]
    status = "redirected" if label == "redirect" else "blocked"
    return {
        "response": messages["redirect" if label == "redirect" else "unsafe"],
        "final_status": status,
        "decision_log": [decision(status, f"risk_label is {label!r}; prompt never sent to a model",
                                  risk_categories=state.get("risk_categories", []))],
    }
