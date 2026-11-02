"""Nodes 2–3: screen_risk and handle_unsafe. Owner: Juee / Daniel.

screen_risk is a STUB that flags phrases from config.yaml. The real version should wrap
toxicity-detection-api/pipeline.py: use the detector's softmax probability as `risk_score`
and the classifier's category as `risk_categories`. Load the models lazily (on first call),
not at import time, so tests and the rest of the graph don't download 1.4 GB.
"""
from gateway import settings
from gateway.nodes._base import decision, safe_node

GENERIC_REFUSAL = "I can't help with that request."


def label_for(score: float) -> str:
    """Turn a risk score into a label using the thresholds in config.yaml (README §5)."""
    cfg = settings.get("risk")
    if score >= cfg["block_threshold"]:
        return "unsafe"
    if score >= cfg["redirect_threshold"]:
        return "redirect"
    return "safe"


# Fail closed: if screening breaks, a human handles the query; it never reaches a model.
@safe_node("screen_risk", fallback=lambda s: {"risk_label": "redirect",
                                              "risk_categories": ["screening_unavailable"]})
def screen_risk(state):
    cfg = settings.get("risk")
    text = state["original_prompt"].lower()
    blocked = [p for p in cfg["stub_block_phrases"] if p in text]
    redirected = [p for p in cfg["stub_redirect_phrases"] if p in text]
    score = 0.95 if blocked else 0.60 if redirected else 0.02
    label = label_for(score)
    return {
        "risk_label": label,
        "risk_score": score,
        "risk_categories": [f"stub:{p}" for p in blocked + redirected],
        "decision_log": [decision(label, f"STUB: phrase match, risk_score {score:.2f}",
                                  risk_score=score, matched=blocked + redirected)],
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
