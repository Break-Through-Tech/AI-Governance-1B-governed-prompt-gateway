"""Nodes 2–3: screen_risk and handle_unsafe. Owner: Juee / Daniel.

screen_risk wraps toxicity-detection-api/pipeline.py. The detector's softmax probability for the
toxic class is `risk_score`, and the classifier's category is `risk_categories` (only for prompts
that are not safe). `risk.backend` in config.yaml picks `model` (real) or `stub` (phrase match,
used by tests). The models load on the first call, not at import time.
"""
import importlib.util
from functools import lru_cache
from pathlib import Path

from gateway import settings
from gateway.nodes._base import decision, safe_node

PIPELINE_PATH = Path(__file__).resolve().parents[2] / "toxicity-detection-api" / "pipeline.py"
GENERIC_REFUSAL = "I can't help with that request."


@lru_cache(maxsize=1)
def _pipeline():
    """Load toxicity-detection-api/pipeline.py on first use. Running it loads all three models."""
    spec = importlib.util.spec_from_file_location("toxicity_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toxic_probability(text: str) -> float:
    """Softmax probability the detector gives the toxic class (index 1), from 0 to 1."""
    import torch

    p = _pipeline()
    with torch.no_grad():
        enc = p.det_tokenizer([text], truncation=True, max_length=512, return_tensors="pt")
        logits = p.det_model(**enc).logits
    return float(torch.softmax(logits, dim=-1)[0, 1])


def label_for(score: float) -> str:
    """Turn a risk score into a label using the thresholds in config.yaml (README §5)."""
    cfg = settings.get("risk")
    if score >= cfg["block_threshold"]:
        return "unsafe"
    if score >= cfg["redirect_threshold"]:
        return "redirect"
    return "safe"


def _stub_score(text: str, cfg: dict) -> tuple[float, list[str], dict]:
    """Phrase match against the stub lists in config.yaml. Used by tests so they run offline."""
    lowered = text.lower()
    blocked = [p for p in cfg["stub_block_phrases"] if p in lowered]
    redirected = [p for p in cfg["stub_redirect_phrases"] if p in lowered]
    score = 0.95 if blocked else 0.60 if redirected else 0.02
    return score, [f"stub:{p}" for p in blocked + redirected], {}


def _model_score(text: str, cfg: dict) -> tuple[float, list[str], dict]:
    """Detector probability as the score; classifier category only when the prompt is not safe."""
    score = _toxic_probability(text)
    if label_for(score) == "safe":
        return score, [], {}
    result = _pipeline().classify(text)
    return score, [result["category"]], {"category_confidence": result["confidence"]}


SCORERS = {"stub": _stub_score, "model": _model_score}


def _reason(backend: str, score: float, label: str, categories: list[str], cfg: dict) -> str:
    """One readable line for the audit log explaining which threshold decided the label."""
    if label == "unsafe":
        rule = f">= block threshold {cfg['block_threshold']:.2f}"
    elif label == "redirect":
        rule = f">= redirect threshold {cfg['redirect_threshold']:.2f}"
    else:
        rule = f"< redirect threshold {cfg['redirect_threshold']:.2f}"
    tail = f"; {', '.join(categories)}" if categories else ""
    return f"{backend}: risk_score {score:.2f} {rule}{tail}"


# Fail closed: if screening breaks, a human handles the query; it never reaches a model.
@safe_node("screen_risk", fallback=lambda s: {"risk_label": "redirect",
                                              "risk_categories": ["screening_unavailable"]})
def screen_risk(state):
    """Score the original prompt, label it with the config thresholds, and log one decision."""
    cfg = settings.get("risk")
    backend = cfg.get("backend", "stub")
    if backend not in SCORERS:
        raise ValueError(f"unknown risk.backend {backend!r}")
    score, categories, extra = SCORERS[backend](state["original_prompt"], cfg)
    label = label_for(score)
    return {
        "risk_label": label,
        "risk_score": round(score, 4),
        "risk_categories": categories,
        "decision_log": [decision(label, _reason(backend, score, label, categories, cfg),
                                  risk_score=round(score, 4), backend=backend, **extra)],
    }


@safe_node("handle_unsafe", fallback=lambda s: {"response": GENERIC_REFUSAL, "final_status": "blocked"})
def handle_unsafe(state):
    """Return the block or redirect message from config.yaml; the prompt never reaches a model."""
    label = state.get("risk_label", "unsafe")
    messages = settings.get("risk")["messages"]
    status = "redirected" if label == "redirect" else "blocked"
    return {
        "response": messages["redirect" if label == "redirect" else "unsafe"],
        "final_status": status,
        "decision_log": [decision(status, f"risk_label is {label!r}; prompt never sent to a model",
                                  risk_categories=state.get("risk_categories", []))],
    }
