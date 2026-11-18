"""screen_risk's model backend, with the detector and classifier replaced by fakes."""
from types import SimpleNamespace
from unittest.mock import patch

from gateway import settings
from gateway.nodes.risk import screen_risk

FAKE_PIPELINE = SimpleNamespace(classify=lambda text: {"category": "Malware/Hacking", "confidence": 0.9})


def _screen(score):
    with settings.override({"risk": {"backend": "model"}}), \
         patch("gateway.nodes.risk._toxic_probability", return_value=score), \
         patch("gateway.nodes.risk._pipeline", return_value=FAKE_PIPELINE):
        return screen_risk({"original_prompt": "anything"})


def test_high_score_is_unsafe_with_category():
    out = _screen(0.91)
    assert out["risk_label"] == "unsafe"
    assert out["risk_categories"] == ["Malware/Hacking"]


def test_middle_score_is_redirect():
    assert _screen(0.65)["risk_label"] == "redirect"


def test_low_score_is_safe_and_skips_classifier():
    out = _screen(0.03)
    assert out["risk_label"] == "safe"
    assert out["risk_categories"] == []


def test_unknown_backend_fails_closed():
    with settings.override({"risk": {"backend": "typo"}}):
        out = screen_risk({"original_prompt": "anything"})
    assert out["risk_label"] == "redirect"
    assert out["errors"]
