"""Tests for when there is no model left to try."""
from unittest.mock import patch

from gateway import routers, settings
from gateway.graph import run
from gateway.nodes import generate
from gateway.nodes.routing import route_model

FAILED_MESSAGE = settings.get("audit")["messages"]["failed"]
QUESTION = "Is water damage from a burst pipe covered?"


def test_after_route():
    assert routers.after_route({"model_id": "gpt-oss-120b"}) == "model chosen"
    assert routers.after_route({"model_id": ""}) == "no eligible model"


def test_route_model_with_every_model_failed():
    all_models = [m["id"] for m in settings.get("routing")["models"]]
    state = {"intent": "unknown", "risk_score": 0.0, "failed_models": all_models}

    result = route_model(state)

    assert result["model_id"] == ""
    assert result["model_tier"] is None
    assert result["decision_log"][0]["decision"] == "no_eligible_model"


def test_every_model_fails(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    all_models = [m["id"] for m in settings.get("routing")["models"]]

    with settings.override({"generation": {"simulate_fail_models": all_models},
                            "routing": {"max_model_fallbacks": 99}}):
        result = run(QUESTION)

    assert result["final_status"] == "failed"
    assert result["response"] == FAILED_MESSAGE

    # each model should only be tried once
    assert len(result["failed_models"]) == len(set(result["failed_models"]))

    # the last two steps should be route_model and then finalize_audit
    nodes = [entry["node"] for entry in result["decision_log"]]
    assert nodes[-2] == "route_model"
    assert nodes[-1] == "finalize_audit"


def test_routing_crash_skips_generation():
    with patch("gateway.nodes.routing.pick_model", side_effect=ValueError("bad config")):
        result = run(QUESTION)

    nodes = [entry["node"] for entry in result["decision_log"]]
    assert "generate_response" not in nodes
    assert result["final_status"] == "failed"
    assert result["response"] == FAILED_MESSAGE


def test_no_api_keys(monkeypatch):
    for key_name, url in generate.PROVIDERS.values():
        monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv("GATEWAY_SIMULATE_GENERATION", raising=False)

    with settings.override({"generation": {"simulate": False}}):
        result = run(QUESTION)

    nodes = [entry["node"] for entry in result["decision_log"]]
    assert "generate_response" not in nodes
    assert result["final_status"] == "failed"
    assert result["response"] == FAILED_MESSAGE


def test_answered_query_has_no_failed_message():
    result = run(QUESTION)
    assert result["final_status"] == "answered"
    assert result["response"] != FAILED_MESSAGE
