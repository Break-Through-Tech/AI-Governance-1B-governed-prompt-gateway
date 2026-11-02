"""Every node only writes the fields it owns (state.WRITES / README §3), and logs exactly one decision."""
import pytest

from gateway.graph import SAMPLE_PROMPT
from gateway.nodes import NODES
from gateway.state import ALWAYS_ALLOWED, WRITES

# A state with every field a node might read, as if all earlier nodes had run.
FULL_STATE = {
    "query_id": "Q-test",
    "original_prompt": SAMPLE_PROMPT,
    "started_at": 0.0,
    "intent": "general", "intent_confidence": 0.0, "original_tokens": 239,
    "risk_label": "safe", "risk_score": 0.02, "risk_categories": [],
    "optimizer_mode": "full", "optimized_prompt": "Is water damage covered?", "optimized_tokens": 6,
    "techniques_applied": [], "optimization_attempts": 1,
    "fidelity_score": 1.0, "fidelity_passed": True, "prompt_for_model": "Is water damage covered?",
    "model_input_tokens": 6,
    "cache_hit": False, "cache_similarity": 0.0,
    "model_id": "gpt-oss-120b", "model_tier": "Large", "cost_estimate": 0.0, "failed_models": [],
    "response": "Yes.", "output_tokens": 1, "generation_error": None,
    "decision_log": [], "errors": [],
}


def test_every_node_has_a_writes_entry():
    assert set(NODES) == set(WRITES)


@pytest.mark.parametrize("name", list(NODES))
def test_node_writes_only_its_own_fields(name):
    out = NODES[name](dict(FULL_STATE))
    extra = set(out) - WRITES[name] - ALWAYS_ALLOWED
    assert not extra, f"{name} wrote fields it doesn't own: {extra}"


@pytest.mark.parametrize("name", list(NODES))
def test_node_logs_one_decision(name):
    out = NODES[name](dict(FULL_STATE))
    assert len(out["decision_log"]) == 1
    entry = out["decision_log"][0]
    assert entry["node"] == name
    assert {"decision", "reason", "metrics", "duration_ms"} <= set(entry)
    assert not out.get("errors"), out.get("errors")


@pytest.mark.parametrize("name", list(NODES))
def test_node_fallback_also_respects_contract(name):
    """A crash (here: an empty state) returns default output that still follows the contract."""
    out = NODES[name]({})
    extra = set(out) - WRITES[name] - ALWAYS_ALLOWED
    assert not extra, f"{name} fallback wrote fields it doesn't own: {extra}"
