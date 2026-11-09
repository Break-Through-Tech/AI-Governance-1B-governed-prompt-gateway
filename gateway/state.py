"""The shared state that carries one query through the gateway (README §4).

This file is the contract between node owners. Adding a field: update the README in the
same PR. Renaming or removing one: get sign-off from every node that reads it.
"""
import operator
from typing import Annotated, Literal, TypedDict


class GatewayState(TypedDict, total=False):
    # input (set by gateway.graph.run)
    query_id: str
    original_prompt: str
    started_at: float                    # time.perf_counter() when the run began

    # 1. classify_intent
    intent: str
    intent_confidence: float
    original_tokens: int

    # 2–3. screen_risk / handle_unsafe
    risk_label: Literal["safe", "unsafe", "redirect"]
    risk_score: float                    # 0 = safe, 1 = certainly unsafe
    risk_categories: list[str]

    # 4. optimize_prompt
    optimizer_mode: Literal["full", "safe"]
    optimized_prompt: str
    optimized_tokens: int
    techniques_applied: list[dict]       # [{"id", "name", "tokens_before", "tokens_after", "changes"}]
    optimization_attempts: int

    # 5. verify_fidelity
    fidelity_score: float
    fidelity_passed: bool
    fidelity_checks: dict                # each check's result, for the UI and the audit log
    prompt_for_model: str                # optimized_prompt if fidelity passed, else original_prompt
    model_input_tokens: int              # tokens in prompt_for_model

    # 6, 9. cache_lookup / cache_store
    cache_hit: bool
    cache_similarity: float

    # 7. route_model
    model_id: str
    model_tier: Literal["Small", "Medium", "Large"]
    cost_estimate: float
    failed_models: Annotated[list[str], operator.add]

    # 8. generate_response
    response: str
    output_tokens: int
    generation_error: str | None

    # 10. finalize_audit
    final_status: Literal["answered", "cached", "blocked", "redirected", "failed"]
    tokens_saved: int
    total_cost: float
    latency_ms: int

    # every node (append-only)
    decision_log: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]


# Which state fields each node may write (README §3 "Writes" column).
# tests/test_contract.py fails if a node returns anything else.
ALWAYS_ALLOWED = {"decision_log", "errors"}

WRITES: dict[str, set[str]] = {
    "classify_intent": {"intent", "intent_confidence", "original_tokens"},
    "screen_risk": {"risk_label", "risk_score", "risk_categories"},
    "handle_unsafe": {"response", "final_status"},
    "optimize_prompt": {"optimized_prompt", "optimized_tokens", "techniques_applied",
                        "optimization_attempts", "optimizer_mode"},
    "verify_fidelity": {"fidelity_score", "fidelity_passed", "fidelity_checks", "optimizer_mode",
                        "prompt_for_model", "model_input_tokens"},
    "cache_lookup": {"cache_hit", "cache_similarity", "response"},
    "route_model": {"model_id", "model_tier", "cost_estimate"},
    "generate_response": {"response", "output_tokens", "generation_error", "failed_models"},
    "cache_store": set(),
    "finalize_audit": {"final_status", "tokens_saved", "total_cost", "latency_ms"},
}
