"""Node 10: finalize_audit. Owner: TBD (integration).

Always the last node. Works out the final status and the totals the dashboard shows.
"""
import time

from gateway.nodes._base import decision, safe_node


def final_status(state) -> str:
    if state.get("risk_label") != "safe":
        return "redirected" if state.get("risk_label") == "redirect" else "blocked"
    if state.get("cache_hit"):
        return "cached"
    if state.get("generation_error") or not state.get("response"):
        return "failed"
    return "answered"


def _latency(state) -> int:
    started = state.get("started_at")
    return int((time.perf_counter() - started) * 1000) if started else 0


@safe_node("finalize_audit", fallback=lambda s: {"final_status": final_status(s), "latency_ms": _latency(s)})
def finalize_audit(state):
    status = final_status(state)
    sent_to_model = status in ("answered", "failed")
    orig, used = state.get("original_tokens"), state.get("model_input_tokens")
    tokens_saved = orig - used if orig is not None and used is not None and status != "blocked" else 0
    total_cost = state.get("cost_estimate", 0.0) if sent_to_model else 0.0
    latency = _latency(state)
    return {
        "final_status": status,
        "tokens_saved": tokens_saved,
        "total_cost": total_cost,
        "latency_ms": latency,
        "decision_log": [decision(status, f"{len(state.get('decision_log', []))} prior decisions",
                                  tokens_saved=tokens_saved, total_cost=total_cost,
                                  errors=len(state.get("errors", [])))],
    }
