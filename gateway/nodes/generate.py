"""Node 8: generate_response. Owner: TBD.

With generation.simulate: true (the default) no API is called; it returns a placeholder so the
demo works offline. The real version calls the provider in the model's config entry (Groq,
OpenRouter, Google AI Studio). On any provider error, return `generation_error` and add the
model to `failed_models`; the graph then sends the query back to route_model for another model.
"""
from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.tokens import estimate_tokens


def _fallback(state):
    model_id = state.get("model_id", "unknown")
    return {"generation_error": "generation crashed", "failed_models": [model_id]}


@safe_node("generate_response", fallback=_fallback)
def generate_response(state):
    cfg = settings.get("generation")
    model_id = state["model_id"]
    if not cfg["simulate"]:
        raise NotImplementedError("real provider calls are not wired in yet; set generation.simulate: true")
    if model_id in cfg["simulate_fail_models"]:
        return {"generation_error": f"simulated failure for {model_id}", "failed_models": [model_id],
                "decision_log": [decision("error", f"simulated provider failure for {model_id}")]}
    response = f"[Simulated response from {model_id}]"
    return {
        "response": response,
        "output_tokens": estimate_tokens(response),
        "generation_error": None,
        "decision_log": [decision("ok", f"SIMULATED: no API call made to {model_id}")],
    }
