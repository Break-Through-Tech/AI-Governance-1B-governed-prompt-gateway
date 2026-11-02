"""Node 7: route_model. Owner: TBD.

STUB: always starts from routing.default_model. The real version should pick the cheapest tier
that fits the intent's complexity and the risk score. Keep the fallback behaviour: never pick a
model listed in `failed_models`; try the same tier first, then the next tier up.
"""
from gateway import settings
from gateway.nodes._base import decision, safe_node


def _models():
    return {m["id"]: m for m in settings.get("routing")["models"]}


def pick_model(start_id: str, failed: list[str]) -> dict:
    """The start model, or if it failed, the next model in the same tier, then tiers above it."""
    cfg = settings.get("routing")
    models = _models()
    tiers = cfg["tier_order"]
    start_tier = tiers.index(models[start_id]["tier"])
    candidates = [models[start_id]] + [m for m in cfg["models"] if m["id"] != start_id]
    for tier in tiers[start_tier:]:
        for m in candidates:
            if m["tier"] == tier and m["id"] not in failed:
                return m
    raise RuntimeError(f"no models left to try (failed: {failed})")


def estimate_cost(model: dict, input_tokens: int) -> float:
    price = model.get("price", settings.get("routing")["default_price_per_1m_input"])
    return round(input_tokens * price / 1_000_000, 8)


def _fallback(state):
    cfg = settings.get("routing")
    model = _models()[cfg["default_model"]]
    return {"model_id": model["id"], "model_tier": model["tier"],
            "cost_estimate": estimate_cost(model, state.get("model_input_tokens", 0))}


@safe_node("route_model", fallback=_fallback)
def route_model(state):
    failed = state.get("failed_models", [])
    model = pick_model(settings.get("routing")["default_model"], failed)
    tokens = state.get("model_input_tokens", state.get("original_tokens", 0))
    cost = estimate_cost(model, tokens)
    reason = "STUB: default model" if not failed else f"fallback after failures: {failed}"
    return {
        "model_id": model["id"],
        "model_tier": model["tier"],
        "cost_estimate": cost,
        "decision_log": [decision(model["id"], reason, tier=model["tier"], input_tokens=tokens,
                                  cost_estimate=cost)],
    }
