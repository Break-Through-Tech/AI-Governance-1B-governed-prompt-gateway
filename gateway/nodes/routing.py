"""Node 7: route_model. Owner: Naivy.

STUB: always starts from routing.default_model. The real version should pick the cheapest tier
that fits the intent's complexity and the risk score. Keep the fallback behaviour: never pick a
model listed in `failed_models`; try the same tier first, then the next tier up.
"""
from gateway import settings
from gateway.nodes._base import decision, safe_node


def _models():
    return {m["id"]: m for m in settings.get("routing")["models"]}


def pick_model(minimum_tier: str, failed: list[str]) -> dict:
    """Choose the cheapest availabel model in the lowest qualifying tier"""
    cfg = settings.get("routing")
    tiers = cfg["tier_order"]
    
    #finds where to begin
    start_tier = tiers.index(minimum_tier) 
    
    for tier in tiers[start_tier:]:
        candidates = [model for model in cfg["models"] if model["tier"] == tier and model["id"] not in failed ]
        
        if candidates:
            return min(candidates, key=lambda model: model.get("price", cfg["default_price_per_1m_input"],),)

    raise RuntimeError(f"no models left to try (failed: {failed})")


def estimate_cost(model: dict, input_tokens: int) -> float:
    price = model.get("price", settings.get("routing")["default_price_per_1m_input"])
    return round(input_tokens * price / 1_000_000, 8)


def _fallback(state):
    cfg = settings.get("routing")
    model = _models()[cfg["default_model"]]
    return {"model_id": model["id"], "model_tier": model["tier"],
            "cost_estimate": estimate_cost(model, state.get("model_input_tokens", 0))}

def minimum_tier_for_intent(intent: str) -> str:
    cfg = settings.get("routing")
    return cfg["intent_min_tier"].get(intent, cfg["unknown_intent_tier"],)

@safe_node("route_model", fallback=_fallback)
def route_model(state):
    failed = state.get("failed_models", [])
    intent = state.get("intent", "unknown")
    minimum_tier = minimum_tier_for_intent(intent)
    model = pick_model(minimum_tier, failed)
    tokens = state.get("model_input_tokens", state.get("original_tokens", 0))
    cost = estimate_cost(model, tokens)
    reason = (f"intent '{intent}' requires at least {minimum_tier}; " f"selected cheapest availabel model in {model['tier']}")
    if failed:
        reason += f"; excluded failed models: {failed}"
    return {
        "model_id": model["id"],
        "model_tier": model["tier"],
        "cost_estimate": cost,
        "decision_log": [decision(model["id"], reason, intent=intent, minimum_tier=minimum_tier,tier=model["tier"], input_tokens=tokens,
                                  cost_estimate=cost)],
    }
