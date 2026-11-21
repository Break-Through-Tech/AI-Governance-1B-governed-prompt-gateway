"""Node 7: route_model. Owner: Naivy.

Selects the lowest qualifying model tier using configured intent and risk
requirements. Chooses the cheapest available model within that tier, excluding
failed models, and moves to higher tiers when needed. 

Cost estimates cover input tokens only 

If there is no model left to try, route_model returns an empty model_id.
Then the graph skips generate_response and goes straight to finalize_audit.
"""
from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.nodes.generate import provider_available, provider_model_id


def pick_model(minimum_tier: str, failed: list[str]) -> dict:
    """Choose the cheapest available model in the lowest qualifying tier"""
    cfg = settings.get("routing")
    tiers = cfg["tier_order"]
    
    #finds where to begin
    start_tier = tiers.index(minimum_tier) 
    
    simulated = settings.generation_simulated()
    for tier in tiers[start_tier:]:
        candidates = [
            model for model in cfg["models"]
            if model["tier"] == tier and model["id"] not in failed
            and ((not model.get("api_env") or provider_model_id(model)) if simulated
                 else provider_available(model))
        ]
        
        if candidates:
            return min(candidates, key=lambda model: model.get("price", cfg["default_price_per_1m_input"],),)

    raise RuntimeError(f"no models left to try (failed: {failed})")


def estimate_cost(model: dict, input_tokens: int) -> float:
    price = model.get("price", settings.get("routing")["default_price_per_1m_input"])
    return round(input_tokens * price / 1_000_000, 8)


def _fallback(state):
    # Never lower the tier or retry an excluded model when routing fails.
    return {"model_id": "", "model_tier": None, "cost_estimate": 0.0}


def minimum_tier_for_intent(intent: str) -> str:
    cfg = settings.get("routing")
    return cfg["intent_min_tier"].get(intent, cfg["unknown_intent_tier"],)

def minimum_tier_for_risk(risk_score: float) -> str:
    cfg = settings.get("routing")
    tiers = cfg["tier_order"]
    
    qualifying_tiers = [
        rule["tier"]
        for rule in cfg["risk_tier_rules"]
        if risk_score >= rule["min_score"]
    ]
    
    return max(qualifying_tiers, key=tiers.index)

@safe_node("route_model", fallback=_fallback)
def route_model(state):
    failed = state.get("failed_models", [])
    intent = state.get("intent", "unknown")
    
    cfg = settings.get("routing")
    tiers = cfg["tier_order"]
    
    intent_tier = minimum_tier_for_intent(intent)
    
    risk_score = state["risk_score"]
    if not 0 <= risk_score <= 1:
        raise ValueError("risk_score must be between 0 and 1")
    risk_tier = minimum_tier_for_risk(risk_score)
    minimum_tier = max(intent_tier, risk_tier, key=tiers.index)
    
    tokens = state.get("model_input_tokens", state.get("original_tokens", 0))
    try:
        model = pick_model(minimum_tier, failed)
    except RuntimeError:
        # pick_model raises this when every model it could use has already failed.
        # Return an empty model_id so the graph skips generate_response.
        reason = f"no model in tier {minimum_tier} or above can be tried"
        return {
            "model_id": "",
            "model_tier": None,
            "cost_estimate": 0.0,
            "decision_log": [decision("no_eligible_model", reason, failed_models=failed)],
        }
    cost = estimate_cost(model, tokens)
    reason = (
        f"intent '{intent}' requires at least {intent_tier}; " 
        f"risk_score {risk_score} requires {risk_tier}; "
        f"minimum tier {minimum_tier}; "
        f"selected cheapest available model in {model['tier']}"
    )
    if failed:
        reason += f"; excluded failed models: {failed}"
    return {
        "model_id": model["id"],
        "model_tier": model["tier"],
        "cost_estimate": cost,
        "decision_log": [
            decision(
                model["id"], 
                reason, 
                intent=intent, 
                intent_tier=intent_tier,
                risk_score=risk_score,
                risk_tier=risk_tier,
                minimum_tier=minimum_tier,
                tier=model["tier"], 
                input_tokens=tokens,
                cost_estimate=cost,
                input_price_per_million=model.get("price"),
                )],
    }
