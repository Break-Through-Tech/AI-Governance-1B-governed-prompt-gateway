import pytest

from gateway import settings
from gateway.nodes.routing import minimum_tier_for_intent, pick_model, minimum_tier_for_risk, route_model

@pytest.fixture
def routing_config():
    with settings.override({
        "routing": {
            "intent_min_tier":{
                "insurance.claims.coverage_question": "Medium",          
                "insurance.claims.file_claim": "Medium",
                "insurance.claims.status": "Small",
                "insurance.claims.settlement": "Large",
                "insurance.policy.question": "Medium",
                "insurance.policy.change": "Medium",
                "insurance.policy.cancel": "Small",
                "insurance.quote.new_policy": "Medium",
                "insurance.billing.payment": "Small",
                "insurance.billing.dispute": "Medium",
                "insurance.support.complaint": "Medium",
                "insurance.support.info": "Small",
                # retail banking 
                "banking.account.open_close": "Medium", 
                "banking.account.balance_activity": "Small",
                "banking.card.manage": "Medium",
                "banking.fees": "Small",
                "banking.atm.problem": "Medium",
                "banking.locations": "Small",
                "banking.loan.apply": "Medium",
                "banking.loan.manage": "Medium",
                "banking.transfer": "Small",
                "banking.password": "Small",
                "support.contact": "Small",
                "unknown": "Medium",
            },
            
            "unknown_intent_tier": "Medium",
            "default_model": "large-model",
            "tier_order": ["Small", "Medium","Large"],
            "default_price_per_1m_input": 3.0,
            "models":[
                {"id": "small-expensive", "tier": "Small", "price": 0.50},
                {"id": "small-cheap", "tier": "Small", "price": 0.10},
                {"id": "medium-model", "tier": "Medium", "price": 0.80},
                {"id": "large-model", "tier": "Large", "price": 0.20},
                
            ],
            "risk_tier_rules":[
                {"min_score": 0.0, "tier": "Small"},
                {"min_score": 0.3, "tier": "Medium"},
                {"min_score": 0.4, "tier": "Large"},
            ],
            
        }
        
    }):
        yield
        
@pytest.mark.parametrize(
    "intent, expected_tier",
    [
        ("insurance.claims.coverage_question", "Medium"),          
        ("insurance.claims.file_claim", "Medium"),
        ("insurance.claims.status", "Small"),
        ("insurance.claims.settlement", "Large"),
        ("insurance.policy.question", "Medium"),
        ("insurance.policy.change", "Medium"),
        ("insurance.policy.cancel", "Small"),
        ("insurance.quote.new_policy", "Medium"),
        ("insurance.billing.payment", "Small"),
        ("insurance.billing.dispute", "Medium"),
        ("insurance.support.complaint", "Medium"),
        ("insurance.support.info", "Small"),
                        # retail banking 
        ("banking.account.open_close", "Medium"), 
        ("banking.account.balance_activity", "Small"),
        ("banking.card.manage", "Medium"),
        ("banking.fees", "Small"),
        ("banking.atm.problem", "Medium"),
        ("banking.locations", "Small"),
        ("banking.loan.apply", "Medium"),
        ("banking.loan.manage", "Medium"),
        ("banking.transfer", "Small"),
        ("banking.password", "Small"),
        ("support.contact", "Small"),
        ("unknown", "Medium"),
        ("unrecognized_label", "Medium"),
    ],
)
def test_intent_minimum_tier(routing_config, intent, expected_tier):
    assert minimum_tier_for_intent(intent) == expected_tier


@pytest.mark.parametrize(
    "score, expected_tier",
    [
        (0.0, "Small"),
        (0.299, "Small"),
        (0.3, "Medium"),
        (0.399, "Medium"),
        (0.4, "Large"),
        (0.499, "Large"),
        
    ],
)
def test_risk_minimum_tier(routing_config, score, expected_tier):
    assert minimum_tier_for_risk(score) == expected_tier

@pytest.mark.parametrize(
    "intent, score, expected_tier", 
    [
        ("banking.locations", 0.1, "Small"),
        ("banking.locations", 0.3, "Medium"),
        ("banking.locations", 0.4, "Large"),
        ("insurance.claims.settlement", 0.1, "Large"),
        ("unknown", 0.1, "Medium"),
    ],
)

def test_routing_combines_intent_and_risk(routing_config, intent, score, expected_tier):
    result = route_model({
        "intent": intent,
        "risk_score": score,
        "model_input_tokens": 100,
        "failed_models": [],
        
    })
    
    assert not result.get("errors")
    assert result["model_tier"] == expected_tier

def test_selects_cheapest_model_in_lowest_suitable_tier(routing_config):
    model = pick_model("Small", [])
    
    assert model["id"] == "small-cheap"
    
def test_does_not_select_below_minimum_tier(routing_config):
    model = pick_model("Medium", [])
    
    assert model["id"] == "medium-model"
    
def test_failed_model_is_skipped_within_same_tier(routing_config):
    model = pick_model("Small", ["small-cheap"])

    assert model["id"] == "small-expensive"


def test_exhausted_tier_moves_up(routing_config):
    model = pick_model(
        "Small",
        ["small-cheap", "small-expensive"],
    )

    assert model["id"] == "medium-model"


def test_no_available_models_raises(routing_config):
    with pytest.raises(RuntimeError, match="no models left"):
        pick_model(
            "Medium",
            ["medium-model", "large-model"],
        )
