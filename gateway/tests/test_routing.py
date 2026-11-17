import pytest

from gateway import settings
from gateway.nodes.routing import minimum_tier_for_intent, pick_model

@pytest.fixture
def routing_config():
    with settings.override({
        "routing": {
            "intent_min_tier":{
                "general": "Small",
                "billing": "Small", 
                "claims": "Medium",
                "complex_reasoning": "Large",
                "unknown": "Medium",
            },
            
            "unknown_intent_tier": "Medium",
            "tier_order": ["Small", "Medium","Large"],
            "default_price_per_1m_input": 3.0,
            "models":[
                {"id": "small-expensive", "tier": "Small", "price": 0.50},
                {"id": "small-cheap", "tier": "Small", "price": 0.10},
                {"id": "medium-model", "tier": "Medium", "price": 0.80},
                {"id": "large-model", "tier": "Large", "price": 0.20},
                
            ],
            
        }
    }):
        yield
        
@pytest.mark.parametrize(
    "intent, expected_tier",
    [
        ("general", "Small"),
        ("billing", "Small"),
        ("claims", "Medium"),
        ("complex_reasoning", "Large"),
        ("unknown", "Medium"),
        ("unrecognized_label", "Medium"),
    ],
)

def test_intent_minimum_tier(routing_config, intent, expected_tier):
    assert minimum_tier_for_intent(intent == expected_tier)

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
