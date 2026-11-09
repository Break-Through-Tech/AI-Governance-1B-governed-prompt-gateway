"""All 10 gateway nodes, in pipeline order. graph.py wires them together."""
from gateway.nodes.audit import finalize_audit
from gateway.nodes.cache import cache_lookup, cache_store
from gateway.nodes.generate import generate_response
from gateway.nodes.intent import classify_intent
from gateway.nodes.optimizer import optimize_prompt, verify_fidelity
from gateway.nodes.risk import handle_unsafe, screen_risk
from gateway.nodes.routing import route_model

NODES = {
    "classify_intent": classify_intent,
    "screen_risk": screen_risk,
    "handle_unsafe": handle_unsafe,
    "optimize_prompt": optimize_prompt,
    "verify_fidelity": verify_fidelity,
    "cache_lookup": cache_lookup,
    "route_model": route_model,
    "generate_response": generate_response,
    "cache_store": cache_store,
    "finalize_audit": finalize_audit,
}
