"""Builds the gateway graph. The only file that imports LangGraph.

    python -m gateway.graph                      # run the sample prompt from the UI
    python -m gateway.graph "your prompt here"   # run your own prompt
    python -m gateway.graph --json "..."         # print the full final state as JSON
    python -m gateway.graph --diagram            # print the Mermaid diagram
"""
import argparse
import json
import time
import uuid
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from gateway import routers
from gateway.nodes import NODES
from gateway.state import GatewayState

# Same sample prompt as website/site/app.js.
SAMPLE_PROMPT = "\n".join([
    "You are a customer service assistant for a regulated insurance provider.    You are always helpful, polite, "
    "and professional. You must always be polite and professional with customers.",
    "Answer only using the policy context provided. Do not give legal advice. If the user requests account changes, "
    "escalate to a licensed agent.",
    "Remember: never provide legal advice, and only answer from the provided policy context.",
    "", "", "",
    "### Policy context",
    "- **Home §4.2:** Water damage from a sudden burst pipe is covered up to the dwelling limit, less your deductible.",
    "- **FAQ:** To update your mailing address, sign in and open Profile settings.",
    "- **Auto §9.4:** Roadside assistance is available on auto policies with the Plus rider.",
    "- **Claims guide:** File within 60 days of the loss and include photos and a repair estimate.",
    "",
    "### Question",
    "Um, so basically I was just wondering, if it is not too much trouble, is water damage from a burst pipe in my "
    "kitchen covered, and in order to file a claim, what do I need to do? Thanks so much in advance!!",
])


def build_graph():
    g = StateGraph(GatewayState)
    for name, node in NODES.items():
        g.add_node(name, node)

    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "screen_risk")
    g.add_conditional_edges("screen_risk", routers.after_risk,
                            {"safe": "optimize_prompt", "unsafe / redirect": "handle_unsafe"})
    g.add_edge("handle_unsafe", "finalize_audit")
    g.add_edge("optimize_prompt", "verify_fidelity")
    g.add_conditional_edges("verify_fidelity", routers.after_fidelity,
                            {"pass": "cache_lookup", "fail, retry left": "optimize_prompt",
                             "fail, no retries": "cache_lookup"})
    g.add_conditional_edges("cache_lookup", routers.after_cache,
                            {"hit": "finalize_audit", "miss": "route_model"})
    g.add_edge("route_model", "generate_response")
    g.add_conditional_edges("generate_response", routers.after_generate,
                            {"ok": "cache_store", "error, fallback left": "route_model",
                             "error, no fallback": "finalize_audit"})
    g.add_edge("cache_store", "finalize_audit")
    g.add_edge("finalize_audit", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_app():
    return build_graph()


def run(prompt: str, query_id: str | None = None) -> dict:
    """Send one prompt through the gateway and return the final GatewayState."""
    state = {"query_id": query_id or f"Q-{uuid.uuid4().hex[:8]}",
             "original_prompt": prompt,
             "started_at": time.perf_counter()}
    return get_app().invoke(state)


def _print_summary(result: dict):
    print(f"\n{result['query_id']}  →  {result['final_status'].upper()}")
    print(f"tokens: {result.get('original_tokens')} original, {result.get('model_input_tokens', '-')} sent to model, "
          f"{result.get('tokens_saved', 0)} saved  ·  cost ${result.get('total_cost', 0):.6f}  ·  "
          f"{result.get('latency_ms')} ms\n")
    for i, e in enumerate(result["decision_log"], 1):
        print(f"{i:>2}. {e['node']:<18} {e['decision']:<28} {e['reason']}")
    if result.get("errors"):
        print("\nerrors:", *[f"  {e['node']}: {e['error']}" for e in result["errors"]], sep="\n")
    if result.get("prompt_for_model"):
        print(f"\nprompt sent to model:\n{result['prompt_for_model']}")
    print(f"\nresponse: {result.get('response')}")


def main():
    parser = argparse.ArgumentParser(description="Run a prompt through the governed prompt gateway.")
    parser.add_argument("prompt", nargs="?", default=SAMPLE_PROMPT)
    parser.add_argument("--json", action="store_true", help="print the full final state as JSON")
    parser.add_argument("--diagram", action="store_true", help="print the Mermaid diagram and exit")
    args = parser.parse_args()
    if args.diagram:
        print(get_app().get_graph().draw_mermaid())
        return
    result = run(args.prompt)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "started_at"}, indent=2, ensure_ascii=False))
    else:
        _print_summary(result)


if __name__ == "__main__":
    main()
