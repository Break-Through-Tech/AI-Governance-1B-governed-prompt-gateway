"""Helpers every node uses: the decision-log entry format and the crash-handling decorator (README §6)."""
import functools
import time


def decision(decision: str, reason: str, **metrics) -> dict:
    """One decision-log entry. safe_node fills in "node" and "duration_ms"."""
    return {"decision": decision, "reason": reason, "metrics": metrics}


def safe_node(name: str, fallback):
    """Wrap a node so a crash never takes down the gateway.

    `fallback(state) -> dict` returns the node's default output (README §6 table). If the node
    raises, that default is returned instead, with the error recorded in `errors` and `decision_log`.
    """
    def wrap(fn):
        @functools.wraps(fn)
        def node(state: dict) -> dict:
            start = time.perf_counter()
            try:
                out = fn(state) or {}
                for entry in out.get("decision_log", []):
                    entry.setdefault("node", name)
                    entry["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
                return out
            except Exception as exc:  # noqa: BLE001 — that's the point of this wrapper
                error = f"{type(exc).__name__}: {exc}"
                out = fallback(state)
                out["errors"] = [{"node": name, "error": error}]
                out["decision_log"] = [{
                    "node": name,
                    "decision": "error_fallback",
                    "reason": f"node crashed ({error}); used default output",
                    "metrics": {},
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                }]
                return out
        node.node_name = name
        return node
    return wrap
