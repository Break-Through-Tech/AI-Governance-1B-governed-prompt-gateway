"""Node 8: generate_response. Owner: TBD."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gateway import settings
from gateway.nodes._base import decision, safe_node
from gateway.tokens import estimate_tokens


PROVIDERS = {
    "Groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions"),
    "OpenRouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions"),
    "Google AI Studio": (
        "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    ),
}


# Audit entries are returned to dashboard clients. Only publish known codes,
# since a provider or intermediary can include request data in the error body.
_GROQ_ERROR_CODES = {
    "model_permission_blocked_org",
    "model_permission_blocked_project",
    "model_terms_required",
    "model_not_found",
    "model_decommissioned",
}


def _groq_error_code(error):
    try:
        data = json.loads(error.read(8192))
    except (OSError, ValueError, RecursionError):
        return None
    details = data.get("error") if isinstance(data, dict) else None
    code = details.get("code") if isinstance(details, dict) else None
    return code if isinstance(code, str) and code in _GROQ_ERROR_CODES else None


def provider_available(model):
    provider = PROVIDERS.get(model.get("provider"))
    return bool(provider and os.getenv(provider[0]))


def _complete(model, prompt):
    key_name, url = PROVIDERS[model["provider"]]
    body = {"model": model["api"], "messages": [{"role": "user", "content": prompt}]}
    request = Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ[key_name]}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as reply:
            payload = json.load(reply)
        response = payload["choices"][0]["message"]["content"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError("missing answer")
    except HTTPError as error:
        provider = model["provider"]
        with error:
            code = _groq_error_code(error) if provider == "Groq" else None
        detail = f" ({code})" if code else ""
        raise RuntimeError(f"{provider} returned HTTP {error.code}{detail}") from None
    except (URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError(f"{model['provider']} did not return a usable completion") from None
    output_tokens = (payload.get("usage") or {}).get("completion_tokens")
    return response, output_tokens if isinstance(output_tokens, int) else estimate_tokens(response)


def _fallback(state):
    model_id = state.get("model_id", "")
    return {"generation_error": "generation failed", "failed_models": [model_id] if model_id else []}


@safe_node("generate_response", fallback=_fallback)
def generate_response(state):
    model_id = state["model_id"]
    if not model_id:
        return {"generation_error": "no eligible model", "decision_log": [
            decision("error", "no eligible model with an available provider")
        ]}
    simulated = settings.generation_simulated()
    if simulated:
        if model_id in settings.get("generation")["simulate_fail_models"]:
            return {"generation_error": f"simulated failure for {model_id}", "failed_models": [model_id],
                    "decision_log": [decision("error", f"simulated provider failure for {model_id}",
                                              simulated=True)]}
        response = f"[Simulated response from {model_id}]"
        output_tokens = estimate_tokens(response)
    else:
        model = next(m for m in settings.get("routing")["models"] if m["id"] == model_id)
        response, output_tokens = _complete(model, state["prompt_for_model"])
    return {
        "response": response,
        "output_tokens": output_tokens,
        "generation_error": None,
        "decision_log": [decision("ok", f"{'Simulated response' if simulated else 'Provider response'} from {model_id}",
                                  simulated=simulated)],
    }
