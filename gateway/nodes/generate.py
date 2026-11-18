"""Node 8: generate_response. Owner: TBD."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
import json
import os
from time import sleep
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


_GOOGLE_ERROR_STATUSES = {
    "INVALID_ARGUMENT", "FAILED_PRECONDITION", "UNAUTHENTICATED", "PERMISSION_DENIED",
    "NOT_FOUND", "RESOURCE_EXHAUSTED", "INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED",
    "model_not_found", "quota_exceeded", "rate_limit_exceeded", "service_unavailable",
}
_GOOGLE_RETRYABLE = {429, 500, 502, 503, 504}
_GOOGLE_MAX_RETRY_WAIT = 5


def _provider_error_code(provider, error):
    if provider not in {"Groq", "Google AI Studio"}:
        return None
    try:
        data = json.loads(error.read(8192))
    except (OSError, HTTPException, ValueError, RecursionError):
        return None
    details = data.get("error") if isinstance(data, dict) else None
    if not isinstance(details, dict):
        return None
    if provider == "Groq":
        code, allowed = details.get("code"), _GROQ_ERROR_CODES
    else:
        code, allowed = details.get("status") or details.get("code"), _GOOGLE_ERROR_STATUSES
    return code if isinstance(code, str) and code in allowed else None


def _retry_delay(error, backoff):
    retry_after = error.headers.get("Retry-After", "") if error.headers else ""
    try:
        if retry_after.isascii() and retry_after.isdecimal():
            seconds = int(retry_after)
        elif retry_after:
            retry_at = parsedate_to_datetime(retry_after)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        else:
            return backoff
    except (ValueError, TypeError, OverflowError):
        return backoff
    # A longer server cooldown belongs outside this synchronous request.
    return max(backoff, seconds) if seconds <= _GOOGLE_MAX_RETRY_WAIT else None


def _request_completion(provider, request):
    backoffs = (1, 2, None) if provider == "Google AI Studio" else (None,)
    for attempt, backoff in enumerate(backoffs, start=1):
        try:
            with urlopen(request, timeout=30) as reply:
                return json.load(reply), attempt
        except HTTPError as error:
            with error:
                code = _provider_error_code(provider, error)
                delay = None
                if backoff is not None and error.code in _GOOGLE_RETRYABLE and code != "quota_exceeded":
                    delay = _retry_delay(error, backoff)
                if delay is None:
                    detail = f" ({code})" if code else ""
                    retries = f" after {attempt} attempts" if attempt > 1 else ""
                    raise RuntimeError(f"{provider} returned HTTP {error.code}{detail}{retries}") from None
            sleep(delay)


def provider_model_id(model):
    env_name = model.get("api_env")
    return os.getenv(env_name, "").strip() if env_name else model.get("api")


def provider_available(model):
    provider = PROVIDERS.get(model.get("provider"))
    return bool(provider and os.getenv(provider[0], "").strip() and provider_model_id(model))


def _complete(model, prompt):
    key_name, url = PROVIDERS[model["provider"]]
    api_model = provider_model_id(model)
    if not api_model:
        raise RuntimeError(f"{model['provider']} model is not configured")
    body = {"model": api_model, "messages": [{"role": "user", "content": prompt}]}
    request = Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ[key_name].strip()}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        payload, attempts = _request_completion(model["provider"], request)
        response = payload["choices"][0]["message"]["content"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError("missing answer")
    except (URLError, TimeoutError, HTTPException, KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError(f"{model['provider']} did not return a usable completion") from None
    usage = payload.get("usage")
    output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return response, output_tokens if isinstance(output_tokens, int) else estimate_tokens(response), attempts


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
    metrics = {"simulated": simulated}
    if simulated:
        if model_id in settings.get("generation")["simulate_fail_models"]:
            return {"generation_error": f"simulated failure for {model_id}", "failed_models": [model_id],
                    "decision_log": [decision("error", f"simulated provider failure for {model_id}",
                                              simulated=True)]}
        response = f"[Simulated response from {model_id}]"
        output_tokens = estimate_tokens(response)
    else:
        model = next(m for m in settings.get("routing")["models"] if m["id"] == model_id)
        response, output_tokens, attempts = _complete(model, state["prompt_for_model"])
        metrics["provider"] = model["provider"]
        if attempts > 1:
            metrics["provider_attempts"] = attempts
    return {
        "response": response,
        "output_tokens": output_tokens,
        "generation_error": None,
        "decision_log": [decision("ok", f"{'Simulated response' if simulated else 'Provider response'} from {model_id}",
                                  **metrics)],
    }
