# Governed Prompt Gateway demo

Run these commands from the repository root using Python 3.12. This starts FastAPI with the live dashboard, standalone risk check, and LangGraph gateway.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.15.0
python -m pip install -r gateway/requirements.txt -r toxicity-detection-api/requirements.txt
GATEWAY_SIMULATE_GENERATION=1 OMP_NUM_THREADS=1 python -m uvicorn main:app --app-dir toxicity-detection-api --host 127.0.0.1 --port 8000
```

On Linux, use `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` for CPU-only PyTorch. `OMP_NUM_THREADS=1` was needed for stable CPU model inference on the tested macOS environment. Keep a single worker so repeat requests share the same cache.

Open the [live dashboard](http://127.0.0.1:8000/dashboard/), [standalone risk screening](http://127.0.0.1:8000/dashboard/risk.html), or [API docs](http://127.0.0.1:8000/docs). The prompt form submits to `POST /gateway/run` with `{"text":"..."}`. All six dashboard stages show the latest execution from that response.

## Model access and provider keys

The gateway downloads the team toxicity detector, the category ensemble, the fine-tuned RoBERTa intent classifier, and the MiniLM cache encoder on demand. Allow outbound HTTPS to Hugging Face and disk space for the weights. Use `hf auth login` if the intent checkpoint requires authentication. If the team supplied a local checkpoint, set `GATEWAY_INTENT_MODEL=/absolute/path/to/checkpoint` before startup. The checkpoint must include the tokenizer and the trained intent label mapping. `HF_HUB_OFFLINE=1` uses existing local files.

The startup command uses **simulated generation** and requires zero provider keys. Intent classification, toxicity screening and category assignment use the team models. Semantic cache comparisons use MiniLM and FAISS. Token optimization and lexical fidelity use the configured gateway heuristics. Simulation returns a labeled placeholder. Live and simulated cache entries use separate scopes.

Live provider access is optional. Groq has returned HTTP 403 on completion requests even when model listing succeeded. The audit shows the status and recognized error codes when supplied. Use simulation for the team demo.

## Live Gemini generation

Inject `GEMINI_API_KEY` into the backend environment through your secret manager or shell. From the activated environment, start the existing gateway with a Gemini text model available to your Google project. Google lists [`gemini-3.8-flash`](https://aistudio.google.com/docs/models/gemini-3.8-flash) as a stable text model compatible with its [chat completions endpoint](https://ai.google.dev/gemini-api/docs/openai). Availability still depends on your project.

```sh
GEMINI_MODEL=gemini-3.8-flash GATEWAY_SIMULATE_GENERATION=0 OMP_NUM_THREADS=1 \
  python -m uvicorn main:app --app-dir toxicity-detection-api --host 127.0.0.1 --port 8000
```

Gemini retries HTTP 429, 500, 502, 503, and 504 up to twice before handing the failure back to the existing router. The audit exposes HTTP status and recognized Google error codes.

Run with only the Gemini provider key in that process to exercise Gemini specifically. When other providers are configured, the existing router still selects an eligible tier and model by its configured rules. The Gemini entry qualifies at every configured minimum tier. Submit a safe prompt from the dashboard and inspect Model Routing and Audit Log for the provider and the generation outcome. A cache hit reuses the stored response. Restart the worker when switching Gemini models to start with an empty cache. Safe cache misses send the prepared prompt to Google, so use public or synthetic prompts for the demo.

## Demo walkthrough

1. Submit **“Can you explain how to file a home insurance claim?”** in a clean worker. Inspect the classified intent, cache miss, selected model, and simulated response.
2. Repeat the same text for an exact cache hit. Then submit **“Can you explain how to file a claim for home insurance?”** to demonstrate a paraphrase hit. With the cached team models, this pair scored about 0.992 against the unchanged 0.90 threshold. Use the score from the current run during the demo.
3. Use the **Unsafe example**. Inspect the returned category and refusal. The cache and generation stages should show skipped.
4. For the fallback demo, restart with the command below, then submit the original safe question once. The first model deliberately fails in simulation, and the next eligible model answers. Review both attempts in Model Routing and Audit Log.

| Stage | What to show |
| --- | --- |
| Overview | Actual response and its source, submitted prompt, intent confidence, and per-request token counts |
| Risk screening | Detector score, safety outcome, categories when reported, and decision rationale |
| Token optimizer | Before and after text and counts, applied techniques, fidelity result, and any original-prompt fallback |
| Semantic cache | Hit or miss, comparison score when available, threshold, and skipped generation on a hit |
| Model routing | Selected tier and model, estimated input cost, and ordered routing and generation attempts |
| Audit log | Chronological node decisions, durations, and error details |

### Optional simulated fallback

Stop the current server first. From the repository root, run this in the activated environment. The override applies only to this process.

```sh
GATEWAY_SIMULATE_GENERATION=1 OMP_NUM_THREADS=1 python - <<'PYTHON'
import sys

import uvicorn
from gateway import settings

sys.path.insert(0, "toxicity-detection-api")
with settings.override({"generation": {"simulate_fail_models": ["qwen3.8-27b"]}}):
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
PYTHON
```

```sh
curl -fsS http://127.0.0.1:8000/gateway/run \
  -H 'Content-Type: application/json' \
  -d '{"text":"How do I file a home insurance claim?"}'
```

`/health` checks service availability. Use `/check` and `/gateway/run` to validate inference. Audit errors identify unavailable models. Screening errors redirect, intent errors route as unknown, and cache errors bypass reuse.

Use synthetic or public prompts in this local demo. The in-memory cache is shared within a worker and resets on restart. Semantic similarity and the context, number, and negation checks are heuristics for reuse. The service has no authentication or tenant isolation. The toxicity detector currently screens the first 512 model tokens, and lexical fidelity estimates preservation of meaning. The displayed input cost is an estimate and excludes provider billing for output and retries.

The dashboard reports observed token reduction for the current prompt and shows financial estimates only when the corresponding data is available. The project targets of 30–50% token reduction and at least 80% F1 require separate evaluation. Demo outputs do not establish either target.

## Regression tests

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q
node --test website/tests/dashboard.test.cjs
for file in website/site/*.js; do node --check "$file" || break; done
```

The offline tests use deterministic screening and intent doubles, real FAISS indexing with fixture embeddings, and intercepted provider requests. Once all Hugging Face weights are cached, run the real-model smoke test in its own process.

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 GATEWAY_TEST_REAL_MODELS=1 \
  python -m pytest -q gateway/tests/test_demo_integration.py::test_real_demo_screening_intent_and_semantic_cache
```
