# Governed Prompt Gateway demo

Run these commands from the repository root using Python 3.12. This starts FastAPI with the live dashboard, standalone risk check, and LangGraph gateway.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.15.0
python -m pip install -r gateway/requirements.txt -r toxicity-detection-api/requirements.txt
OMP_NUM_THREADS=1 python -m uvicorn main:app --app-dir toxicity-detection-api --host 127.0.0.1 --port 8000
```

On Linux, use `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` for CPU-only PyTorch. `OMP_NUM_THREADS=1` was needed for stable CPU model inference on the tested macOS environment. Keep a single worker so repeat requests share the same cache.

Open the [live dashboard](http://127.0.0.1:8000/dashboard/), [standalone risk screening](http://127.0.0.1:8000/dashboard/risk.html), or [API docs](http://127.0.0.1:8000/docs). The live panel submits to `POST /gateway/run` with `{"text":"..."}` and shows the returned decisions, response, and audit. The walkthrough below it uses sample data.

## Model access and provider keys

The gateway downloads the team toxicity detector, the category ensemble, the fine-tuned RoBERTa intent classifier, and the MiniLM cache encoder on demand. Allow outbound HTTPS to Hugging Face and disk space for the weights. Use `hf auth login` if the intent checkpoint requires authentication. If the team supplied a local checkpoint, set `GATEWAY_INTENT_MODEL=/absolute/path/to/checkpoint` before startup. The checkpoint must include the tokenizer and the trained intent label mapping. `HF_HUB_OFFLINE=1` uses existing local files.

Export at least one provider key before starting the server. The environment variable names are `GROQ_API_KEY`, `OPENROUTER_API_KEY`, and `GEMINI_API_KEY`. The router considers configured models for providers with keys, applies the intent and risk tier floors, and retries eligible alternatives after a provider failure. A missing qualifying model produces a failed generation in the audit.

For a credential-free run of the graph, set `GATEWAY_SIMULATE_GENERATION=1` on the Uvicorn command. This simulates generation and keeps the real screening, intent, and cache backends selected. The response and audit explicitly identify simulation. Simulation and live generation use separate cache entries.

## Demo walkthrough

1. Run **“How do I file a home insurance claim?”** and inspect screening, intent, optimization, fidelity, and routing in the audit.
2. Repeat the same request in the same server process to verify a cache hit and skipped generation. Try **“How can I submit a home insurance claim?”** to inspect the paraphrase score. A hit depends on the actual score, configured threshold, and matching cache scope.
3. Use the dashboard’s unsafe example to inspect the category, verify a blocked result, and confirm that the cache and generation stages are skipped.
4. Use the optimization example and inspect the prepared prompt alongside the original. Fidelity can cause a retry and can restore the original prompt.

```sh
curl -fsS http://127.0.0.1:8000/gateway/run \
  -H 'Content-Type: application/json' \
  -d '{"text":"How do I file a home insurance claim?"}'
```

`/health` checks service availability. Use `/check` and `/gateway/run` to validate inference. Audit errors identify unavailable models. Screening errors redirect, intent errors route as unknown, and cache errors bypass reuse.

Use synthetic or public prompts in this local demo. The in-memory cache is shared within a worker and resets on restart. Semantic similarity and the context, number, and negation checks are heuristics for reuse. The service has no authentication or tenant isolation. The toxicity detector currently screens the first 512 model tokens, and lexical fidelity estimates preservation of meaning. The displayed input cost is an estimate and excludes provider billing for output and retries.

## Offline regression tests

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q gateway/tests
```

The suite uses deterministic intent and risk doubles, tests real FAISS indexing with fixture embeddings, and exercises generation through simulated and intercepted provider calls.
