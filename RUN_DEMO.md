# Governed Prompt Gateway demo

## Install and launch

Use **64-bit CPython 3.12** and Git. Clone the demo branch, then run the commands for your platform from the repository root.

```sh
git clone --branch demo/complete-gateway https://github.com/Break-Through-Tech/AI-Governance-1B-governed-prompt-gateway.git
cd AI-Governance-1B-governed-prompt-gateway
```

**macOS** (Apple Silicon, macOS 15 or later for the pinned FAISS wheel)

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.15.0
.venv/bin/python -m pip install -r gateway/requirements.txt
.venv/bin/python -m gateway.demo
```

**Linux** (x86-64 or ARM64 with glibc 2.28 or later, including compatible WSL distributions)

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r gateway/requirements.txt
.venv/bin/python -m gateway.demo
```

**Windows PowerShell** (x86-64 Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r gateway/requirements.txt
.\.venv\Scripts\python.exe -m gateway.demo
```

Open [the dashboard](http://127.0.0.1:8000/dashboard/). The launcher binds to localhost, uses one worker so repeat requests share a cache, and defaults OpenMP to one thread for CPU inference. **Generation is simulated by default, even if provider keys are present.** Stop the server with Ctrl+C.

The standalone risk screen is at [`/dashboard/risk.html`](http://127.0.0.1:8000/dashboard/risk.html). FastAPI exposes interactive [API docs](http://127.0.0.1:8000/docs).

## Model access and provider keys

The first safe prompt downloads any missing weights over HTTPS from Hugging Face. Allow the first run to finish before closing the server. Later requests reuse the local model files. The four repositories are:

- `hmmmm1234/customer-intent-roberta` for intent
- `jueedeshmukh/01_toxic_detection_daniel` for safety screening
- `jueedeshmukh/toxic_prompt_classification_riya` for risk categories
- `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings

On the tested Apple Silicon machine, the Hub cache occupied about **1.8 GiB** and the Python environment about **1.2 GiB**. Leave at least **5 GiB** free for the environment, weights, and temporary downloads. The running server used about **1.6 GiB RAM** after the four-prompt smoke test. The demo runs on CPU. The Hub cache defaults to your home directory and can be relocated with `HF_HOME` before launch.

The startup command uses **simulated generation** and requires zero provider keys. Intent classification, toxicity screening and category assignment use the team models. Semantic cache comparisons use MiniLM and FAISS. Token optimization and lexical fidelity use the configured gateway heuristics. Simulation returns a labeled placeholder. Live and simulated cache entries use separate scopes.

Live provider access is optional. Groq has returned HTTP 403 on completion requests even when model listing succeeded. The audit shows the status and recognized error codes when supplied. Use simulation for the team demo.

## Live Gemini generation

Inject `GEMINI_API_KEY` into the backend environment through your secret manager or shell. From the activated environment, start the existing gateway with a Gemini text model available to your Google project. Google lists [`gemini-3.8-flash`](https://aistudio.google.com/docs/models/gemini-3.8-flash) as a stable text model compatible with its [chat completions endpoint](https://ai.google.dev/gemini-api/docs/openai). Availability still depends on your project.

```sh
GEMINI_MODEL=gemini-3.8-flash GATEWAY_SIMULATE_GENERATION=0 OMP_NUM_THREADS=1 \
  python -m gateway.demo
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

## Troubleshooting

- **Python or wheel unavailable.** Check the venv reports Python 3.12 and that the interpreter matches your CPU architecture. On Debian or Ubuntu, install the `venv` package matching Python 3.12 if venv creation fails. On Windows, WSL with a compatible Linux distribution is an alternative for native DLL or wheel issues. The pinned PyTorch release does not provide an Intel macOS wheel. Use a compatible Linux environment if available. Native Windows and Linux instructions are based on published wheels. On Windows, Hugging Face may use extra disk space when it has to copy cached files instead of linking them.
- **Model download stalls or fails.** Confirm outbound HTTPS to Hugging Face and its download hosts, free disk space, and that `HF_HUB_OFFLINE` is unset for the first run. The audit identifies the unavailable node. For a restricted Hub repository, use `hf auth login` from the venv if you have access. If the team supplies a local intent checkpoint, set `GATEWAY_INTENT_MODEL` to its folder containing the model, tokenizer, and trained label mapping.
- **Port already in use.** Add `--port 8001` to the launch command and open [`http://127.0.0.1:8001/dashboard/`](http://127.0.0.1:8001/dashboard/).
- **Unexpected live attempt.** Remove any inherited `GATEWAY_SIMULATE_GENERATION=0` setting before launch. The default uses no provider keys.

## Regression tests

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q
node --test website/tests/dashboard.test.cjs
for file in website/site/*.js; do node --check "$file" || break; done
```

On PowerShell, run the equivalent commands from the repository root.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m pytest -q
node --test website/tests/dashboard.test.cjs
Remove-Item Env:HF_HUB_OFFLINE
```

Use the venv Python or activate the environment first. Node.js is only needed for the dashboard tests. The offline tests use deterministic screening and intent doubles, real FAISS indexing with fixture embeddings, and intercepted provider requests. Once all Hugging Face weights are cached, run the real-model smoke test in its own process.

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 GATEWAY_TEST_REAL_MODELS=1 \
  python -m pytest -q gateway/tests/test_demo_integration.py::test_real_demo_screening_intent_and_semantic_cache
```
