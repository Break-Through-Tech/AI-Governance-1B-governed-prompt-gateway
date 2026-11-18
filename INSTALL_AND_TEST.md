# Live Governed Prompt Gateway — Steps 1 and 2

**This overlay updates** `gateway/nodes/risk.py`, `toxicity-detection-api/main.py` and adds `website/site/gateway-live.html` and `website/site/gateway-live.js`.

**Important:** The overlay expects the repo's existing `gateway/` LangGraph package, `toxicity-detection-api/pipeline.py` and the existing frontend to remain present. It replaces the previous risk-only FastAPI `main.py` with an extended version that keeps `/check`, `/health`, and `/dashboard`.

## Install

From a **new terminal** (leave existing Uvicorn running until ready to restart):

```bash
cd ~/Downloads/AI-Governance-1B-governed-prompt-gateway
git status --short
# Only if not already on a dedicated demo branch:
git switch -c demo/complete-gateway
unzip -o ~/Downloads/Prompt_Gateway_Steps_1_and_2.zip -d .
source toxicity-detection-api/.venv/bin/activate
python -m pip install -r gateway/requirements.txt
```

Stop the existing Uvicorn process with Ctrl+C. Then from the repository root:

```bash
cd toxicity-detection-api
python -m uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000/dashboard/gateway-live.html

## Test

Use these in another terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/check -H 'Content-Type: application/json' -d '{"text":"Is water damage covered by my insurance policy?"}'
curl -s -X POST http://127.0.0.1:8000/gateway/run -H 'Content-Type: application/json' -d '{"text":"Is water damage covered by my insurance policy?"}' | python3 -m json.tool
```

Inspect `decision_log`: `screen_risk` should say `ML toxicity detector decision`, not `STUB: phrase match`. Unsafe prompts should have `final_status=blocked` and **no** `model_id` or `optimized_prompt`. The model loads on first request, which can be slow. Repeat a safe prompt twice to observe the existing exact-match cache.

## Scope and limitations

- The detector currently returns a binary safe/unsafe label, not a calibrated risk probability. `risk_score` is 0/1 as a **policy proxy**, not a model confidence score.
- The model category `confidence` (when present) is the category classifier's confidence, not the probability of toxicity.
- Intent and model selection are still stubs; caching remains exact-match and responses are simulated, per `gateway/config.yaml`.
- The ML detector is known to miss politely phrased malicious requests. Do not claim complete jailbreak detection.
- No API keys are required in simulation mode. Use synthetic data only.
- Do not use `--reload` during initial model load; startup and model download may take time.
