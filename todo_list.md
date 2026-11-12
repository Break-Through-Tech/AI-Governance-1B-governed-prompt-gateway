# Governed Prompt Gateway: To-Do List

_Status as of 2026-11-11 (branch `main` / `langraph_setup`, after PR #16)._

The LangGraph scaffold is built and the full graph runs end to end
(`python -m gateway.graph "your prompt"`). **4 of 10 nodes are finished**; the other 6 are stubs that
return fixed or simulated values so the pipeline still works while each owner builds theirs.
See `gateway/README.md` for the full contract (state fields, edges, rules).

---

## ✅ Done

### Infrastructure
- [x] Graph scaffold: `state.py`, `graph.py`, `routers.py`, `settings.py`, `config.yaml`, `tokens.py`
- [x] All 10 nodes wired into the graph, with crash fallbacks (`@safe_node`)
- [x] Contract, path and optimizer tests (`pytest gateway/tests`)
- [x] CLI: run a prompt, `--json` output, `--diagram` Mermaid export
- [x] Website (`website/server.py` + `website/site/`) serving the UI and the spaCy token-count endpoint
- [x] Semantic cache with FAISS merged (`toxicity-detection-api/cache.py`, PR #15)
- [x] Toxicity detection API (`toxicity-detection-api/`)

### Nodes
| # | Node | Owner | What it does |
|---|---|---|---|
| 3 | `handle_unsafe` | Juee / Daniel | Returns the block/redirect message from `config.yaml`; unsafe prompts never reach a model |
| 4 | `optimize_prompt` | Riya | Shortens the prompt with several techniques |
| 5 | `verify_fidelity` | Riya | Checks the shortened prompt kept its meaning; retries or falls back to the original |
| 10 | `finalize_audit` | _Integration owner (TBD)_ | Sets final status, tokens saved and latency for the audit log |

---

## 🔲 To do

### Nodes still stubbed
- [ ] **`classify_intent`** (node 1). Owner: **_unassigned_**
  - Now: always returns `general`.
  - Needed: a real intent classifier (claims, billing, policy question, …). Keep the same returned fields.
- [ ] **`screen_risk`** (node 2). Owner: Juee / Daniel
  - Now: matches phrases listed in `config.yaml`.
  - Needed: wrap `toxicity-detection-api/pipeline.py`, using the detector's softmax probability as
    `risk_score` and the classifier's category as `risk_categories`. Load the models lazily (on first
    call), not at import time.
- [ ] **`cache_lookup`** (node 6). Owner: Vanessa
  - Now: in-memory cache that only matches identical prompts (case/whitespace normalized).
  - Needed: wrap `SemanticCache` from `toxicity-detection-api/cache.py`; `lookup` must also return the
    similarity score (the UI shows it); load the encoder lazily; set `cache.backend: semantic` in
    `config.yaml`.
- [ ] **`cache_store`** (node 9). Owner: Vanessa
  - Now: saves to the in-memory cache.
  - Needed: save to the semantic cache.
- [ ] **`route_model`** (node 7). Owner: **_unassigned_**
  - Now: always picks `routing.default_model` (`gpt-oss-120b`).
  - Needed: pick the cheapest tier that fits the intent's complexity and the risk score. Keep the
    fallback behaviour: never pick a model in `failed_models`; try the same tier first, then the next
    tier up.
- [ ] **`generate_response`** (node 8). Owner: **_unassigned_**
  - Now: simulated reply, no API call (`generation.simulate: true`).
  - Needed: call the provider in the model's config entry (Groq, OpenRouter, Google AI Studio). On any
    error, return `generation_error` and add the model to `failed_models` so the graph reroutes.

### Integration
- [ ] Add `POST /api/gateway/run {"prompt": "..."}` to `website/server.py`, returning the final
      `GatewayState` as JSON
- [ ] Point each UI page at its fields from that response (Risk screening, Token optimizer, Semantic
      cache, Model routing, Audit log)
- [ ] Fill in `requirements.txt` at the repo root (currently empty); gateway deps are in
      `gateway/requirements.txt`

### Open decisions (from `gateway/README.md` §9)
- [ ] Assign owners for `classify_intent`, `route_model`, `generate_response` and `finalize_audit`
- [ ] Cache position: keep it after optimization (current, matches the UI) or move it before?
- [ ] Browser-side optimizer in `app.js`: keep as instant preview, or move entirely to the backend?
- [ ] Tune starting thresholds in `config.yaml` on the datasets (`risk.block_threshold` 0.80,
      `risk.redirect_threshold` 0.50, `optimizer.fidelity.min_similarity` 0.80,
      `cache.min_similarity` 0.90, …)
- [ ] Standardize on Python 3.12 once the real risk node is wired in (`scikit-learn==1.6.1` needs it)
- [ ] Add a negation check to `verify_fidelity` (it doesn't yet catch a dropped "not"/"never")

---

## Rules for finishing a stub
1. Keep the function name, the `@safe_node(...)` decorator and the returned field names.
2. Return exactly one `decision(...)` in `decision_log`.
3. Put thresholds and model names in your section of `config.yaml`, never hardcoded.
4. Load big models lazily, never at import time.
5. `pytest gateway/tests` must still pass.
