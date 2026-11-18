# Live Risk Screening integration (no Docker)

This overlay preserves the team's existing `website/site/app.js`, `styles.css`, and `toxicity-detection-api/pipeline.py`. The dashboard gains a visible **Live risk screening** link and a separate working two-column screening page.

## Install into the repository

Copy the three folders/files in this archive into the *root* of the existing repository, merging folders. Make a backup or a new Git branch first. These four files are added/replaced:

- `toxicity-detection-api/main.py` (replaced)
- `website/site/index.html` (replaced with dashboard + new link)
- `website/site/risk.html` (new)
- `website/site/risk.js` (new)

## Run

Use Python 3.12 (this project's pinned scikit-learn 1.6.1 may not install on Python 3.14).

```sh
cd ~/Downloads/AI-Governance-1B-governed-prompt-gateway/toxicity-detection-api
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000/dashboard/risk.html for live screening and http://127.0.0.1:8000/dashboard/ for the overall dashboard.

The first screening request downloads several models (~1.4 GB), requiring network and available storage; this may take minutes. The UI is never permitted to fake a success if the API fails. Loading **/health** only confirms the server is running, not that a model inference has completed.

Other dashboard stages (cache, routing, audit) still contain prototype/example data. Detection `safe` does not imply the other checks were performed. This change does not enforce downstream LLM routing.
