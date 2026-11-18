"""FastAPI service for real risk screening and end-to-end governed gateway."""
from functools import lru_cache
from pathlib import Path
import sys

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Governed Prompt Gateway")


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=16000)

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value):
        if not value.strip():
            raise ValueError("Enter a prompt to screen.")
        return value


@lru_cache(maxsize=1)
def get_pipeline():
    from pipeline import run_pipeline
    return run_pipeline


def pipeline_dependency():
    try:
        return get_pipeline()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Screening unavailable. No result was issued.") from error


@app.get("/health")
def health():
    return {"status": "ready", "model_loaded": "pipeline" in sys.modules}


@app.post("/check")
def check(q: Query, run=Depends(pipeline_dependency)):
    try:
        outcome = run(q.text)
        if not isinstance(outcome, dict) or outcome.get("label") not in {"safe", "unsafe"}:
            raise ValueError("Invalid model response")
        return outcome
    except Exception as error:
        raise HTTPException(status_code=503, detail="Screening unavailable. No result was issued.") from error


@app.post("/gateway/run")
def run_gateway(q: Query):
    try:
        from gateway.graph import run
        result = run(q.text)
        # This is an in-memory demo. Do not persist raw prompts or audit data.
        result.pop("started_at", None)
        if result.get("risk_label") not in ("safe", "unsafe", "redirect"):
            raise RuntimeError("Gateway returned no valid risk decision")
        return result
    except Exception as error:
        raise HTTPException(status_code=503, detail="Gateway unavailable. No result was issued.") from error


site = ROOT / "website" / "site"
if site.is_dir():
    app.mount("/dashboard", StaticFiles(directory=site, html=True), name="dashboard")
