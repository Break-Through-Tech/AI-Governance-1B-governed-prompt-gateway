from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_pipeline

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/check")
def check(q: Query):
    return run_pipeline(q.text)