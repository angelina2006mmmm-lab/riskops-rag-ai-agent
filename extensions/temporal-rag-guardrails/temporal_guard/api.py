"""FastAPI wrapper and demo-level quality counters, synthetic data only."""
from time import perf_counter
from fastapi import FastAPI
from pydantic import BaseModel,Field
from .agent import assess
from .telemetry import record,snapshot
app=FastAPI(title="Temporal Guardrail Agent - synthetic demo")
class RiskQuestion(BaseModel):
    question:str=Field(min_length=1,max_length=2000)
    indicator:str=Field(min_length=1,max_length=70)
    value:float
    as_of:str
@app.post("/assess")
def assess_endpoint(payload:RiskQuestion):
    start=perf_counter()
    result=assess(payload.question,payload.indicator,payload.value,payload.as_of)
    record(result.status,(perf_counter()-start)*1000)
    return result.to_dict()
@app.get("/health")
def health():return {"ok":True}
@app.get("/metrics")
def metrics():
    """In-memory counts/latency, no user queries or policy text."""
    return snapshot()
