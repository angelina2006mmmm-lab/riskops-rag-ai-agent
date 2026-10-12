from fastapi import FastAPI
from pydantic import BaseModel,Field
from .agent import assess
app=FastAPI(title="Temporal Guardrail Agent — synthetic demo")
class RiskQuestion(BaseModel):
    question: str=Field(min_length=1,max_length=2000)
    indicator: str=Field(min_length=1,max_length=70)
    value: float
    as_of: str
@app.get("/health")
def health(): return {"ok":True}
@app.post("/assess")
def assess_endpoint(payload:RiskQuestion):
    return assess(payload.question,payload.indicator,payload.value,payload.as_of).to_dict()
