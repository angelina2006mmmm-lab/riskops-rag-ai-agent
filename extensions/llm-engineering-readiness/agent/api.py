"""FastAPI transport adapter; business rules are isolated in service.py."""
from datetime import date
from threading import Lock
from collections import Counter
from typing import Literal
from fastapi import FastAPI,Depends
from pydantic import BaseModel,Field,ConfigDict
from .service import RiskService
from .domain import InMemoryPolicies
from .observability import make_tracer

class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    question:str=Field(min_length=1,max_length=400)
    topic:Literal["supplier_outage_minutes","fraud_score"]
    value:float=Field(ge=0,allow_inf_nan=False)
    as_of:date

def create_app(service=None):
    app=FastAPI(title="Educational risk agent engineering",version="1.0.0")
    svc=service if service is not None else RiskService(InMemoryPolicies(),make_tracer())
    counts=Counter();lock=Lock()
    def get_service():return svc
    @app.get("/health")
    def health():return {"status":"ok","version":"1.0.0"}
    @app.get("/metrics")
    def metrics():
        with lock:return {"outcomes":dict(counts),"note":"in-memory local demo; no user inputs"}
    @app.post("/v1/review")
    def review(payload:ReviewRequest,s:RiskService=Depends(get_service)):
        try:
            result=s.evaluate(payload.question,payload.topic,payload.value,payload.as_of)
            return result.__dict__
        finally:
            with lock:counts["requests"]+=1
    return app

app=create_app()
