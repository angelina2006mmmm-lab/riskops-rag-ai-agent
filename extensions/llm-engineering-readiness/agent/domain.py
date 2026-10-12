"""Typed domain contracts keep HTTP, retrieval and telemetry independent."""
from dataclasses import dataclass
from datetime import date
from typing import Protocol

@dataclass(frozen=True)
class Evidence:
    source_id: str
    version: int
    valid_from: date
    topic: str
    threshold: float

class Retriever(Protocol):
    def find(self, question: str, as_of: date, topic: str) -> list[Evidence]: ...

class Tracer(Protocol):
    def span(self, name: str): ...

class InferenceError(ValueError):
    pass

class InMemoryPolicies:
    """Synthetic temporal evidence with two versions; replaceable by a DB retriever."""
    _rows = [
        Evidence("SUP-01",1,date(2025,1,1),"supplier_outage_minutes",90.),
        Evidence("SUP-01",2,date(2026,1,1),"supplier_outage_minutes",120.),
        Evidence("FRD-01",1,date(2025,1,1),"fraud_score",.75),
    ]
    def find(self,question:str,as_of:date,topic:str)->list[Evidence]:
        if not question.strip():return []
        active=[x for x in self._rows if x.topic==topic and x.valid_from<=as_of]
        if not active:return []
        current=max(active,key=lambda x:(x.valid_from,x.version))
        return [current]

class NullTracer:
    def span(self,name:str):
        from contextlib import nullcontext
        return nullcontext()
