"""Single-purpose, easily mocked agent logic; fail closed without evidence."""
import math
from dataclasses import dataclass
from datetime import date
from .domain import Retriever,Tracer,InferenceError

@dataclass(frozen=True)
class RiskResult:
    status:str
    risk_band:str | None
    citations:tuple[str,...]
    message:str

class RiskService:
    def __init__(self,retriever:Retriever,tracer:Tracer):
        self._retriever=retriever
        self._tracer=tracer

    def evaluate(self,question:str,topic:str,value:float,as_of:date)->RiskResult:
        if not math.isfinite(value) or value<0:raise InferenceError("Invalid indicator")
        if topic not in {"supplier_outage_minutes","fraud_score"}:raise InferenceError("Unsupported topic")
        with self._tracer.span("risk_evidence_review"):
            evidence=self._retriever.find(question,as_of,topic)
            if not evidence:return RiskResult("abstain",None,(),"No dated evidence")
            item=evidence[0]
            if item.topic!=topic or item.valid_from>as_of:
                return RiskResult("abstain",None,(),"Invalid evidence")
            tag=f"{item.source_id}@v{item.version}"
            band="review" if value>=item.threshold else "standard"
            return RiskResult("grounded",band,(tag,),"Synthetic policy only, requires human review")
