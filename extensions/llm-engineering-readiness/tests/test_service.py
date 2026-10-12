from contextlib import contextmanager
from datetime import date
import pytest
from fastapi.testclient import TestClient
from agent.api import create_app
from agent.domain import Evidence,InMemoryPolicies,InferenceError
from agent.service import RiskService
from agent.observability import make_tracer

class RecordingClient:
    def __init__(self):self.names=[]
    @contextmanager
    def start_as_current_observation(self,*,as_type,name):
        assert as_type=="span"
        self.names.append(name)
        yield self

class BrokenRetriever:
    def find(self,question,as_of,topic):return []
class FutureRetriever:
    def find(self,question,as_of,topic):
        return [Evidence("FUT-01",1,date(2100,1,1),topic,50)]

def svc(retriever=None,tracer=None):
    return RiskService(retriever or InMemoryPolicies(),tracer or make_tracer(False))

def test_versions_and_thresholds():
    s=svc()
    old=s.evaluate("supplier outage","supplier_outage_minutes",100,date(2025,6,1))
    new=s.evaluate("supplier outage","supplier_outage_minutes",100,date(2026,6,1))
    assert old.citations==("SUP-01@v1",) and old.risk_band=="review"
    assert new.citations==("SUP-01@v2",) and new.risk_band=="standard"

def test_fail_closed_and_invalid_topics():
    assert svc(BrokenRetriever()).evaluate("supplier","supplier_outage_minutes",140,date.today()).status=="abstain"
    assert svc(FutureRetriever()).evaluate("supplier","supplier_outage_minutes",140,date.today()).status=="abstain"
    with pytest.raises(InferenceError):svc().evaluate("hello","unsupported",5,date.today())

def test_non_finite_rejected():
    for value in [float("nan"),float("inf"),-1]:
        with pytest.raises(InferenceError):svc().evaluate("supplier","supplier_outage_minutes",value,date.today())

def test_tracing_v4_client_injected():
    fake=RecordingClient()
    result=svc(tracer=make_tracer(enabled=True,client=fake)).evaluate("supplier","supplier_outage_minutes",130,date.today())
    assert result.status=="grounded" and fake.names==["risk_evidence_review"]

def test_langfuse_real_sdk_import():
    from langfuse import get_client
    assert callable(get_client)

def test_langfuse_requires_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY",raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY",raising=False)
    with pytest.raises(RuntimeError):make_tracer(True)

def test_fastapi_contract_and_metrics():
    c=TestClient(create_app(svc()))
    assert c.get("/health").status_code==200
    result=c.post("/v1/review",json={"question":"supplier outage","topic":"supplier_outage_minutes","value":130,"as_of":"2026-10-12"})
    assert result.status_code==200 and result.json()["citations"]==["SUP-01@v2"]
    assert c.get("/metrics").json()["outcomes"]["requests"]==1

def test_fastapi_invalid_payloads():
    c=TestClient(create_app(svc()))
    for bad in ({"value":"NaN"},{"value":-5},{"topic":"bogus"},{"question":""},{"surprise":123}):
        a={"question":"supplier","topic":"supplier_outage_minutes","value":90,"as_of":"2026-10-12"}
        a.update(bad)
        assert c.post("/v1/review",json=a).status_code==422
