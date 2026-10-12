import pytest
from risk_agent.graph import search_docs,retrieve,assess,route,answer,abstain
def test_retrieval():
 assert search_docs("IT service downtime")[0]["source"]=="OPS-01"
def test_assess():
 assert assess({"indicator":"downtime_minutes","value":135})["assessment"]=="high"
 assert assess({"indicator":"bad","value":15})["assessment"]=="not_assessed"
def test_abstain():
 state={"question":"unknown_xyz","evidence":[]}
 assert route(state)=="abstain"
 assert abstain(state)["status"]=="insufficient_evidence"
def test_citations():
 state={"evidence":search_docs("IT service downtime"),"assessment":"high"}
 assert "OPS-01" in answer(state)["answer"]
def test_graph():
 pytest.importorskip("langgraph")
 from risk_agent.graph import build_graph
 result=build_graph().invoke({"question":"IT service downtime","indicator":"downtime_minutes","value":135})
 assert result["status"]=="grounded_demo"
