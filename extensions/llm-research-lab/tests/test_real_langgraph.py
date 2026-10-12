from research_lab.agentic_graph_rag import build_graph, demo
def test_end_to_end_real_langgraph():
    graph=build_graph()
    grounded=graph.invoke({"question":"supplier","as_of":"2025-10-01"})
    assert grounded["status"]=="grounded"
    assert "SUP-01 v1" in grounded["answer"]
    no_evidence=graph.invoke({"question":"no_such_entity","as_of":"2026-10-01"})
    assert no_evidence["status"]=="abstain"
def test_demo():
    assert demo()["positive"]["status"]=="grounded"
