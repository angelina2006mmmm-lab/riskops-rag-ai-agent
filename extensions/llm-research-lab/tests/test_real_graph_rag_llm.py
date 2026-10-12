from research_lab.graph_rag_llm import build_graph,evaluate
def test_real_pretrained_graphrag():
    report=evaluate()
    assert report["positive_status"]=="grounded_draft"
    assert report["negative_status"]=="abstain"
def test_empty_input_skips_llm():
    out=build_graph().invoke({"question":"","as_of":"2026-10-01"})
    assert out["status"]=="abstain"
