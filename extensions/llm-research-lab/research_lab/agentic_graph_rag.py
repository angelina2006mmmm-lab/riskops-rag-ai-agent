"""Runnable LangGraph graph-enhanced RAG on synthetic documents.

Agent loop: entity graph retrieval -> evidence validation -> cited evidence
or abstention. A deterministic extractive answer is used; this is NOT
an LLM generation benchmark or full GraphRAG.
"""
from typing import TypedDict
from .graph_rag import retrieve

class AgentState(TypedDict, total=False):
    question: str
    as_of: str
    evidence: list[dict]
    answer: str
    status: str
    used_tools: list[str]

def graph_retriever(state: AgentState) -> dict:
    docs=retrieve(state.get("question",""),state.get("as_of","2026-10-01"),max_hops=3,top_k=4)
    return {"evidence":docs,"used_tools":["versioned_entity_graph_retrieval"]}

def guard_evidence(state: AgentState) -> str:
    docs=state.get("evidence") or []
    if not docs:return "abstain"
    if any(not x.get("id") or not x.get("path") or not x.get("effective_from") for x in docs):return "abstain"
    return "cite"

def cite_node(state: AgentState) -> dict:
    docs=state["evidence"]
    citations="; ".join(f"{x['id']} v{x['version']} ({x['effective_from']})" for x in docs)
    return {"status":"grounded","answer":f"Учебные источники: {citations}. Решение принимает специалист."}

def abstain_node(state: AgentState) -> dict:
    return {"status":"abstain","answer":"Нет подтверждённых источников для запроса."}

def build_graph():
    from langgraph.graph import StateGraph, START, END
    flow=StateGraph(AgentState)
    flow.add_node("graph_retriever",graph_retriever)
    flow.add_node("cite",cite_node)
    flow.add_node("abstain",abstain_node)
    flow.add_edge(START,"graph_retriever")
    flow.add_conditional_edges("graph_retriever",guard_evidence,{"cite":"cite","abstain":"abstain"})
    flow.add_edge("cite",END);flow.add_edge("abstain",END)
    return flow.compile()

def demo():
    graph=build_graph()
    pos=graph.invoke({"question":"supplier outage","as_of":"2026-10-01"})
    neg=graph.invoke({"question":"unknownqqq","as_of":"2026-10-01"})
    assert pos["status"]=="grounded" and "SUP-01 v2" in pos["answer"]
    assert neg["status"]=="abstain"
    return {"positive":pos,"negative":neg,"disclaimer":"No generative LLM call; synthetic policies."}

if __name__=="__main__":
    import json
    print(json.dumps(demo(),ensure_ascii=False,indent=2))
