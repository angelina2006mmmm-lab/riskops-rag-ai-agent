"""Real LangGraph graph when optional langgraph package is installed."""
from typing import TypedDict
from .graph_rag import retrieve
class State(TypedDict,total=False):
 question:str
 as_of:str
 evidence:list[dict]
 status:str
def retrieval_node(s:State):return {"evidence":retrieve(s["question"],s.get("as_of","2026-10-01"))}
def route(s:State):return "grounded" if s.get("evidence") else "abstain"
def grounded_node(s:State):return {"status":"grounded"}
def abstain_node(s:State):return {"status":"abstain"}
def compile_graph():
 from langgraph.graph import StateGraph,START,END
 g=StateGraph(State)
 for name,fn in [("retrieve",retrieval_node),("grounded",grounded_node),("abstain",abstain_node)]:g.add_node(name,fn)
 g.add_edge(START,"retrieve")
 g.add_conditional_edges("retrieve",route,{"grounded":"grounded","abstain":"abstain"})
 g.add_edge("grounded",END);g.add_edge("abstain",END)
 return g.compile()
