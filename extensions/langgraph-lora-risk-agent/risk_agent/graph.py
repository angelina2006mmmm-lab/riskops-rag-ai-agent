"""Deterministic LangGraph risk workflow, synthetic data only."""
from typing import TypedDict
import re
POLICIES = [
 ("OPS-01","IT service downtime over 120 minutes is high operational risk."),
 ("OPS-02","More than 7 fraud alerts per hour require manual investigation."),
 ("OPS-03","Unauthorized data access requires incident documentation."),
 ("OPS-04","Supplier outage over 60 minutes requires backup assessment."),
]
THRESHOLDS={"downtime_minutes":120,"fraud_alerts":7,"supplier_outage_minutes":60}
class RiskState(TypedDict,total=False):
 question:str
 indicator:str
 value:float
 evidence:list[dict]
 assessment:str
 answer:str
 status:str
def search_docs(question:str):
 terms=set(re.findall(r"[\w-]+",question.lower()))
 ranked=[(len(terms & set(re.findall(r"[\w-]+",txt.lower()))),source,txt) for source,txt in POLICIES]
 return [{"source":source,"excerpt":txt} for score,source,txt in sorted(ranked,key=lambda r:(-r[0],r[1])) if score>0][:2]
def retrieve(state:RiskState):
 return {"evidence":search_docs(state.get("question",""))}
def assess(state:RiskState):
 key=state.get("indicator")
 val=state.get("value")
 if key not in THRESHOLDS or val is None or val<0:
  return {"assessment":"not_assessed"}
 return {"assessment":"high" if val>THRESHOLDS[key] else "review"}
def route(state:RiskState):
 return "answer" if state.get("evidence") else "abstain"
def answer(state:RiskState):
 refs=", ".join(x["source"] for x in state["evidence"])
 return {"status":"grounded_demo","answer":f"Учебная оценка: {state.get('assessment')}. Источники: {refs}. Нужна проверка человеком."}
def abstain(state:RiskState):
 return {"status":"insufficient_evidence","answer":"Нет подтверждающей учебной политики."}
def build_graph():
 from langgraph.graph import StateGraph,START,END
 graph=StateGraph(RiskState)
 for name,fn in [("retrieve",retrieve),("assess",assess),("answer",answer),("abstain",abstain)]:
  graph.add_node(name,fn)
 graph.add_edge(START,"retrieve")
 graph.add_edge("retrieve","assess")
 graph.add_conditional_edges("assess",route,{"answer":"answer","abstain":"abstain"})
 graph.add_edge("answer",END)
 graph.add_edge("abstain",END)
 return graph.compile()
if __name__=="__main__":
 print(build_graph().invoke({"question":"IT service downtime 135 minutes","indicator":"downtime_minutes","value":135}))
