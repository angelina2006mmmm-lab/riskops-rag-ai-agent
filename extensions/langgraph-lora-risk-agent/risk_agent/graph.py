"""Evidence-gated LangGraph demonstration with entirely synthetic risk policies.

Deterministic offline workflow: no actual LLM inference or financial advice.
"""
import math
import re
from typing import TypedDict

POLICIES = (
 ("OPS-01","IT service downtime: more than 120 minutes is high operational risk. Escalate the incident."),
 ("OPS-02","Fraud alerts: more than 7 suspicious transactions per hour require manual investigation."),
 ("OPS-03","Data access incidents: unauthorized access requires incident documentation and security triage."),
 ("OPS-04","Supplier outage: longer than 60 minutes requires backup supplier service assessment."),
 ("OPS-05","Every operational risk assessment must cite policy evidence and avoid inventing thresholds."),
)
THRESHOLDS = {"downtime_minutes":120,"fraud_alerts":7,"supplier_outage_minutes":60}
INDICATOR_SOURCE = {"downtime_minutes":"OPS-01","fraud_alerts":"OPS-02","supplier_outage_minutes":"OPS-04"}

class RiskState(TypedDict,total=False):
 question:str
 indicator:str
 value:float
 evidence:list[dict]
 assessment:str
 answer:str
 status:str

def tokens(text:str)->set[str]:
 return set(re.findall(r"[\w-]+",text.lower()))

def search_docs(question:str,top_k:int=3)->list[dict]:
 """Lexical baseline, not semantic embeddings."""
 if not isinstance(question,str) or not question.strip() or not 1<=top_k<=5:return []
 q=tokens(question)
 hits=[(len(q & tokens(txt)),src,txt) for src,txt in POLICIES]
 return [{"source":src,"excerpt":txt,"overlap":score} for score,src,txt in sorted(hits,key=lambda x:(-x[0],x[1])) if score>0][:top_k]

def assess_indicator(indicator:str|None,value:float|None)->str:
 if indicator not in THRESHOLDS:return "not_assessed: unknown indicator"
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
  return "not_assessed: invalid value"
 return "high" if value>THRESHOLDS[indicator] else "review"

def retrieve_node(state:RiskState)->RiskState:
 return {"evidence":search_docs(state.get("question",""))}
def evaluate_node(state:RiskState)->RiskState:
 return {"assessment":assess_indicator(state.get("indicator"),state.get("value"))}
def route_on_evidence(state:RiskState)->str:
 evidence=state.get("evidence") or []
 if not evidence:return "abstain"
 indicator=state.get("indicator")
 if indicator is not None:
  if state.get("assessment","").startswith("not_assessed"):return "abstain"
  required=INDICATOR_SOURCE.get(indicator)
  if required is None or not any(d.get("source")==required for d in evidence):return "abstain"
 return "answer"
def answer_node(state:RiskState)->RiskState:
 sources=", ".join(str(d["source"]) for d in state["evidence"])
 return {"status":"grounded_demo","answer":f"Учебная оценка: {state.get('assessment','не проводилась')}. Источники: {sources}. Только синтетические политики; проверка человеком обязательна."}
def abstain_node(state:RiskState)->RiskState:
 return {"status":"insufficient_evidence","answer":"Нет согласованных источников или допустимого показателя; воздерживаюсь от вывода о риске."}

def run_offline(state:RiskState)->RiskState:
 """Dependency-free reference execution, useful for offline unit tests."""
 out=dict(state)
 out.update(retrieve_node(out));out.update(evaluate_node(out))
 out.update(answer_node(out) if route_on_evidence(out)=="answer" else abstain_node(out))
 return out

def build_graph():
 """Real LangGraph StateGraph, requiring the separately installed package."""
 from langgraph.graph import StateGraph,START,END
 graph=StateGraph(RiskState)
 for name,fn in [("retrieve",retrieve_node),("evaluate",evaluate_node),("answer",answer_node),("abstain",abstain_node)]:
  graph.add_node(name,fn)
 graph.add_edge(START,"retrieve");graph.add_edge("retrieve","evaluate")
 graph.add_conditional_edges("evaluate",route_on_evidence,{"answer":"answer","abstain":"abstain"})
 graph.add_edge("answer",END);graph.add_edge("abstain",END)
 return graph.compile()
if __name__=="__main__":
 try: result=build_graph().invoke({"question":"IT service downtime 135 minutes","indicator":"downtime_minutes","value":135})
 except ImportError:
  print("LangGraph not installed: using deterministic reference workflow.")
  result=run_offline({"question":"IT service downtime 135 minutes","indicator":"downtime_minutes","value":135})
 print(result["status"],result["answer"])
