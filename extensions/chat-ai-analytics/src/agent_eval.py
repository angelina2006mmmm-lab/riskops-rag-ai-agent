"""Measured LangGraph routing through separate synthetic chat specialists.

Actual StateGraph execution with specialist nodes, evidence verification and
metrics. Specialists are deterministic, NOT autonomous generative AI agents.
"""
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Protocol, TypedDict

from .experiment import INTENTS, PromptPolicy, load_cases

ROOT = Path(__file__).resolve().parents[1]
# Only synthetic policies, no banking access or real transactions.
POLICY = {
    "balance": ("kb_account", "SYN-ACCOUNT-v1", "Account summary: show approved navigation."),
    "card_block": ("kb_card_safety", "SYN-CARD-v1", "Card loss: direct to secure human support."),
    "transfer_status": ("kb_transfer", "SYN-TRANSFER-v1", "Transfers: show transaction status flow."),
    "loan_payment": ("kb_loan", "SYN-LOAN-v1", "Loan payments: show repayment schedule help."),
}
INJECTION = re.compile(
    r"ignore (?:previous|all) instructions|отмени предыдущие инструкции|"
    r"print system prompt|раскрой системный промпт", re.I
)

class Router(Protocol):
    def route(self, case_id: str, text: str) -> str: ...

class RuleRouter:
    def __init__(self, variant="B"):
        self.policy = PromptPolicy(variant)
    def route(self, case_id: str, text: str) -> str:
        return self.policy.predict(text)

class ReplayLLMRouter:
    """Replays outputs from an actual model experiment, not a live LLM call."""
    def __init__(self, report_path, variant="B"):
        rows = json.loads(Path(report_path).read_text(encoding="utf-8"))["row_outputs"][variant]
        self.predictions = {row["case_id"]: row["prediction"] for row in rows}
    def route(self, case_id: str, text: str) -> str:
        return self.predictions.get(case_id, "other")

class State(TypedDict, total=False):
    case_id: str
    text: str
    intent: str
    status: str
    tool: str
    citation: str
    answer: str
    tool_calls: list[str]

class ChatGraph:
    def __init__(self, router: Router):
        self.router = router
    def route_node(self, s: State) -> State:
        text = s.get("text","")
        if INJECTION.search(text):
            return {"intent":"other","status":"blocked","tool_calls":[]}
        label = self.router.route(s.get("case_id",""), text)
        if label not in INTENTS:
            label = "other"
        return {"intent":label,"status":"routed","tool_calls":[]}
    def choose_specialist(self, s: State) -> str:
        return s["intent"] if s["status"] != "blocked" else "other"
    def specialist(self, intent: str):
        def act(s: State) -> State:
            tool, source, answer = POLICY[intent]
            return {"tool":tool,"citation":source,"answer":answer,
                    "tool_calls":s["tool_calls"]+[tool]}
        return act
    def fallback(self, s: State) -> State:
        return {"answer":"No verified source for this request.",
                "tool":"","citation":"","tool_calls":[]}
    def verify(self, s: State) -> State:
        intent = s["intent"]
        if s["status"] == "blocked":
            return {"status":"blocked","answer":"Request rejected.","citation":""}
        if intent == "other":
            return {"status":"abstain","citation":""}
        tool, source, _ = POLICY[intent]
        valid = s.get("tool") == tool and s.get("citation") == source
        return {"status":"grounded" if valid else "abstain",
                "answer":s.get("answer","") if valid else "Source check failed.",
                "citation":source if valid else ""}
    def compile(self):
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(State)
        g.add_node("router",self.route_node)
        for label in POLICY:
            g.add_node(label,self.specialist(label))
        g.add_node("other",self.fallback)
        g.add_node("verify",self.verify)
        g.add_edge(START,"router")
        g.add_conditional_edges("router",self.choose_specialist,
                                {label:label for label in INTENTS})
        for label in INTENTS:
            g.add_edge(label,"verify")
        g.add_edge("verify",END)
        return g.compile()

def percentile(values, q):
    if not values:return 0.0
    s=sorted(values)
    return round(s[min(len(s)-1,math.ceil(q*len(s))-1)],3)

def evaluate(cases, router=None):
    graph=ChatGraph(router or RuleRouter("B")).compile()
    traces=[];durations=[]
    for c in cases:
        start=time.perf_counter()
        result=graph.invoke({"case_id":c.case_id,"text":c.text})
        elapsed=round((time.perf_counter()-start)*1000,3)
        durations.append(elapsed)
        expected_tool=POLICY[c.intent][0] if c.intent in POLICY else ""
        calls=result.get("tool_calls",[])
        expected_citation=POLICY[c.intent][1] if c.intent in POLICY else ""
        trace={
          "case_id":c.case_id, "expected_intent":c.intent,
          "predicted_intent":result["intent"],"status":result["status"],
          "tool_calls":calls, "expected_tool":expected_tool,
          "correct_tool": calls == ([expected_tool] if expected_tool else []),
          "citation_valid": result.get("citation","")==(
              POLICY[result["intent"]][1] if result["status"]=="grounded" else ""
          ),
          "latency_ms":elapsed,
        }
        traces.append(trace)
    n=len(traces)
    if not n:raise ValueError("No cases")
    return {
      "framework":"langgraph.StateGraph",
      "design":"Rule-based specialist agents, no autonomous generative agents",
      "cases":n,
      "task_success_rate":round(sum(x["predicted_intent"]==x["expected_intent"] for x in traces)/n,4),
      "correct_tool_rate":round(sum(x["correct_tool"] for x in traces)/n,4),
      "citation_valid_rate":round(sum(x["citation_valid"] for x in traces)/n,4),
      "abstain_rate":round(sum(x["status"]=="abstain" for x in traces)/n,4),
      "blocked_rate":round(sum(x["status"]=="blocked" for x in traces)/n,4),
      "invalid_tool_calls":sum(any(tool not in {v[0] for v in POLICY.values()}
                                   for tool in x["tool_calls"]) for x in traces),
      "p50_latency_ms":round(statistics.median(durations),3),
      "p95_latency_ms":percentile(durations,.95),
      "errors":[x for x in traces if x["predicted_intent"]!=x["expected_intent"]][:15],
      "traces":traces,
      "limitations":"Synthetic policies and deterministic specialists, no bank systems or human-executed action"
    }

def run():
    cases=load_cases(ROOT/"data/chat_cases.jsonl")
    report={"rule_based_B":evaluate(cases,RuleRouter("B"))}
    llm_report=ROOT/"reports/real_llm_prompt_eval.json"
    if llm_report.exists():
        selection={row["case_id"] for row in json.loads(llm_report.read_text(encoding="utf-8"))["row_outputs"]["B"]}
        chosen=[c for c in cases if c.case_id in selection]
        report["recorded_real_llm_B"]=evaluate(chosen,ReplayLLMRouter(llm_report,"B"))
        report["note"]="LLM predictions are replayed from offline experiment, not generated in the graph"
    dest=ROOT/"reports/multi_agent_quality.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({name:{"cases":item["cases"],"success":item["task_success_rate"]}
                      for name,item in report.items() if isinstance(item,dict)},ensure_ascii=False))
    return report

if __name__=="__main__":run()
