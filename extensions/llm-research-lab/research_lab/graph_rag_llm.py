"""End-to-end graph-grounded generation with real pretrained TinyStories-1M.

Synthetic mini GraphRAG / Agentic RAG proof of concept. Retrieval uses
version-aware entity-policy graph; LangGraph routes to a pretrained
causal LM for a short DRAFT, attaches verified citations or abstains.
No claim of factual accuracy or production safety.
"""
from datetime import date
from typing import TypedDict
from .graph_rag import retrieve

MODEL_ID="roneneldan/TinyStories-1M"

class AgentState(TypedDict,total=False):
    question:str
    as_of:str
    plan:str
    sources:list[dict]
    draft:str
    answer:str
    status:str
    tools:list[str]

def plan_node(state:AgentState):
    return {"plan":"search_versioned_policies" if state.get("question","").strip() else "abstain"}

def choose_start(state:AgentState):
    return "retrieve" if state.get("plan")=="search_versioned_policies" else "abstain"

def graph_tool_node(state:AgentState):
    docs=retrieve(state["question"],state.get("as_of","2026-10-01"),top_k=3,max_hops=3)
    return {"sources":docs,"tools":["entity_graph_versioned_retrieval"]}

def verify_sources(state:AgentState):
    day=date.fromisoformat(state.get("as_of","2026-10-01"))
    docs=state.get("sources",[])
    if not docs or any(not d.get("id") or not d.get("path") or date.fromisoformat(d["effective_from"])>day for d in docs):
        return "abstain"
    return "generate"

def llm_generate_node(state:AgentState):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    mdl=AutoModelForCausalLM.from_pretrained(MODEL_ID)
    mdl.eval()
    facts=" ".join(d["text"] for d in state["sources"])[:500]
    prompt=("Summarize this fictional policy. Do not introduce additional facts.\n"
           "Context: "+facts+"\nQuestion: "+state["question"]+"\nSummary:")
    inp=tok(prompt,return_tensors="pt",truncation=True,max_length=180)
    with torch.no_grad():
        out=mdl.generate(**inp,max_new_tokens=14,do_sample=False,pad_token_id=tok.eos_token_id)
    generated=tok.decode(out[0,inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    return {"draft":generated or "(empty model response)","tools":state.get("tools",[])+["pretrained_lm_generation"]}

def cite_node(state:AgentState):
    cited="; ".join(f"{doc['id']} v{doc['version']} ({doc['effective_from']})" for doc in state["sources"])
    return {"status":"grounded_draft","answer":f"Черновик LLM (непроверенный): {state['draft']}\nИсточники из графа: {cited}. Проверка человеком обязательна."}

def abstain_node(state:AgentState):
    return {"status":"abstain","answer":"Нет подтверждённой версии политики для запроса."}

def build_graph():
    from langgraph.graph import StateGraph,START,END
    g=StateGraph(AgentState)
    for name,fn in [("plan",plan_node),("retrieve",graph_tool_node),("generate",llm_generate_node),("cite",cite_node),("abstain",abstain_node)]:
        g.add_node(name,fn)
    g.add_edge(START,"plan")
    g.add_conditional_edges("plan",choose_start,{"retrieve":"retrieve","abstain":"abstain"})
    g.add_conditional_edges("retrieve",verify_sources,{"generate":"generate","abstain":"abstain"})
    g.add_edge("generate","cite")
    g.add_edge("cite",END);g.add_edge("abstain",END)
    return g.compile()

def evaluate():
    graph=build_graph()
    found=graph.invoke({"question":"supplier outage","as_of":"2026-10-01"})
    missing=graph.invoke({"question":"unknown_ontology","as_of":"2026-10-01"})
    assert found["status"]=="grounded_draft" and "SUP-01 v2" in found["answer"]
    assert "pretrained_lm_generation" in found["tools"]
    assert missing["status"]=="abstain"
    return {"positive_status":found["status"],"negative_status":missing["status"],
            "retrieved_sources":[x["id"] for x in found["sources"]],
            "model":MODEL_ID,
            "disclaimer":"Citations are retrieved; generated text is NOT quality-validated or production-ready."}

if __name__=="__main__":
    import json
    print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
