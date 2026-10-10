"""FastAPI endpoints for retrieval, policy checks and the RAG agent."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from riskops.agent import RiskAgent
from riskops.knowledge import Retriever
from riskops.llm import ChatCompletionClient, LLMConfigurationError
from riskops.multi_agent import MultiAgentCoordinator
from riskops.tools import Tools

app = FastAPI(
    title="RiskOps RAG + AI Agent (educational)",
    description="Синтетические политики, безопасный локальный режим, опциональная LLM",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_agent() -> RiskAgent:
    return RiskAgent(Tools(Retriever()))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)


class AssessRequest(BaseModel):
    indicator: Literal["downtime_minutes", "emissions_tons"]
    value: float = Field(ge=0, allow_inf_nan=False)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["offline", "llm"] = "offline"
    indicator: Literal["downtime_minutes", "emissions_tons"] | None = None
    value: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def both_or_neither(self):
        if (self.indicator is None) != (self.value is None):
            raise ValueError("indicator и value указываются вместе")
        return self


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "riskops-rag-ai-agent"}


@app.get("/documents")
def documents() -> dict:
    docs = sorted(set(p.source for p in get_agent().tools.retriever.passages))
    return {"documents": docs, "count": len(docs)}


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    return get_agent().tools.execute(
        "search_docs", {"query": payload.query, "top_k": payload.top_k}
    )


@app.post("/assess")
def assess(payload: AssessRequest) -> dict:
    return get_agent().tools.execute("assess_indicator", payload.model_dump())


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    agent = get_agent()
    if payload.mode == "offline":
        return agent.offline(payload.question, payload.indicator, payload.value).to_dict()
    try:
        llm = ChatCompletionClient()
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return agent.llm(payload.question, llm).to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/multi-agent/brief")
def multi_agent_brief(payload: AskRequest) -> dict:
    """Three offline role-separated specialists; no real LLM is called here."""
    return MultiAgentCoordinator(get_agent().tools.retriever).run(
        payload.question, payload.indicator, payload.value
    )
