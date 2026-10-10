"""Small role-based multi-agent DEMO (deterministic; not independent LLM agents).

Three isolated specialists exchange typed results:
researcher -> risk analyst -> reviewer. The separate /ask?mode=llm endpoint
illustrates genuine model-driven tool calling when a provider is configured.
"""

from dataclasses import asdict, dataclass

from riskops.knowledge import Retriever
from riskops.tools import assess_indicator


@dataclass
class ResearchReport:
    question: str
    evidence: list[dict]


@dataclass
class RiskReport:
    indicator: str | None
    assessment: dict | None


@dataclass
class ReviewReport:
    has_sources: bool
    claim_sources_known: bool
    notes: list[str]


class ResearchAgent:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def run(self, question: str) -> ResearchReport:
        hits = self.retriever.search(question, limit=3)
        return ResearchReport(question, [h.to_dict() for h in hits])


class IndicatorAgent:
    def run(self, indicator: str | None, value: float | None) -> RiskReport:
        if indicator is None or value is None:
            return RiskReport(None, None)
        return RiskReport(indicator, assess_indicator(indicator, value))


class EvidenceReviewAgent:
    def __init__(self, retriever: Retriever):
        self.allowed_sources = {p.source for p in retriever.passages}

    def run(self, research: ResearchReport, risk: RiskReport) -> ReviewReport:
        all_sources = [e["source"] for e in research.evidence]
        if risk.assessment:
            all_sources.append(risk.assessment["source"])
        known = bool(all_sources) and all(s in self.allowed_sources for s in all_sources)
        notes = ["Источники в локальной базе знаний" if known else "Нет проверенных источников"]
        if risk.assessment:
            notes.append("Оценка выполнена по синтетическим порогам, требуется проверка человека")
        return ReviewReport(bool(all_sources), known, notes)


class MultiAgentCoordinator:
    def __init__(self, retriever: Retriever):
        self.researcher = ResearchAgent(retriever)
        self.analyst = IndicatorAgent()
        self.reviewer = EvidenceReviewAgent(retriever)

    def run(self, question: str, indicator: str | None = None,
            value: float | None = None) -> dict:
        research = self.researcher.run(question)
        risk = self.analyst.run(indicator, value)
        review = self.reviewer.run(research, risk)
        if not review.claim_sources_known:
            overview = "Нельзя сформировать подтверждённый обзор без источников."
        elif risk.assessment:
            overview = (
                f"Учебный риск по {risk.assessment['indicator']}: "
                f"{risk.assessment['risk_level']}. Требуется проверка эксперта."
            )
        else:
            overview = "Найдены источники; числовая оценка не запрашивалась."
        return {
            "engine": "offline_multi_agent_roles",
            "overview": overview,
            "stages": {
                "researcher": asdict(research),
                "indicator_analyst": asdict(risk),
                "evidence_reviewer": asdict(review),
            },
        }
