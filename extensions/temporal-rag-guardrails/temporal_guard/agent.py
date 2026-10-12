"""Offline temporal retrieval with source/version checks and refusal paths."""
from dataclasses import dataclass,asdict
from datetime import date
from .policies import select_policies
from .guards import looks_like_injection,valid_number
@dataclass(frozen=True)
class RiskResult:
    status: str
    conclusion: str
    source_id: str | None = None
    version: int | None = None
    effective_from: str | None = None
    threshold: float | None = None
    reason: str | None = None
    def to_dict(self): return asdict(self)
CUES = {
 "downtime_minutes":("downtime","простой","недоступ"),
 "fraud_alerts":("fraud","мошенн","подозр"),
 "supplier_outage_minutes":("supplier","поставщик"),
 "access_incidents":("access","доступ"),
 "training_hours":("training","обучен"),
}
def assess(question: str,indicator: str,value,as_of: str) -> RiskResult:
    if not isinstance(question,str) or not question.strip():
        return RiskResult("abstain","Нет содержательного запроса.",reason="empty_question")
    if looks_like_injection(question):
        return RiskResult("blocked","Запрос содержит подозрительные инструкции.",reason="user_prompt_pattern")
    try:
        date.fromisoformat(as_of)
        policies = select_policies(as_of)
    except (ValueError,TypeError):
        return RiskResult("abstain","Некорректная дата.",reason="invalid_date")
    if not valid_number(value):
        return RiskResult("abstain","Некорректное числовое значение.",reason="invalid_value")
    matching=[x for x in policies.values() if x["indicator"]==indicator]
    if len(matching)!=1:
        return RiskResult("abstain","Для показателя не найдено актуальной политики.",reason="missing_policy")
    policy=matching[0]
    if looks_like_injection(policy["text"]) or policy.get("trust")=="untrusted":
        return RiskResult("blocked","Ненадёжный источник исключён из оценки.",reason="retrieved_document_injection")
    if not any(c in question.lower() for c in CUES.get(indicator,())):
        return RiskResult("abstain","Запрос и показатель не совпадают.",reason="topic_mismatch")
    grade = "high" if value>policy["threshold"] else "review"
    return RiskResult("grounded",f'Учебная оценка: {grade}; по политике {policy["id"]} v{policy["version"]}.',
          source_id=policy["id"],version=policy["version"],effective_from=policy["effective_from"],threshold=policy["threshold"])
