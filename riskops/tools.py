"""Only allowlisted, side-effect-free tools are exposed to the agent."""

import json
import math
from pathlib import Path

from riskops.knowledge import PROJECT_ROOT, Retriever

THRESHOLDS_FILE = PROJECT_ROOT / "data" / "risk_thresholds.json"


def assess_indicator(indicator: str, value: float, config: Path = THRESHOLDS_FILE) -> dict:
    """Evaluate ONLY documented synthetic thresholds. Not professional risk advice."""
    policies = json.loads(config.read_text(encoding="utf-8"))
    if indicator not in policies:
        raise ValueError(f"Недопустимый показатель. Доступны: {', '.join(policies)}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value должен быть числом")
    if not math.isfinite(value) or value < 0:
        raise ValueError("value должен быть конечным неотрицательным числом")
    policy = policies[indicator]
    label = "высокий" if value >= policy["high_from"] else (
        "средний" if value >= policy["medium_from"] else "низкий"
    )
    return {
        "indicator": indicator,
        "label": policy["label"],
        "value": value,
        "unit": policy["unit"],
        "risk_level": label,
        "thresholds": {
            "medium_from": policy["medium_from"],
            "high_from": policy["high_from"],
        },
        "source": policy["source"],
        "disclaimer": "Учебная синтетическая шкала, не реальный норматив",
    }


class Tools:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def execute(self, name: str, arguments: dict) -> dict:
        if name == "search_docs":
            q = arguments.get("query", "")
            k = arguments.get("top_k", 3)
            if not isinstance(q, str) or not q.strip():
                raise ValueError("query должен быть непустой строкой")
            if type(k) is not int or not 1 <= k <= 5:
                raise ValueError("top_k должен быть целым числом от 1 до 5")
            return {"hits": [h.to_dict() for h in self.retriever.search(q, k)]}
        if name == "assess_indicator":
            return assess_indicator(arguments.get("indicator", ""), arguments.get("value"))
        raise ValueError(f"Неизвестный инструмент: {name}")


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the synthetic risk policy knowledge base; return cited passages",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_indicator",
            "description": "Calculate a risk level from synthetic published policy thresholds",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["downtime_minutes", "emissions_tons"],
                    },
                    "value": {"type": "number", "minimum": 0},
                },
                "required": ["indicator", "value"],
                "additionalProperties": False,
            },
        },
    },
]
