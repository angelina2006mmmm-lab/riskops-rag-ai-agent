"""Fully offline demonstration, no HTTP server or model API needed."""
import json
from riskops.knowledge import Retriever
from riskops.tools import Tools
from riskops.agent import RiskAgent

if __name__ == "__main__":
    agent = RiskAgent(Tools(Retriever()))
    result = agent.offline("Как оценивается простой сервиса 135 минут?", "downtime_minutes", 135)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
