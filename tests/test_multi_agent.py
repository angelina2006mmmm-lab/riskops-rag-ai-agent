from riskops.knowledge import Retriever
from riskops.multi_agent import MultiAgentCoordinator


def test_roles_work_together():
    result = MultiAgentCoordinator(Retriever()).run(
        "Простой сервиса и риски", "downtime_minutes", 130
    )
    assert result["engine"] == "offline_multi_agent_roles"
    assert result["stages"]["indicator_analyst"]["assessment"]["risk_level"] == "высокий"
    assert result["stages"]["evidence_reviewer"]["claim_sources_known"]


def test_no_risk_indicator_requested():
    result = MultiAgentCoordinator(Retriever()).run("Эскалация инцидента")
    assert result["stages"]["indicator_analyst"]["assessment"] is None


def test_api_multi_agent(client):
    r = client.post("/multi-agent/brief", json={"question": "Порог выбросов CO2e",
                    "indicator": "emissions_tons", "value": 51})
    assert r.status_code == 200
    assert r.json()["stages"]["indicator_analyst"]["assessment"]["risk_level"] == "высокий"
