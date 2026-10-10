from riskops.llm import ChatCompletionClient, LLMConfigurationError


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents(client):
    r = client.get("/documents")
    assert r.json()["count"] == 5


def test_search_api(client):
    r = client.post("/search", json={"query": "риск простоя"})
    assert r.status_code == 200
    assert r.json()["hits"]


def test_search_limits(client):
    assert client.post("/search", json={"query": "risk", "top_k": 99}).status_code == 422


def test_assess_api(client):
    r = client.post("/assess", json={"indicator": "emissions_tons", "value": 55})
    assert r.status_code == 200
    assert r.json()["risk_level"] == "высокий"


def test_assess_rejects_nan(client):
    r = client.post("/assess", json={"indicator": "emissions_tons", "value": -10})
    assert r.status_code == 422


def test_ask_offline(client):
    r = client.post("/ask", json={"question": "Сколько минут простой?", "mode": "offline"})
    assert r.status_code == 200
    assert r.json()["engine"] == "offline_extract"
    assert "sources" in r.json()


def test_ask_indicator_requires_value(client):
    r = client.post("/ask", json={"question": "простой", "indicator": "downtime_minutes"})
    assert r.status_code == 422


def test_llm_requires_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "Политика ESG", "mode": "llm"})
    assert r.status_code == 400
    assert "OPENAI_API_KEY" in r.json()["detail"]
