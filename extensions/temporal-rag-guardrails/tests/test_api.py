from fastapi.testclient import TestClient
from temporal_guard.api import app
client=TestClient(app)
def test_health():
    assert client.get("/health").json()=={"ok":True}
def test_endpoint():
    r=client.post("/assess",json={"question":"IT downtime","indicator":"downtime_minutes","value":130,"as_of":"2026-10-01"})
    assert r.status_code==200 and r.json()["version"]==2
def test_validation():
    assert client.post("/assess",json={"question":"","indicator":"downtime_minutes","value":130,"as_of":"2026-10-01"}).status_code==422
