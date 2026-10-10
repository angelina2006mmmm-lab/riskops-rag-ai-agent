import pytest
from fastapi.testclient import TestClient

from riskops.api import app, get_agent
from riskops.agent import RiskAgent
from riskops.knowledge import Retriever
from riskops.tools import Tools


@pytest.fixture
def agent():
    return RiskAgent(Tools(Retriever(backend="tfidf")))


@pytest.fixture
def client(agent):
    app.dependency_overrides.clear()
    get_agent.cache_clear()
    # cached factory normally used; tests run against the same offline documents
    with TestClient(app) as client:
        yield client
    get_agent.cache_clear()
