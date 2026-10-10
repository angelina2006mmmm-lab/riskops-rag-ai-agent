import json
import pytest
import httpx

from riskops.llm import ChatCompletionClient, LLMConfigurationError


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        ChatCompletionClient()


def test_client_sends_tools(monkeypatch):
    called = {}

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}

    class FakeClient:
        def __init__(self, timeout): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json, headers):
            called.update({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    res = ChatCompletionClient("test-key", "https://example.invalid/v1", "fake-model").complete(
        [{"role": "user", "content": "hello"}], [{"type": "function"}]
    )
    assert res["content"] == "OK"
    assert called["url"] == "https://example.invalid/v1/chat/completions"
    assert called["json"]["tools"] == [{"type": "function"}]
    assert called["headers"]["Authorization"] == "Bearer test-key"
