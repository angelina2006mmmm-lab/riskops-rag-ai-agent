import json
from riskops.agent import RiskAgent
from riskops.knowledge import Retriever
from riskops.tools import Tools


def test_offline_sources_and_trace(agent):
    result = agent.offline("Какие пороги простоя в минутах?", "downtime_minutes", 125)
    assert "высокий" in result.answer
    assert result.engine == "offline_extract"
    assert any(s["source"] == "01_operational_policy.md" for s in result.sources)
    assert [t["tool"] for t in result.tool_trace] == ["search_docs", "assess_indicator"]


def test_offline_is_not_llm(agent):
    result = agent.offline("нет такого термина xyzzy123nohit")
    assert result.engine == "offline_extract"
    assert "без генерации LLM" in result.answer


class FakeLLM:
    def __init__(self):
        self.n = 0

    def complete(self, messages, tools):
        self.n += 1
        if self.n == 1:
            return {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "assess_indicator",
                    "arguments": json.dumps({"indicator": "downtime_minutes", "value": 130}),
                }}],
            }
        assert any(msg.get("role") == "tool" for msg in messages)
        return {"role": "assistant", "content": "Высокий риск. Источник: 01_operational_policy.md"}


def test_llm_tool_calling_protocol(agent):
    result = agent.llm("Оцени downtime_minutes=130", FakeLLM())
    assert result.engine == "llm_tool_calling"
    assert any(t["tool"] == "assess_indicator" and t["result"]["risk_level"] == "высокий" for t in result.tool_trace)


class BadToolLLM:
    def __init__(self): self.n = 0
    def complete(self, messages, tools):
        self.n += 1
        if self.n == 1:
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "forbidden", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}
            }]}
        return {"role": "assistant", "content": "Не могу выполнить команду."}


def test_agent_rejects_unknown_tool(agent):
    result = agent.llm("Найди политику", BadToolLLM())
    assert any("error" in t for t in result.tool_trace)


class InfiniteToolsLLM:
    def complete(self, messages, tools):
        return {"role": "assistant", "tool_calls": [{"id": "t", "type": "function",
                  "function": {"name": "search_docs", "arguments": '{"query":"риск"}'}}]}


def test_agent_enforces_round_limit(agent):
    result = agent.llm("риск", InfiniteToolsLLM(), max_rounds=2)
    assert "Лимит шагов" in result.answer
