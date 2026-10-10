import pytest

from riskops.knowledge import Retriever
from riskops.tools import Tools, assess_indicator


@pytest.mark.parametrize("minutes,expected", [
    (0, "низкий"), (29.99, "низкий"), (30, "средний"),
    (119.9, "средний"), (120, "высокий"), (250, "высокий"),
])
def test_downtime_limits(minutes, expected):
    got = assess_indicator("downtime_minutes", minutes)
    assert got["risk_level"] == expected
    assert got["source"] == "01_operational_policy.md"


@pytest.mark.parametrize("value,expected", [(0, "низкий"), (20, "средний"), (50, "высокий")])
def test_emissions(value, expected):
    assert assess_indicator("emissions_tons", value)["risk_level"] == expected


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "2", True])
def test_invalid_value(bad):
    with pytest.raises(ValueError):
        assess_indicator("downtime_minutes", bad)


def test_unknown_indicator():
    with pytest.raises(ValueError):
        assess_indicator("unknown", 2)


def test_allowlisted_tool_only():
    tools = Tools(Retriever())
    with pytest.raises(ValueError):
        tools.execute("run_shell", {"command": "rm -rf /"})


def test_search_tool():
    out = Tools(Retriever()).execute("search_docs", {"query": "эскалация простоя", "top_k": 2})
    assert 1 <= len(out["hits"]) <= 2
    assert "source" in out["hits"][0]


def test_bad_tool_args():
    with pytest.raises(ValueError):
        Tools(Retriever()).execute("search_docs", {"query": "a", "top_k": 100})
