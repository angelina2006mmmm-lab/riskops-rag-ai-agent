from pathlib import Path
import pytest
from src.experiment import load_cases,PromptPolicy,metrics,evaluate,stable_arm,extract_amount,PII_RE
from src.analytics import build_db,monitor
from src.llm_adapter import RoutingService
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def cases():return load_cases(ROOT/"data/chat_cases.jsonl")
def test_balanced_fixture(cases):
    assert len(cases)==100 and len({x.case_id for x in cases})==100
    assert all(sum(c.intent==x for c in cases)==20 for x in {c.intent for c in cases})
def test_metrics_better_on_authored_fixture(cases):
    x=evaluate(cases)
    assert x["A"]["macro_f1"]<x["B"]["macro_f1"]
def test_metrics_confusion_total(cases):
    report=metrics(cases,PromptPolicy("B"))
    assert sum(sum(x.values()) for x in report["confusion"].values())==100
def test_card_safety_priority():
    assert PromptPolicy("B").predict("У меня украли карту и был перевод")=="card_block"
def test_abstain_on_ambiguous_question():
    assert PromptPolicy("B").predict("Где мои деньги?")=="other"
def test_amount_rubles():
    assert extract_amount("Перевела 1200 руб.")==1200
def test_amount_symbol():
    assert extract_amount("Внести 890₽")==890
def test_amount_missing():
    assert extract_amount("Сколько денег?") is None
def test_pii_guard():
    assert PII_RE.search("1234567890123456")
    assert PromptPolicy("B").predict("Баланс 1234567890123456")=="other"
def test_stable_arm():
    assert stable_arm("same_user")==stable_arm("same_user")
def test_both_arms_present():
    assert {stable_arm(str(i)) for i in range(100)}=={"A","B"}
def test_variant_rejection():
    with pytest.raises(ValueError):PromptPolicy("C")
def test_mock_structured_llm():
    class Mock:
        def complete(self,system_prompt,user_message):return '{"intent":"card_block"}'
    assert RoutingService(Mock()).route("Потеряла карту").intent=="card_block"
def test_mock_invalid_json():
    class Mock:
        def complete(self,system_prompt,user_message):return 'not json'
    assert RoutingService(Mock()).route("Потеряла карту").intent=="other"
def test_mock_invalid_intent():
    class Mock:
        def complete(self,system_prompt,user_message):return '{"intent":"admin"}'
    assert RoutingService(Mock()).route("Потеряла карту").intent=="other"
def test_sql_monitoring_window(cases):
    db=build_db(cases)
    rows=monitor(db,ROOT/"sql/chat_monitor.sql")
    assert len(rows)==20 and sum(x["sessions"] for x in rows)==100
    assert all(0<=x["running_accuracy"]<=1 for x in rows)
def test_synthetic_events_no_raw_chat(cases):
    db=build_db(cases)
    cols=[r["name"] for r in db.execute("PRAGMA table_info(outcomes)")]
    assert "text" not in cols and "raw_chat" not in cols
