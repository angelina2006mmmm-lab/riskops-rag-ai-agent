from pathlib import Path
import pytest

from riskops.knowledge import Retriever, load_passages


def test_five_markdown_sources():
    docs = load_passages()
    assert len(set(p.source for p in docs)) == 5
    assert len({p.id for p in docs}) == len(docs)


def test_retrieve_operational_threshold():
    hits = Retriever(backend="tfidf").search("простой 120 минут высокий риск", limit=3)
    assert hits
    assert any(h.source == "01_operational_policy.md" for h in hits)


def test_retrieve_esg_threshold():
    hits = Retriever(backend="tfidf").search("выбросы CO2e 50 тонны", limit=5)
    assert any(h.source == "02_esg_policy.md" for h in hits)


def test_citations_and_order():
    hits = Retriever(backend="tfidf").search("инцидент проверка логов", limit=4)
    assert all(h.id and h.source and h.heading for h in hits)
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits)-1))


def test_blank_query():
    assert Retriever().search("    ") == []


@pytest.mark.parametrize("count", [0, 9, -1])
def test_invalid_k(count):
    with pytest.raises(ValueError):
        Retriever().search("простой", limit=count)


def test_missing_documents(tmp_path: Path):
    with pytest.raises(ValueError):
        load_passages(tmp_path)
