"""Retrieval evaluation: simple Recall@k on small synthetic test set."""
import argparse
import json
from pathlib import Path
from riskops.knowledge import Retriever

ROOT = Path(__file__).resolve().parent.parent


def evaluate(k: int = 3) -> dict:
    retriever = Retriever(backend="tfidf")
    examples = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = 0
    details = []
    for e in examples:
        sources = [h.source for h in retriever.search(e["question"], limit=k)]
        found = e["expected_source"] in sources
        matches += int(found)
        details.append({"question": e["question"], "hit": found, "sources": sources})
    return {"metric": f"Recall@{k}", "score": matches / len(examples), "total": len(examples), "details": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.k), ensure_ascii=False, indent=2))
