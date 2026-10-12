import json
from pathlib import Path
from .agent import assess
CASES=Path(__file__).resolve().parents[1]/"eval"/"cases.jsonl"
def evaluate():
    cases=[json.loads(s) for s in CASES.read_text(encoding="utf-8").splitlines() if s.strip()]
    correct=0
    for row in cases:
        out=assess(row["question"],row["indicator"],row["value"],row["as_of"])
        correct+=out.status==row["expected"] and ("version" not in row or out.version==row["version"])
    return {"cases":len(cases),"passed":correct,"scenario_accuracy":correct/len(cases),
            "disclaimer":"Hand-authored synthetic functional checks; no inference about model quality."}
if __name__=="__main__":print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
