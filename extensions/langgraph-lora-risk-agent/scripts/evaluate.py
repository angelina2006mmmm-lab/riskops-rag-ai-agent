"""Small, synthetic-only evaluation of deterministic workflow and evidence links."""
import json
from pathlib import Path
from risk_agent.graph import run_offline

def run(path="eval/questions.jsonl"):
 rows=[json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
 status_ok=0;source_ok=0;source_cases=0;failures=[]
 for row in rows:
  output=run_offline(row)
  matched=output["status"]==row["expected_status"]
  status_ok+=matched
  if "expected_source" in row:
   source_cases+=1
   has_source=any(d["source"]==row["expected_source"] for d in output["evidence"])
   source_ok+=has_source
   matched=matched and has_source
  if not matched:failures.append({"question":row["question"],"actual":output["status"]})
 return {"cases":len(rows),"status_accuracy":status_ok/len(rows),"citation_recall":source_ok/source_cases,"citation_cases":source_cases,"failures":failures,"note":"Hand-authored synthetic examples, not real-world LLM performance."}
if __name__=="__main__":
 print(json.dumps(run(),ensure_ascii=False,indent=2))
