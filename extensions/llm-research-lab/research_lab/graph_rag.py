"""Version-aware entity graph retrieval with provenance; no LLM/vector embeddings."""
from collections import defaultdict,deque
from datetime import date
import json,re
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"data"/"policies.json"
def tokens(s):return set(re.findall(r"[a-zа-яё0-9_-]+",s.lower()))
def load_policies(as_of="2026-10-01",file=DATA):
 cutoff=date.fromisoformat(as_of)
 rows=json.loads(Path(file).read_text(encoding="utf-8"));latest={}
 for row in rows:
  start=date.fromisoformat(row["effective_from"])
  if start>cutoff:continue
  old=latest.get(row["id"])
  if old is None or (start,row["version"])>(date.fromisoformat(old["effective_from"]),old["version"]):latest[row["id"]]=row
 return latest
def build_graph(policies):
 adjacency=defaultdict(set)
 for doc in policies.values():
  pn="policy:"+doc["id"]
  for entity in doc["entities"]:
   en="entity:"+entity
   adjacency[pn].add(en);adjacency[en].add(pn)
 return adjacency
def retrieve(question,as_of="2026-10-01",max_hops=3,top_k=3):
 if not question.strip():return []
 policies=load_policies(as_of);adj=build_graph(policies);query=tokens(question)
 starts=[node for node in adj if node.startswith("entity:") and tokens(node[7:])&query]
 if not starts:return []
 visited=set(starts);queue=deque((node,0,[node]) for node in starts);found={}
 while queue:
  node,depth,path=queue.popleft()
  if node.startswith("policy:"):
   key=node[7:]
   row={"id":key,"version":policies[key]["version"],"effective_from":policies[key]["effective_from"],"text":policies[key]["text"],"path":path,"hops":depth}
   if key not in found or depth<found[key]["hops"]:found[key]=row
  if depth>=max_hops:continue
  for other in sorted(adj[node]):
   if other not in visited:
    visited.add(other);queue.append((other,depth+1,path+[other]))
 return sorted(found.values(),key=lambda x:(x["hops"],x["id"]))[:top_k]
def research_answer(question,as_of="2026-10-01"):
 evidence=retrieve(question,as_of)
 return {"status":"grounded" if evidence else "abstain","evidence":evidence,"model":"symbolic multi-hop retrieval; no LLM"}
def evaluate(file=None):
 f=Path(file) if file else Path(__file__).resolve().parents[1]/"data"/"eval.jsonl"
 rows=[json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
 correct=0
 for row in rows:
  res=research_answer(row["question"],row["as_of"])
  ids=[x["id"] for x in res["evidence"]]
  correct+=res["status"]==row["expected_status"] and (not row.get("expected_source") or row["expected_source"] in ids)
 return {"cases":len(rows),"correct":correct,"scenario_accuracy":round(correct/len(rows),3),"limits":"Hand-authored synthetic functional evaluation, not LLM performance"}
if __name__=="__main__":print(json.dumps(evaluate(),indent=2))
