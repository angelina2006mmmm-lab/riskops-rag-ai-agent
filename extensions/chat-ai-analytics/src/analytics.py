"""Synthetic SQLite chat monitoring: window functions, REGEXP, subqueries."""
import sqlite3,re,json
from pathlib import Path
from .experiment import load_cases,stable_arm,PromptPolicy
def build_db(cases):
    db=sqlite3.connect(":memory:");db.row_factory=sqlite3.Row
    db.create_function("REGEXP",2,lambda pat,s:bool(re.search(pat,s or "")))
    db.execute("""CREATE TABLE outcomes(
     id TEXT PRIMARY KEY,arm TEXT,actual TEXT,predicted TEXT,correct INTEGER,
     day TEXT,latency_ms REAL,escalated INTEGER,error_code TEXT)""")
    rows=[]
    for i,c in enumerate(cases):
        arm=stable_arm(c.case_id);pred=PromptPolicy(arm).predict(c.text)
        rows.append((c.case_id,arm,c.intent,pred,int(pred==c.intent),
           f"2026-10-{(i%10)+1:02d}",round(35+(i*17)%72,2),
           int(pred=="other" and c.intent!="other"),
           "SYNTHETIC_MISROUTE" if pred!=c.intent else "NONE"))
    db.executemany("INSERT INTO outcomes VALUES(?,?,?,?,?,?,?,?,?)",rows)
    return db
def monitor(db,query_path):
    return [dict(r) for r in db.execute(Path(query_path).read_text(encoding="utf-8"))]
def run():
    root=Path(__file__).resolve().parents[1]
    db=build_db(load_cases(root/"data/chat_cases.jsonl"))
    records=monitor(db,root/"sql/chat_monitor.sql")
    (root/"reports").mkdir(exist_ok=True)
    (root/"reports/monitoring.json").write_text(json.dumps({"results":records,"data":"simulated events and latency"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Rows: {len(records)}")
if __name__=="__main__":run()
