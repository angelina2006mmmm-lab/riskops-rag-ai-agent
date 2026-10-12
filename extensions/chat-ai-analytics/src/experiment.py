"""Educational offline chat-intent evaluation. No real LLM, customers or A/B traffic."""
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
INTENTS=("balance","card_block","transfer_status","loan_payment","other")
A={"balance":r"\b(?:баланс|остаток)\b","card_block":r"\b(?:заблокировать|блокировка)\b","transfer_status":r"\b(?:перевод|перевести)\b","loan_payment":r"\b(?:кредит|платёж)\b"}
B={"card_block":r"\b(?:заблокировать|блокировка|потерял[аи]?|украли|заморозить|пропала)\b","transfer_status":r"\b(?:перевод|перевести|отправил[аи]?|ушли|дошли|дошёл|поступили|транзакция)\b","loan_payment":r"\b(?:кредит|платёж|погасить|погашение|рассрочка|взнос)\b","balance":r"\b(?:баланс|остаток|сколько денег|доступно|сумма на счёте|сколько на карте)\b"}
PROMPTS={"A":"Classify a chat intent. Return JSON intent.","B":"Classify intent including paraphrases; priority lost-card > transfer > loan > balance. Abstain as other if uncertain. Return JSON intent."}
PII_RE=re.compile(r"\b\d{12,19}\b")
@dataclass(frozen=True)
class Message:
    case_id:str
    text:str
    intent:str
    priority:str
class Classifier(Protocol):
    def predict(self,text:str)->str:...
class PromptPolicy:
    def __init__(self,variant):
        if variant not in ("A","B"):raise ValueError("variant")
        self.rules=A if variant=="A" else B
    def predict(self,text):
        if PII_RE.search(text):return "other"
        for cls,pattern in self.rules.items():
            if re.search(pattern,text.lower().replace("ё","е")):return cls
        return "other"
def load_cases(path):
    cases=[Message(**json.loads(s)) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not cases or len(set(x.case_id for x in cases))!=len(cases):raise ValueError("empty or duplicate cases")
    return cases
def metrics(cases:list[Message],model:Classifier):
    actual=[c.intent for c in cases];pred=[model.predict(c.text) for c in cases]
    class_metrics={};f1s=[]
    for cls in INTENTS:
        tp=sum(a==cls and b==cls for a,b in zip(actual,pred))
        fp=sum(a!=cls and b==cls for a,b in zip(actual,pred))
        fn=sum(a==cls and b!=cls for a,b in zip(actual,pred))
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1s.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
        class_metrics[cls]={"precision":round(precision,4),"recall":round(recall,4),"support":sum(a==cls for a in actual)}
    return {"n":len(cases),"accuracy":round(sum(a==b for a,b in zip(actual,pred))/len(cases),4),
            "macro_f1":round(sum(f1s)/len(f1s),4),"classes":class_metrics,
            "confusion":{a:{b:sum(x==a and y==b for x,y in zip(actual,pred)) for b in INTENTS} for a in INTENTS}}
def stable_arm(case_id:str):return "A" if int(hashlib.sha256(case_id.encode()).hexdigest()[:8],16)%2==0 else "B"
def extract_amount(text):
    m=re.search(r"(?<!\d)(\d{1,6})\s*(?:₽|руб(?:лей|ля|\.)?)(?!\w)",text,re.I)
    return int(m.group(1)) if m else None
def evaluate(cases):
    return {"method":"Deterministic routing mock, NOT live LLM prompts or online A/B","dataset":"100 synthetic messages",
            "A":metrics(cases,PromptPolicy("A")),"B":metrics(cases,PromptPolicy("B")),
            "limitations":"Synthetic paraphrase templates favor variant B; no claims of model generalization."}
def run():
    root=Path(__file__).resolve().parents[1]
    r=evaluate(load_cases(root/"data/chat_cases.jsonl"))
    (root/"reports").mkdir(exist_ok=True)
    (root/"reports/offline_prompt_eval.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:r[k]["macro_f1"] for k in ("A","B")},ensure_ascii=False))
if __name__=="__main__":run()
