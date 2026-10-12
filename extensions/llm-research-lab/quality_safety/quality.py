"""Reproducible synthetic RAG-evaluation suite; NOT a live-LLM safety benchmark."""
import json,re
from pathlib import Path
from datetime import date
from collections import Counter

ROOT=Path(__file__).resolve().parent
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def active_policy(rows,as_of):
    cutoff=date.fromisoformat(as_of)
    current={}
    for row in rows:
        if date.fromisoformat(row['valid_from'])>cutoff:continue
        old=current.get(row['id'])
        if old is None or (row['valid_from'],row['version'])>(old['valid_from'],old['version']):
            current[row['id']]=row
    return current

def retrieve(rows,question,as_of,k=3):
    if not 1<=k<=10:raise ValueError('Invalid k')
    words=set(re.findall(r'[a-zа-яё_0-9]+',question.lower()))
    scored=[]
    for policy in active_policy(rows,as_of).values():
        score=sum(len(words.intersection(re.findall(r'[a-zа-яё_0-9]+',term.lower())))>0 for term in policy['entities'])
        if score:scored.append((score,policy))
    scored.sort(key=lambda x:(-x[0],x[1]['id']))
    return [p for _,p in scored[:k]]

def retrieval_report(policies,cases,k=3):
    if not cases:raise ValueError('No evaluation cases')
    sums=Counter()
    for case in cases:
        got=retrieve(policies,case['query'],case['as_of'],k)
        actual=[x['id'] for x in got];expected=set(case['expected_ids'])
        hit=set(actual)&expected
        sums['recall']+=len(hit)/len(expected) if expected else float(not actual)
        sums['precision']+=len(hit)/len(actual) if actual else float(not expected)
        sums['mrr']+=next((1/rank for rank,key in enumerate(actual,1) if key in expected),0) if expected else float(not actual)
        sums['abstain']+=float((not actual)==case['abstain'])
    n=len(cases)
    return {'cases':n,'recall_at_3':round(sums['recall']/n,4),
            'precision_at_3':round(sums['precision']/n,4),
            'mrr_at_3':round(sums['mrr']/n,4),
            'abstention_accuracy':round(sums['abstain']/n,4)}

PATTERNS=[
 r'ignore (?:all )?(?:previous|prior) instructions',
 r'отмени (?:все )?предыдущие инструкции',
 r'print (?:the )?(?:system prompt|api[_ -]?key)',
 r'раскрой (?:системный промпт|секретный ключ)',
 r'execute (?:shell|python|code)',r'run (?:rm -rf|curl .+\\| ?bash)',
 r'override (?:system|developer) (?:instructions|message)',
 r'you are now (?:system|developer)']

def blocked(text):
    return any(re.search(p,text,re.I) for p in PATTERNS)

def security_report(cases):
    if not cases:raise ValueError('No attacks')
    tp=fp=tn=fn=0
    for row in cases:
        attack=row['is_attack'];pred=blocked(row['text'])
        if attack and pred:tp+=1
        elif attack:fn+=1
        elif pred:fp+=1
        else:tn+=1
    return {'cases':len(cases),'attack_samples':tp+fn,'attack_detected':tp,
      'benign_samples':tn+fp,'benign_false_positive':fp,
      'attack_block_rate':round(tp/(tp+fn),4) if tp+fn else None,
      'benign_false_positive_rate':round(fp/(tn+fp),4) if tn+fp else None,
      'scope':'Heuristic checks, NOT reliable against adaptive attackers'}

def citation_audit(claims,evidence,as_of,policies):
    active=active_policy(policies,as_of)
    valid={key for key,value in active.items() if any(d['id']==key and d['version']==value['version'] for d in evidence)}
    supported=sum(bool(c.get('citations')) and set(c['citations']).issubset(valid) for c in claims)
    return {'claims':len(claims),'with_valid_source_ids':supported,
            'unsupported':len(claims)-supported,'scope':'Checks IDs/versions; NOT semantic entailment'}

def run():
    policies=json.loads((ROOT/'policies.json').read_text(encoding='utf-8'))
    cases=read_jsonl(ROOT/'retrieval_cases.jsonl')
    attacks=read_jsonl(ROOT/'security_cases.jsonl')
    return {'retrieval':retrieval_report(policies,cases),
      'prompt_injection':security_report(attacks),
      'citation_check':citation_audit(
         [{'citations':['SUP-01']},{'citations':[]}],
         retrieve(policies,'supplier','2026-10-01'),'2026-10-01',policies),
      'limitations':'Synthetic, hand-authored evaluation. No generated answer factuality or production security claims'}

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
