"""Reproducible OFFLINE real-LLM prompt comparison on synthetic Russian chats.

Two prompts, same labelled examples, actual Transformers generate(). Not online A/B.
"""
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable
from .experiment import INTENTS, load_cases

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
PROMPT_A = (
    "Classify the banking chat. Return ONLY a JSON object with key intent. "
    "Labels: balance, card_block, transfer_status, loan_payment, other."
)
PROMPT_B = (
    "You are a conservative Russian banking-chat router. Choose one intent: "
    "balance=account balance; card_block=lost/stolen card; "
    "transfer_status=missing transfer; loan_payment=loan payment; other=unclear. "
    "Prioritize lost or stolen card over transfers. "
    "Example: 'Потеряла карту' -> " + '{"intent":"card_block"}' + ". "
    "Example: 'Когда взнос по кредиту?' -> " + '{"intent":"loan_payment"}' + ". "
    'Return ONLY JSON {"intent":"one_of_the_five_labels"}.'
)
PROMPTS_USED = {"A": PROMPT_A, "B": PROMPT_B}

def parse_intent(raw: str):
    """Strict JSON contract. If invalid, abstain; never execute generated content."""
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return "other", False
    if not isinstance(obj, dict) or set(obj) != {"intent"}:
        return "other", False
    label = obj["intent"]
    if not isinstance(label, str) or label not in INTENTS:
        return "other", False
    return label, True

def take_balanced_subset(cases, per_class=2):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    chosen = []
    for label in INTENTS:
        candidates = sorted((c for c in cases if c.intent == label), key=lambda x: x.case_id)
        if len(candidates) < per_class:
            raise ValueError("insufficient cases")
        chosen.extend(candidates[:per_class])
    return chosen

def compute_macro_f1(rows):
    scores = []
    for label in INTENTS:
        tp = sum(r["actual"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["actual"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["actual"] == label and r["prediction"] != label for r in rows)
        scores.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
    return round(statistics.mean(scores), 4)

def summarize(rows):
    if not rows:
        raise ValueError("empty evaluation")
    times = sorted(r["latency_ms"] for r in rows)
    return {
        "cases": len(rows),
        "accuracy": round(sum(r["actual"] == r["prediction"] for r in rows)/len(rows), 4),
        "macro_f1": compute_macro_f1(rows),
        "valid_json_rate": round(sum(r["json_valid"] for r in rows)/len(rows), 4),
        "abstention_rate": round(sum(r["prediction"] == "other" for r in rows)/len(rows), 4),
        "p50_latency_ms": round(statistics.median(times), 2),
        "p95_latency_ms": round(times[min(len(times)-1, math.ceil(.95*len(times))-1)], 2),
    }

def compare(cases, complete: Callable[[str, str], str], per_class=2):
    """Both arms see identical synthetic examples; never include labels in prompts."""
    selected = take_balanced_subset(cases, per_class)
    rows = {"A": [], "B": []}
    for case in selected:
        arms = ("A", "B") if int(case.case_id[-2:])%2 == 0 else ("B", "A")
        for arm in arms:
            started = time.perf_counter()
            raw = complete(PROMPTS_USED[arm], case.text)
            elapsed = (time.perf_counter()-started)*1000
            intent, valid = parse_intent(raw)
            rows[arm].append({
                "case_id": case.case_id, "actual": case.intent,
                "prediction": intent, "json_valid": valid,
                "latency_ms": round(elapsed, 3), "raw_model_output": raw[:300]
            })
    return {
        "method": "Real pretrained LLM offline prompting; NOT an online or causal A/B test",
        "prompts": PROMPTS_USED,
        "results": {arm: summarize(rows[arm]) for arm in ("A", "B")},
        "row_outputs": rows,
        "limitations": "Ten authored synthetic messages, no independent benchmark, no causal A/B effect or generalization claim"
    }

class TransformersCompletion:
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)
        self.model.eval()
    def __call__(self, system, user):
        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        with self.torch.inference_mode():
            output = self.model.generate(inputs, max_new_tokens=32, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs.shape[1]:],
                                     skip_special_tokens=True).strip()

def run():
    llm = TransformersCompletion()
    evaluation = compare(load_cases(ROOT/"data/chat_cases.jsonl"), llm)
    evaluation["model"] = llm.model_id
    evaluation["compute"] = "CPU greedy inference, fixed seed 42"
    dest = ROOT/"reports/real_llm_prompt_eval.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(evaluation,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"model":llm.model_id,"results":evaluation["results"]},ensure_ascii=False))
    return evaluation
if __name__=="__main__":
    run()
