"""A CPU smoke test with the *actual TRL GRPOTrainer*, not a reasoning benchmark.

1-2 optimizer steps on a public tiny pretrained model and artificial rewards.
No evidence of useful real-world GRPO alignment or production-level performance.
"""
import json,os
from pathlib import Path

MODEL="sshleifer/tiny-gpt2"
SAMPLES=[
 {"prompt":"Fictional policy: supplier outage, should we review? Answer:","expected":"review"},
 {"prompt":"Fictional policy: missing policy evidence, what to do? Answer:","expected":"abstain"},
 {"prompt":"Fictional policy: invalid risk source, what to do? Answer:","expected":"abstain"},
 {"prompt":"Fictional policy: high incident severity, what to do? Answer:","expected":"review"},
]

def reward(completions,expected,**kwargs):
    """Simple token matching; deliberately not a sophisticated reward model."""
    assert len(completions)==len(expected)
    return [float(target in str(comp).lower()) + 0.1 * len(set(str(comp).strip().lower())) / max(1, len(str(comp).strip())) for comp,target in zip(completions,expected)]

def train(max_steps=4,report_path="reports/grpo_trl_smoke.json"):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig,GRPOTrainer
    torch.manual_seed(37);torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token=tok.eos_token;tok.padding_side="left"
    args=GRPOConfig(
        output_dir="/tmp/synthetic-risk-grpo",
        max_steps=max_steps,
        num_generations=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_prompt_length=64,
        max_completion_length=12,
        learning_rate=5e-5,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        gradient_checkpointing=False,
        use_vllm=False,
        bf16=False,fp16=False,
    )
    trainer=GRPOTrainer(model=MODEL,reward_funcs=reward,args=args,
                        train_dataset=Dataset.from_list(SAMPLES),
                        processing_class=tok)
    outcome=trainer.train()
    if int(outcome.global_step)<1:raise RuntimeError("GRPO optimizer did not complete")
    result={"trainer":"trl.GRPOTrainer","model":MODEL,"steps":int(outcome.global_step),
            "train_loss":float(outcome.training_loss),
            "samples":len(SAMPLES),"num_generations":2,
            "scope":"CPU smoke test on synthetic preferences; no general reasoning improvements shown"}
    target=Path(report_path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=="__main__":train(max_steps=int(os.environ.get("GRPO_STEPS","4")))
