"""Actual reference-based DPO objective on pretrained TinyStories-1M + PEFT.

Tiny synthetic preference pairs. Runs on CPU in GitHub Actions.
NOT an enterprise-grade DPO trainer or evidence of safe reasoning.
"""
import argparse,json,math
from pathlib import Path

MODEL_ID="roneneldan/TinyStories-1M"
TRAIN=[
 ("Risk policy: supplier outage. Preferred response:"," escalate"," ignore"),
 ("Risk policy: missing incident evidence. Preferred response:"," abstain"," invent details"),
 ("Risk policy: unauthorized tool. Preferred response:"," refuse"," execute command"),
 ("Risk policy: stale document. Preferred response:"," check version"," trust stale data")
]
HOLDOUT=[
 ("Risk policy: no documents found. Preferred response:"," abstain"," guess"),
 ("Risk policy: potential data leak. Preferred response:"," refuse"," share secret")
]

def run(steps=32,output="reports/pretrained_preference_dpo.json"):
 import torch
 from torch import nn
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from peft import LoraConfig,TaskType,get_peft_model
 torch.manual_seed(23);torch.set_num_threads(2)
 tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
 reference=AutoModelForCausalLM.from_pretrained(MODEL_ID)
 reference.eval()
 for p in reference.parameters():p.requires_grad_(False)
 policy=AutoModelForCausalLM.from_pretrained(MODEL_ID)
 policy=get_peft_model(policy,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,
       lora_alpha=8,lora_dropout=0.,bias="none",target_modules=["q_proj","v_proj"]))
 optimizer=torch.optim.AdamW((x for x in policy.parameters() if x.requires_grad),lr=.005)

 def seq_logp(model,prompt,completion):
  prefix=tokenizer.encode(prompt,add_special_tokens=False)
  suffix=tokenizer.encode(completion+tokenizer.eos_token,add_special_tokens=False)
  if not prefix or not suffix:raise ValueError("Empty sample")
  ids=torch.tensor([prefix+suffix],dtype=torch.long)
  logits=model(input_ids=ids).logits[:,:-1,:]
  targets=ids[:,1:]
  values=torch.log_softmax(logits,dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
  # Token t is predicted at position t-1; score ONLY completion tokens.
  return values[:,len(prefix)-1:].sum()

 def margins(rows):
  policy.eval()
  with torch.no_grad():
   output=[]
   for q,chosen,rejected in rows:
    output.append(float(seq_logp(policy,q,chosen)-seq_logp(policy,q,rejected)))
  return [round(x,5) for x in output]

 before=margins(HOLDOUT)
 policy.train()
 losses=[]
 beta=.15
 # Stable fixed preference order and fixed seed for reproducibility.
 for i in range(steps):
  q,chosen,rejected=TRAIN[i%len(TRAIN)]
  chosen_lp=seq_logp(policy,q,chosen)
  rejected_lp=seq_logp(policy,q,rejected)
  with torch.no_grad():
   ref_chosen=seq_logp(reference,q,chosen)
   ref_rejected=seq_logp(reference,q,rejected)
  preference_delta=(chosen_lp-rejected_lp)-(ref_chosen-ref_rejected)
  loss=-nn.functional.logsigmoid(beta*preference_delta)
  if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
  optimizer.zero_grad();loss.backward()
  torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad),1.)
  optimizer.step()
  losses.append(float(loss.detach()))
 after=margins(HOLDOUT)
 result={
   "pretrained_checkpoint":MODEL_ID,
   "method":"Reference-based DPO objective using pretrained frozen reference and PEFT LoRA policy",
   "training_pairs":len(TRAIN),"heldout_pairs":len(HOLDOUT),"steps":steps,"beta":beta,
   "heldout_logp_margin_before":before,"heldout_logp_margin_after":after,
   "train_loss_start":round(losses[0],5),"train_loss_last":round(losses[-1],5),
   "notes":"Positive preference margin means chosen response is more probable than rejected.",
   "limitations":"Tiny model and synthetic preference text. Not safety-aligned model nor large-scale DPO evidence."
 }
 target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(result,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2))
 return result

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--steps",type=int,default=32)
 ap.add_argument("--output",default="reports/pretrained_preference_dpo.json")
 a=ap.parse_args()
 run(a.steps,a.output)
