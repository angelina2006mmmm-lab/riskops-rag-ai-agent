"""Constrained GRPO-style updates on pretrained TinyStories-1M, with exploration.

Synthetic high/low risk binary decision, not natural-language reasoning.
Grouped advantage + PPO clipping + reference KL; explicitly not TRL GRPOTrainer.
"""
from pathlib import Path
import argparse,json
def experiment(steps=120,output="reports/pretrained_group_reward_v2.json"):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import LoraConfig,TaskType,get_peft_model
 torch.manual_seed(41);torch.set_num_threads(2)
 model_id="roneneldan/TinyStories-1M"
 tok=AutoTokenizer.from_pretrained(model_id)
 tok.pad_token=tok.eos_token;tok.padding_side="left"
 candidate_tokens=[tok.encode(" yes",add_special_tokens=False),tok.encode(" no",add_special_tokens=False)]
 if not all(len(x)==1 for x in candidate_tokens):raise ValueError("Single-token action mapping missing")
 ids=[x[0] for x in candidate_tokens]
 train_prompts=[
  "A loan is high risk. Does it need manual review? Answer:",
  "A loan is low risk. Does it need manual review? Answer:",
  "A transaction has high risk. Escalate? Answer:",
  "A transaction has low risk. Escalate? Answer:",
  "A supplier has high risk. Escalate? Answer:",
  "A supplier has low risk. Escalate? Answer:"
 ]
 heldout=[
  "This credit case is high risk. Should we escalate? Answer:",
  "This credit case is low risk. Should we escalate? Answer:",
  "This operational event has high risk. Escalate? Answer:",
  "This operational event has low risk. Escalate? Answer:"
 ]
 train_truth=torch.tensor([0,1,0,1,0,1])
 held_truth=torch.tensor([0,1,0,1])
 base=AutoModelForCausalLM.from_pretrained(model_id)
 def logits(model,prompts):
  inputs=tok(prompts,return_tensors="pt",padding=True)
  return model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"]).logits[:,-1,ids]/4.0
 with torch.no_grad():
  ref_probs=torch.softmax(logits(base,train_prompts),dim=-1)
 model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,
    r=8,lora_alpha=16,lora_dropout=0,target_modules=["q_proj","v_proj"],bias="none"))
 optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.018)
 def accuracy(prompts,labels):
  model.eval()
  with torch.no_grad():
   return round(float((logits(model,prompts).argmax(-1)==labels).float().mean()),4)
 before_train=accuracy(train_prompts,train_truth)
 before_val=accuracy(heldout,held_truth)
 for _ in range(steps):
  model.train()
  with torch.no_grad():
   old_policy=torch.softmax(logits(model,train_prompts),dim=-1)
   behavior=.85*old_policy+.15/2
   actions=torch.multinomial(behavior,8,replacement=True)
   prior_logp=torch.log(behavior.gather(1,actions).clamp_min(1e-8))
   reward=(actions==train_truth[:,None]).float()
   advantage=(reward-reward.mean(1,keepdim=True))/reward.std(1,keepdim=True,unbiased=False).clamp_min(.2)
  logp_model=torch.log_softmax(logits(model,train_prompts),dim=-1)
  now_probs=logp_model.exp()
  new_behavior=.85*now_probs+.15/2
  ratios=(torch.log(new_behavior.gather(1,actions).clamp_min(1e-8))-prior_logp).exp()
  surrogate=torch.minimum(ratios*adv,ratios.clamp(.8,1.2)*adv).mean()
  kl=(now_probs*(logp_model-ref_probs.log())).sum(1).mean()
  loss=-surrogate+.001*kl
  optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);optim.step()
 after_train=accuracy(train_prompts,train_truth)
 after_val=accuracy(heldout,held_truth)
 result={
   "pretrained_model":model_id,"method":"constrained group-relative policy gradient, clipped ratio and KL",
   "train_prompts":len(train_prompts),"holdout_prompts":len(heldout),
   "group_size":8,"steps":steps,"train_accuracy_before":before_train,"train_accuracy_after":after_train,
   "heldout_accuracy_before":before_val,"heldout_accuracy_after":after_val,
   "limitations":"Two constrained next-token actions, tiny pretrained model, hand-crafted task; not full GRPOTrainer and not LLM reasoning quality."
 }
 f=Path(output);f.parent.mkdir(parents=True,exist_ok=True)
 f.write_text(json.dumps(result,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2))
 return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=120);p.add_argument("--output",default="reports/pretrained_group_reward_v2.json")
 args=p.parse_args();experiment(args.steps,args.output)
