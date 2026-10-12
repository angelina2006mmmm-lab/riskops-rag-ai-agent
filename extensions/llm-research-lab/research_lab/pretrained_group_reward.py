"""A constrained-action GRPO-style experiment on actual pretrained tiny GPT-2.

Group-normalized rewards, PPO-style clipped importance ratio, reference KL.
Two possible next tokens (' yes'/' no'), not free-form reasoning or full TRL GRPO.
"""
import argparse,json
from pathlib import Path

MODEL_ID="sshleifer/tiny-gpt2"
def experiment(steps=50,output="reports/pretrained_group_reward.json"):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from peft import LoraConfig,TaskType,get_peft_model
 torch.manual_seed(7);torch.set_num_threads(2)
 tok=AutoTokenizer.from_pretrained(MODEL_ID)
 tok.pad_token=tok.eos_token
 tok.padding_side="left"
 candidates=[tok.encode(" yes",add_special_tokens=False),tok.encode(" no",add_special_tokens=False)]
 assert all(len(x)==1 for x in candidates),"Action tokens must be single GPT-2 tokens"
 tokens=[x[0] for x in candidates]
 train_prompts=["Risk: high operational risk. Should review be escalated? Answer:",
                "Risk: low operational risk. Should review be escalated? Answer:"]
 heldout_prompts=["There is high risk. Escalate the incident? Answer:",
                  "There is low risk. Escalate the incident? Answer:"]
 wanted=torch.tensor([0,1])
 base=AutoModelForCausalLM.from_pretrained(MODEL_ID)
 def tokenize(prompts):
  return tok(prompts,return_tensors="pt",padding=True)
 def select_logits(model,prompts):
  d=tokenize(prompts)
  return model(input_ids=d["input_ids"],attention_mask=d["attention_mask"]).logits[:,-1,tokens]
 with torch.no_grad():
  reference_train=torch.log_softmax(select_logits(base,train_prompts),dim=-1)
 peft=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,
      r=4,lora_alpha=8,lora_dropout=0.0,target_modules=["c_attn"],
      fan_in_fan_out=True,bias="none"))
 optimizer=torch.optim.AdamW((x for x in peft.parameters() if x.requires_grad),lr=.006)
 def accuracy():
  peft.eval()
  with torch.no_grad():
   vals=select_logits(peft,heldout_prompts).argmax(dim=1)
   return float((vals==wanted).float().mean().item())
 before=accuracy()
 peft.train()
 for _ in range(steps):
  with torch.no_grad():
   old=torch.log_softmax(select_logits(peft,train_prompts),dim=-1)
   actions=torch.multinomial(old.exp(),6,replacement=True)
   old_actions=old.gather(1,actions)
   rewards=(actions==wanted[:,None]).float()
   adv=(rewards-rewards.mean(1,keepdim=True))/rewards.std(1,keepdim=True,unbiased=False).clamp_min(.2)
  logits=select_logits(peft,train_prompts)
  logp=torch.log_softmax(logits,dim=-1)
  ratio=(logp.gather(1,actions)-old_actions).exp()
  objective=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
  probs=logp.exp()
  ref_kl=(probs*(logp-reference_train)).sum(1).mean()
  loss=-objective+.002*ref_kl
  optimizer.zero_grad();loss.backward();optimizer.step()
 after=accuracy()
 result={"model":MODEL_ID,"method":"grouped relative policy update on pretrained LM logits",
   "action_space":"two constrained next-token responses: yes/no",
   "training_prompts":2,"heldout_prompts":2,"group_size":6,"steps":steps,
   "heldout_accuracy_before":before,"heldout_accuracy_after":after,
   "limitations":"Not TRL GRPOTrainer, no open-ended generation, no evidence of improved LLM reasoning."}
 p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2));return result
if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("--steps",type=int,default=50)
 parser.add_argument("--output",default="reports/pretrained_group_reward.json")
 args=parser.parse_args()
 experiment(args.steps,args.output)
