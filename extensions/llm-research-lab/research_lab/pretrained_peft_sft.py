"""Verified PEFT LoRA-SFT on an actual pretrained (tiny) GPT-2 checkpoint.

Requires Hugging Face model download; not enterprise LLM fine-tuning.
Output is a small reproducible research demonstration on synthetic questions.
"""
import argparse, json
from pathlib import Path

MODEL_ID="roneneldan/TinyStories-1M"
TRAIN=[
 ("Policy: supplier outage over threshold. Action:", " escalate"),
 ("Policy: unauthorized system access. Action:", " escalate"),
 ("Policy: control has no evidence. Action:", " abstain"),
 ("Policy: missing incident log. Action:", " abstain"),
]
HOLDOUT=[
 ("Policy: vendor outage beyond SLA. Action:", " escalate"),
 ("Policy: insufficient policy sources. Action:", " abstain"),
]
def examples(tok,rows):
 import torch
 for prompt,target in rows:
  a=tok(prompt,add_special_tokens=False)["input_ids"]
  b=tok(target+tok.eos_token,add_special_tokens=False)["input_ids"]
  ids=a+b
  if not b:raise ValueError("No target tokens")
  yield torch.tensor([ids]),torch.tensor([[-100]*len(a)+b])

def experiment(steps=65,output="reports/pretrained_peft_sft.json",adapter_dir="artifacts/lora_adapter"):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import LoraConfig,TaskType,get_peft_model
 torch.manual_seed(11);torch.set_num_threads(2)
 tok=AutoTokenizer.from_pretrained(MODEL_ID)
 tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(MODEL_ID)
 cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,lora_alpha=8,lora_dropout=0.0,
                target_modules=["q_proj","v_proj"],fan_in_fan_out=False,bias="none")
 model=get_peft_model(base,cfg)
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 total=sum(p.numel() for p in model.parameters())
 assert 0<trainable<total
 data=list(examples(tok,TRAIN))
 holdout=list(examples(tok,HOLDOUT))
 def eval_loss():
  model.eval()
  with torch.no_grad():
   return round(float(sum(model(input_ids=x,labels=y).loss.item() for x,y in holdout)/len(holdout)),5)
 before=eval_loss()
 optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=0.007)
 train_losses=[]
 model.train()
 for i in range(steps):
  x,y=data[i%len(data)]
  l=model(input_ids=x,labels=y).loss
  optimizer.zero_grad();l.backward();optimizer.step()
  train_losses.append(float(l.detach()))
 after=eval_loss()
 Path(adapter_dir).mkdir(parents=True,exist_ok=True)
 model.save_pretrained(adapter_dir)
 result={
  "base_model":MODEL_ID,"checkpoint_type":"public pretrained tiny GPT-Neo / TinyStories weights",
  "method":"Hugging Face PEFT LoRA supervised fine-tuning","train_examples":len(TRAIN),
  "heldout_examples":len(HOLDOUT),"steps":steps,"trainable_parameters":trainable,
  "total_parameters":total,"heldout_loss_before":before,"heldout_loss_after":after,
  "last_train_loss":round(train_losses[-1],5),
  "limitations":"Tiny pretrained checkpoint + synthetic prompts; no production LLM performance claim."
 }
 p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2));return result

if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("--steps",type=int,default=65)
 parser.add_argument("--output",default="reports/pretrained_peft_sft.json")
 parser.add_argument("--adapter-dir",default="artifacts/lora_adapter")
 args=parser.parse_args()
 experiment(args.steps,args.output,args.adapter_dir)
