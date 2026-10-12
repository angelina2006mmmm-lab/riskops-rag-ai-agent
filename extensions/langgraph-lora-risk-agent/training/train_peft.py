"""Optional PEFT LoRA SFT: requires downloads and compute; unverified pretrained run."""
import argparse,json
from pathlib import Path
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--model",default="HuggingFaceTB/SmolLM2-135M-Instruct")
 parser.add_argument("--steps",type=int,default=12)
 parser.add_argument("--output",default="artifacts/risk-lora")
 args=parser.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import LoraConfig,TaskType,get_peft_model
 tok=AutoTokenizer.from_pretrained(args.model)
 if tok.pad_token is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model)
 model.config.use_cache=False
 model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,target_modules=["q_proj","v_proj"]))
 model.train()
 device="cuda" if torch.cuda.is_available() else "cpu"
 model.to(device)
 rows=[json.loads(s) for s in Path("examples/synthetic_sft.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
 for i in range(args.steps):
  row=rows[i%len(rows)]
  prompt="Вопрос по учебным рискам: "+row["instruction"]+"\nОтвет: "
  full=prompt+row["response"]+tok.eos_token
  ids=tok(full,add_special_tokens=False,truncation=True,max_length=256)["input_ids"]
  prompt_len=len(tok(prompt,add_special_tokens=False)["input_ids"])
  if prompt_len>=len(ids):raise ValueError("Target truncated")
  x=torch.tensor([ids],device=device)
  labels=torch.tensor([[-100]*prompt_len+ids[prompt_len:]],device=device)
  loss=model(input_ids=x,labels=labels).loss
  opt.zero_grad();loss.backward();opt.step()
  print("step",i+1,"loss",float(loss))
 Path(args.output).mkdir(parents=True,exist_ok=True)
 model.save_pretrained(args.output)
 tok.save_pretrained(args.output)
if __name__=="__main__":main()
