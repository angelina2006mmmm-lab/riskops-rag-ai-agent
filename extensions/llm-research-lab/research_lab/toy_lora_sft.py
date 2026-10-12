"""Tiny locally pretrained causal Transformer + low-rank output-head SFT; not a real pretrained LLM."""
import json
from pathlib import Path
import torch
from torch import nn
ALPHABET="abcdefghijklmnopqrstuvwxyz0123456789 .:?=\n~"
TO_ID={ch:i for i,ch in enumerate(ALPHABET)}
class ToyCausalTransformer(nn.Module):
 def __init__(self,rank=4):
  super().__init__();d=32
  self.embed=nn.Embedding(len(ALPHABET),d);self.pos=nn.Embedding(90,d)
  layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=64,dropout=0,batch_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
  self.head=nn.Linear(d,len(ALPHABET))
  self.lora_A=nn.Parameter(torch.randn(rank,d)*.01)
  self.lora_B=nn.Parameter(torch.zeros(len(ALPHABET),rank))
  self.rank=rank
 def forward(self,x,use_adapter=True):
  n=x.shape[1];h=self.embed(x)+self.pos(torch.arange(n))[None,:,:]
  mask=torch.triu(torch.ones(n,n,dtype=torch.bool),diagonal=1)
  h=self.encoder(h,mask=mask);logits=self.head(h)
  if use_adapter:logits=logits+((h@self.lora_A.T)@self.lora_B.T)/self.rank*8
  return logits
def encode(text):
 if not set(text).issubset(TO_ID):raise ValueError("Unknown alphabet symbol")
 return torch.tensor([[TO_ID[c] for c in text]],dtype=torch.long)
def lm_loss(model,line,answer_only=False,adapter=True):
 x=encode(line)
 if x.shape[1]>90:raise ValueError("Example too long")
 preds=model(x[:,:-1],use_adapter=adapter)
 y=x[:,1:].clone()
 if answer_only:
  end=line.index("answer:")+len("answer:")
  y[:,:max(0,end-1)]=-100
 return nn.functional.cross_entropy(preds.reshape(-1,len(ALPHABET)),y.reshape(-1),ignore_index=-100)
def experiment(pretrain_steps=65,sft_steps=100,output=None):
 torch.set_num_threads(1);torch.manual_seed(9)
 m=ToyCausalTransformer()
 base=["the bank records the risk event.~","the analyst checks policy notes.~","a dated event needs a source.~","risk monitoring compares periods.~","credit models should be tested.~"]
 opt=torch.optim.AdamW(m.parameters(),lr=.008)
 for i in range(pretrain_steps):
  loss=lm_loss(m,base[i%len(base)],adapter=False)
  opt.zero_grad();loss.backward();opt.step()
 for name,param in m.named_parameters():param.requires_grad=name in {"lora_A","lora_B"}
 assert not m.head.weight.requires_grad
 train=["question: is the audit complete?\nanswer: review~","question: is the policy valid?\nanswer: review~","question: is the source current?\nanswer: review~","question: is the incident closed?\nanswer: review~"]
 heldout=["question: is the report ready?\nanswer: review~","question: is the risk low?\nanswer: review~"]
 with torch.no_grad():before=float(sum(lm_loss(m,s,answer_only=True) for s in heldout)/len(heldout))
 opt=torch.optim.AdamW([m.lora_A,m.lora_B],lr=.065)
 for i in range(sft_steps):
  loss=lm_loss(m,train[i%len(train)],answer_only=True)
  opt.zero_grad();loss.backward();opt.step()
 m.eval()
 with torch.no_grad():after=float(sum(lm_loss(m,s,answer_only=True) for s in heldout)/len(heldout))
 result={"pretraining":"tiny causal Transformer, trained locally from scratch on synthetic text","adapter":"manual low-rank update to final output head; frozen backbone","sft_examples":len(train),"heldout_prompts":len(heldout),"heldout_token_loss_before":round(before,4),"heldout_token_loss_after":round(after,4),"pretrain_steps":pretrain_steps,"sft_steps":sft_steps,"limitations":"Not downloaded pretrained LLM, not PEFT/QLoRA, tiny synthetic dataset"}
 if output:Path(output).write_text(json.dumps(result,indent=2),encoding="utf-8")
 return result
if __name__=="__main__":
 print(json.dumps(experiment(output=Path(__file__).resolve().parents[1]/"reports"/"toy_lora_sft.json"),indent=2))
