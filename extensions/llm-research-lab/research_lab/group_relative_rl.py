"""Grouped relative policy gradient (toy numeric classification); NOT LLM GRPO."""
import json
from pathlib import Path
import torch
from torch import nn
def advantages(rewards):
 return (rewards-rewards.mean(1,keepdim=True))/rewards.std(1,keepdim=True,unbiased=False).clamp_min(.2)
def task_batch(n,gen):
 d=torch.randn(n,1,generator=gen)
 return torch.cat([d,torch.ones_like(d)],1),(d[:,0]>0).long()
def experiment(steps=120,group_size=5,output=None):
 torch.set_num_threads(1);torch.manual_seed(17);rng=torch.Generator().manual_seed(101)
 model=nn.Linear(2,2)
 with torch.no_grad():model.weight.zero_();model.bias.zero_()
 reference=[p.detach().clone() for p in model.parameters()]
 opt=torch.optim.Adam(model.parameters(),lr=.055)
 x_eval,y_eval=task_batch(1500,torch.Generator().manual_seed(200))
 def accuracy():
  with torch.no_grad():return float((model(x_eval).argmax(1)==y_eval).float().mean())
 before=accuracy()
 for _ in range(steps):
  x,y=task_batch(96,rng)
  with torch.no_grad():
   old=torch.log_softmax(model(x),dim=-1)
   actions=torch.multinomial(old.exp(),group_size,replacement=True,generator=rng)
   old_chosen=old.gather(1,actions)
   reward=(actions==y[:,None]).float()
   adv=advantages(reward)
  now=torch.log_softmax(model(x),dim=-1).gather(1,actions)
  ratio=torch.exp(now-old_chosen)
  old_ref=torch.log_softmax(nn.functional.linear(x,reference[0],reference[1]),dim=-1)
  curr=torch.softmax(model(x),dim=-1)
  kl=(curr*(torch.log_softmax(model(x),dim=-1)-old_ref)).sum(-1).mean()
  loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.002*kl
  opt.zero_grad();loss.backward();opt.step()
 result={"method":"group-relative advantage, clipped surrogate, KL reference (toy)","before_accuracy":round(before,4),"after_accuracy":round(accuracy(),4),"steps":steps,"group_size":group_size,"limitations":"Numeric classification, no language-model weights, not full GRPO for LLM reasoning"}
 if output:Path(output).write_text(json.dumps(result,indent=2),encoding="utf-8")
 return result
if __name__=="__main__":
 print(json.dumps(experiment(output=Path(__file__).resolve().parents[1]/"reports"/"group_relative_rl.json"),indent=2))
