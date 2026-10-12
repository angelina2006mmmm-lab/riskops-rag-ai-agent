"""Offline LoRA mathematics smoke test (not a pretrained LLM fine-tune)."""
import torch
from torch import nn
def run_smoke(steps=40):
 torch.manual_seed(21)
 base=nn.Linear(8,4,bias=False)
 base.weight.requires_grad=False
 A=nn.Parameter(torch.randn(2,8)*0.01)
 B=nn.Parameter(torch.zeros(4,2))
 x=torch.randn(32,8)
 target=base(x).detach()+.5*x[:,:4]
 def forward(): return base(x)+(x@A.T)@B.T*2
 opt=torch.optim.Adam([A,B],lr=.08)
 before=nn.functional.mse_loss(forward(),target).item()
 for _ in range(steps):
  opt.zero_grad()
  loss=nn.functional.mse_loss(forward(),target)
  loss.backward()
  opt.step()
 after=nn.functional.mse_loss(forward(),target).item()
 assert after<before and base.weight.requires_grad is False
 return before,after
if __name__=="__main__":
 print("LoRA training loss:",run_smoke())
