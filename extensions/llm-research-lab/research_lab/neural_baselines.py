"""RNN vs TransformerEncoder on synthetic temporal classification; no pretraining."""
import json
from pathlib import Path
import torch
from torch import nn
def dataset(n,seed):
 gen=torch.Generator().manual_seed(seed)
 x=torch.randint(0,5,(n,8),generator=gen)
 y=((x[:,0]==1)&(x[:,-1]>=3)).long()
 return x,y
class RNNClassifier(nn.Module):
 def __init__(self):
  super().__init__()
  self.embed=nn.Embedding(5,12);self.rnn=nn.RNN(12,20,batch_first=True);self.head=nn.Linear(20,2)
 def forward(self,x):return self.head(self.rnn(self.embed(x))[0][:,-1])
class TransformerClassifier(nn.Module):
 def __init__(self):
  super().__init__()
  self.embed=nn.Embedding(5,20);self.pos=nn.Embedding(8,20)
  layer=nn.TransformerEncoderLayer(d_model=20,nhead=4,dim_feedforward=32,dropout=0,batch_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False);self.head=nn.Linear(20,2)
 def forward(self,x):
  p=self.pos(torch.arange(x.shape[1],device=x.device))[None,:]
  return self.head(self.encoder(self.embed(x)+p).mean(1))
def train_and_evaluate(epochs=7,output=None):
 torch.set_num_threads(1)
 tx,ty=dataset(1500,42);vx,vy=dataset(400,43);ex,ey=dataset(500,44)
 result={"description":"RNN vs Transformer on synthetic 8-step token sequences; no pretraining","train":len(tx),"validation":len(vx),"test":len(ex),"epochs":epochs}
 for name,cls in [("RNN",RNNClassifier),("TransformerEncoder",TransformerClassifier)]:
  torch.manual_seed(42);model=cls();opt=torch.optim.AdamW(model.parameters(),lr=.009)
  for _ in range(epochs):
   model.train()
   for ix in torch.randperm(len(tx)).split(100):
    loss=nn.functional.cross_entropy(model(tx[ix]),ty[ix])
    opt.zero_grad();loss.backward();opt.step()
  model.eval()
  with torch.no_grad():
   va=float((model(vx).argmax(1)==vy).float().mean())
   ea=float((model(ex).argmax(1)==ey).float().mean())
  result[name]={"validation_accuracy":round(va,4),"test_accuracy":round(ea,4),"parameters":sum(p.numel() for p in model.parameters())}
 if output:Path(output).write_text(json.dumps(result,indent=2),encoding="utf-8")
 return result
if __name__=="__main__":
 print(json.dumps(train_and_evaluate(output=Path(__file__).resolve().parents[1]/"reports"/"neural_comparison.json"),indent=2))
