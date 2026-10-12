import pytest
import torch
from research_lab.neural_baselines import dataset,RNNClassifier,TransformerClassifier,train_and_evaluate
from research_lab.group_relative_rl import advantages,experiment as run_rl
from research_lab.toy_lora_sft import ToyCausalTransformer,encode,experiment as run_lora
def test_neural_shapes():
 x,y=dataset(18,19)
 assert x.shape==(18,8) and y.shape==(18,)
 assert RNNClassifier()(x).shape==(18,2)
 assert TransformerClassifier()(x).shape==(18,2)
def test_neural_training(tmp_path):
 r=train_and_evaluate(epochs=5,output=tmp_path/"nn.json")
 assert r["test"]==500 and r["RNN"]["test_accuracy"]>.5
def test_group_advantage():
 x=torch.tensor([[0.,1.,1.,0.],[1.,1.,1.,1.]])
 a=advantages(x)
 assert torch.allclose(a[0].mean(),torch.tensor(0.),atol=1e-5)
 assert torch.allclose(a[1],torch.zeros_like(a[1]))
def test_group_policy_learning(tmp_path):
 r=run_rl(steps=65,output=tmp_path/"rl.json")
 assert r["after_accuracy"]>r["before_accuracy"]+.3
def test_toy_transformer_shapes():
 x=encode("risk.~")
 assert ToyCausalTransformer()(x).shape==(1,len("risk.~"),len("abcdefghijklmnopqrstuvwxyz0123456789 .:?=\n~"))
def test_toy_lora_sft(tmp_path):
 r=run_lora(pretrain_steps=24,sft_steps=55,output=tmp_path/"toy.json")
 assert r["heldout_token_loss_after"]<r["heldout_token_loss_before"]
