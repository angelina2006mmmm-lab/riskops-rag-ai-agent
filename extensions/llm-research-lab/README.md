# LLM Research Lab — synthetic, reproducible experiments

Student research portfolio aimed at the **LLM Researcher (Risk Modeling)** internship. This code is an educational collection of independent, small experiments, not a production agent and **not Sberbank software**. All policies/data are artificial.

## Implementations

1. **Version-aware graph retrieval** — an entity-policy bipartite graph with multi-hop traversal, explicit citation paths and effective-date version selection. Retrieved evidence returned without LLM generation. [Code](research_lab/graph_rag.py), [policies](data/policies.json), [evaluation](data/eval.jsonl).
2. **LangGraph adapter** — actual `StateGraph` wiring with `retrieve → grounded/abstain`. The local integration test is **skipped** because the optional `langgraph` dependency is unavailable in the authoring environment; install it to execute. [Code](research_lab/langgraph_adapter.py).
3. **RNN versus TransformerEncoder** — 8-step synthetic sequence classification with identical train/validation/test sample sizes. Measured after 7 epochs: RNN test accuracy **0.916**; TransformerEncoder **1.000** (a simple artificial task, not a real NLP benchmark). [Code](research_lab/neural_baselines.py), [report](reports/neural_comparison.json).
4. **Locally pretrained toy causal Transformer + low-rank SFT** — a tiny character-level LM is pretrained **from scratch** on 5 synthetic sentences; main model is frozen and low-rank updates train its output head on 4 tiny supervised examples. Held-out answer-token loss **6.2487 → 0.6648** on **2** synthetic prompts. Not Hugging Face PEFT, not adaptation of a downloaded pretrained LLM, and no meaningful real-world model quality claim. [Code](research_lab/toy_lora_sft.py), [report](reports/toy_lora_sft.json).
5. **Grouped relative reward policy optimization** — toy two-choice numeric classification with grouped rewards, normalized relative advantages, clipped importance ratio and KL reference penalty. Test accuracy **0.504 → 0.9987**; **NOT a complete GRPO trainer** or an improvement to LLM reasoning. [Code](research_lab/group_relative_rl.py), [report](reports/group_relative_rl.json).
6. **Algorithms** — implementations of two pointers, binary search, heap, stack and binary-tree traversal with tests. [Code](research_lab/algorithms.py).

## Run locally

```bash
cd extensions/llm-research-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m research_lab.graph_rag
python -m research_lab.neural_baselines
python -m research_lab.toy_lora_sft
python -m research_lab.group_relative_rl
```

Optional LangGraph integration (requires a separate install):

```bash
python -m pip install langgraph
python -m pytest -q tests/test_graph_and_algorithms.py -k real_langgraph
```

**Local verification:** 20 tests passed; 1 LangGraph integration test skipped. Synthetic graph retrieval: **10/10 hand-authored cases**. Numbers should not be extrapolated to production, risk decisions, actual papers or large language models. No RLHF/GRPO fine-tuning of language models or genuine GraphRAG end-to-end LLM inference was run.

## Research limitations and next milestones

- Entity recognition uses **explicit keywords**, not learned entity extraction or knowledge-graph embeddings.
- Document graph is hand-authored and tiny; no real multi-document/long-context benchmark.
- RNN/Transformer task is easy and does not prove the superiority of an architecture on realistic data.
- Toy SFT is low-rank updating the LM head of a tiny model trained from scratch; no full pretrained-model LoRA/SFT, no real tokenizer such as BPE or SentencePiece.
- Grouped RL demo has only a two-action policy; not actual LLM reasoning improvement.
- Next: verify LangGraph runtime, run PEFT/LoRA on a genuine pretrained open model with a held-out dataset, integrate real generation and build GraphRAG evaluation over larger corpora; compare methods transparently.


## Verified integrations and pretrained-language-model experiments (October 2026)

The following were **executed on GitHub Actions**, not just saved as scripts:

| Component | Execution | What it proves (and doesn't) |
|---|---|---|
| [Version-aware GraphRAG-like agent with LangGraph](research_lab/agentic_graph_rag.py) | [Integration CI passed](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38251970601) | Real StateGraph routing, versioned graph retrieval, citations and abstain. No generative model in this first workflow |
| [Graph-enhanced RAG + actual pretrained generation](research_lab/graph_rag_llm.py) | [Generation CI passed](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38253376624) | A real TinyStories-1M causal LM receives synthetic retrieved text and generates a draft; sources are verified and appended. No manual answer-quality benchmark |
| [PEFT LoRA SFT of pretrained TinyStories-1M](research_lab/pretrained_peft_sft.py) | [Pretrained training CI passed](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38253843710) | 65 steps, 8,192 trainable adapter parameters. Held-out token loss **14.86255 → 2.97834** on *only 2 synthetic prompts*. [Machine-readable metrics](reports/pretrained_peft_sft.json) |
| [Group-relative reward updates on pretrained TinyStories-1M](research_lab/pretrained_group_reward.py) | [Training CI passed](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38253843710) | Constrained **yes/no** next-token policy, grouped advantages, clipping and KL. Holdout accuracy **0.5 → 0.5** (no observed improvement). [Machine-readable metrics](reports/pretrained_group_reward.json) |
| [More diverse group-relative reward v2](research_lab/pretrained_group_reward_v2.py) | [Training CI passed](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38254085807) | 120 steps on six synthetic prompts with exploration, clipped objective and KL; **heldout 0.5 → 1.0 on only 4 yes/no prompts**, not free-form reasoning. [Measured metrics](reports/pretrained_group_reward_v2.json) |

**Precisely what remains outside scope:** enterprise-scale GraphRAG, reliable autonomous multi-step LLM agents, actual RLHF/TRL-GRPO of full free-form reasoning, production evaluation, red teaming, human/financial datasets. The synthetic SFT holdout contains only two prompts: improving its loss does not imply better real-world answers.

### Reproduce the GitHub Actions results

- [Run LangGraph graph-integrated tests](../../.github/workflows/llm-real-llm-checks.yml)
- [Run retrieval + real generation CI](../../.github/workflows/llm-graph-generation.yml)
- [Run PEFT LoRA SFT and group-relative rewards](../../.github/workflows/llm-pretrained-research.yml)

