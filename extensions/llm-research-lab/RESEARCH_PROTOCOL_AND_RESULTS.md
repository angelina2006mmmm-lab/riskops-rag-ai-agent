# LLM Researcher — reproducible experiment notes

**Purpose:** a research-style index of experiments that were implemented and verified in this repository. Built for a student's LLM researcher portfolio. This is **not** a formal scientific publication, bank deployment, or an evaluation of large production language models. All datasets are synthetic.

## Research questions and falsifiable tests

| Question | Experiment and baseline | Reported observation | Limits / interpretation |
|---|---|---|---|
| Can recurrent and attention-based architectures learn a simple sequence task? | RNN versus 1-layer TransformerEncoder; same train/validation/test lengths. [Source](research_lab/neural_baselines.py) · [numbers](reports/neural_comparison.json). | Test accuracy 0.916 / 1.000; train=1500, validation=400, test=500, 7 epochs. | Hand-specified simple classification; different parameter counts and no multiple random seeds. Do **not** claim attention is generally superior. |
| Does low-rank adaptation update a real pretrained model's behaviour? | Frozen pretrained TinyStories-1M weights with PEFT LoRA/SFT. [Source](research_lab/pretrained_peft_sft.py) · [metrics](reports/pretrained_peft_sft.json). | Synthetic held-out token loss 14.86255 → 2.97834 over **two** examples after 65 steps; 8,192 adapter parameters. | Two examples, no credible generalization / statistical certainty / external validation; not fine-tuning a large bank LLM. |
| Does reference-based DPO training execute end-to-end? | PEFT LoRA policy + frozen pretrained reference; four synthetic preference pairs. [Source](research_lab/pretrained_preference_dpo.py) · [successful CI](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38304700640). | 32 training steps completed; two held-out pairs only. | A running objective is not proof of better language-model safety or broad preference alignment. |
| Can an actual TRL GRPOTrainer optimize a pretrained checkpoint? | TRL GRPOTrainer, tiny GPT-2, 4 synthetic prompts, 2 generations. [Source](../llm-engineering-readiness/grpo/train.py) · [successful CI](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38306359894). | 4 optimizer steps completed. | Smoke test only; no controlled reasoning benchmark, no claim of improved reasoning. |
| Can graph-based evidence selection and basic guardrails be measured? | Versioned graph retrieval, two-tool LangGraph and manually labeled prompt-injection tests. [Evaluation suite](quality_safety/README.md) · [CI](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38304554475). | 10 retrieval cases, 20 attack/benign cases, 11 tests. | Tiny hand-authored cases; ID-based citation validation is not semantic factuality or robust adversarial defence. |

## Research quality checklist

1. **Data provenance and leakage:** use documented synthetic data; distinguish temporal policy validity from actual point-in-time finance evaluation. Never use an answer in the prompt as a hidden test label.
2. **Baselines first:** compare against a simple baseline with the same split, cost and metrics; document parameter count and computation budget.
3. **Controlled validation:** keep training, validation, and held-out test separate, track random seeds, and repeat experiments on several seeds before making confident comparisons.
4. **ML metrics:** record loss/accuracy but also calibration, false positives, confidence intervals and real error examples before drawing conclusions.
5. **Agent quality:** split retrieval Recall@k, citation correctness, abstention, tool safety, factuality, operational latency and business risk; these are separate measures.
6. **Safety and failure handling:** benchmark adaptive prompt injection, tool authorization, temporal-document conflicts, hallucination, privacy and degraded inference.
7. **Production criteria:** define P95 latency, availability, tracing privacy, rollback, model versions, data contracts, monitoring drift and incident ownership.

## What these experiments do NOT support

- No statistical claim that Transformer outperforms RNN in natural-language tasks.
- No evidence that the GRPO smoke test improved open-ended LLM reasoning.
- No external dataset/real financial-domain generalization for the 2-example SFT evaluation.
- No production-grade GraphRAG, multi-agent autonomous business workflow, or verified hosted Langfuse integration.
- No executed vLLM or Unsloth GPU job. Optional scripts are environment checks only.

## Next best reproducible research milestone

Create a sufficiently large document QA benchmark with train/dev/test and temporal shifts, compare lexical retrieval vs embedding retrieval vs graph retrieval under the same LLM, evaluate answer faithfulness with human labels, ablate tool access, and train LoRA/DPO/GRPO on one common held-out task with fixed seeds. Report cost and confidence intervals. Until that work is run, keep all conclusions scoped to the synthetic demos.
