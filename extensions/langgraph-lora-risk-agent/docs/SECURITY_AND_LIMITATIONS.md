# Security & limitations

- Every policy and risk threshold is invented; no bank or client data.
- LangGraph workflow is deterministic, **not an autonomous LLM-agent**.
- Retrieval is word-overlap baseline, **not** semantic vector RAG or GraphRAG.
- Unsupported, negative, nonnumeric, NaN and infinite risk indicators fail closed.
- Risk claims require evidence from the **corresponding** policy ID; otherwise system abstains.
- Retrieved material is never executed as Python or shell commands.
- Benchmark has only 10 synthetic hand-authored questions; metrics do not generalize.
- LoRA smoke test adjusts low-rank PyTorch weights on a frozen basic layer, **not** a pretrained LLM.
- PEFT/SFT script requires a model download and actual training to produce meaningful model metrics; no pretrained fine-tuning results are claimed.
- Next steps: LangGraph integration test in CI, human-evaluated held-out sources, LLM prompt-injection suite, authentication, telemetry, latency and token-cost tracking.
