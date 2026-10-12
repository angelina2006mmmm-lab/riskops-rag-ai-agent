# LLM Agent Quality & Safety

Reproducible evaluation of a small educational risk agent on hand-authored synthetic documents. No real bank data, certified LLM safety result or production deployment.

Implemented:
- Retrieval evaluation: Recall@3, Precision@3, MRR@3 and abstention accuracy on 10 labelled synthetic cases.
- Temporal citation verification: only evidence with currently applicable policy ID/version counts as verified.
- Prompt-injection screening: 20 labelled attack/benign cases, detection and benign false-positive rates. Regex rules are easy to bypass and NOT strong prompt-injection defense.
- Real LangGraph StateGraph with TWO allowlisted tools: policy_lookup and risk_indicator; versions, citation checking and fail-closed abstention.
- Monitoring: per-outcome counters, latency and privacy-safe audit event without raw prompts.
- Maintainable code: separate evaluation, orchestration and metrics classes; pytest + GitHub Actions.

Run from extensions/llm-research-lab:
  python -m pip install 'pytest>=8,<10' 'langgraph>=0.6,<2'
  python -m pytest -q quality_safety/tests/
  python -m quality_safety.quality
  python -m quality_safety.multi_tool_graph

Limitations: Checks of citation IDs are not semantic/factual support tests. Data and thresholds are synthetic. No production LLM reasoning guarantee, real-world jailbreak testing, service observability SLA, authenticated Langfuse deployment, GPU vLLM or Unsloth benchmark.

See parent LLM Research Lab for verified pretrained TinyStories-1M PEFT LoRA/SFT and limited grouped-reward experiments.
