# LLM engineering readiness — tested service and optional GPU demos

This is a **deterministic risk-policy review microservice**, not a generative LLM or an autonomous AI agent. Its purpose is to prove engineering skills useful for agent integration, not model quality.

Verified in CI: Python dependency injection with typed Retriever/Tracer interfaces (SOLID), FastAPI & Pydantic validation, temporal policy versions, abstention, metrics, Langfuse SDK v4 adapter tested against an injected fake, and Docker build/startup/health request.

A configured real Langfuse server and trace ingestion are NOT verified: this would require credentials and network access. No actual user question or sensitive inputs are written to the trace.

## Local run
```sh
pip install -r requirements.txt
pytest -q tests
uvicorn agent.api:app --reload
docker build -t llm-engineering-demo .
docker run --rm -p 127.0.0.1:8000:8000 llm-engineering-demo
```

Use POST /v1/review, GET /health, GET /metrics. All policy thresholds and evidence are synthetic.

## Optional GPU checks, NOT performed in CPU CI
```sh
python -m gpu.preflight
python -m gpu.vllm_demo
python -m gpu.unsloth_demo
```
Install vLLM or Unsloth in a compatible CUDA environment first. A script or import does **not** prove inference, successful fine-tuning, production throughput or usable GPU experience.

## Verified GRPO smoke (not model-quality validation)

[TRL GRPOTrainer — 4 actual CPU optimizer steps](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38306359894) using pretrained `sshleifer/tiny-gpt2`, four synthetic prompts and two generations per prompt. Training loss recorded in [metrics JSON](reports/grpo_trl_smoke.json). This does not establish an improvement in reasoning or readiness for business decisions.
