# Temporal RAG & LLM Guardrails — synthetic operational risk agent

Educational Python/FastAPI PoC for an LLM internship: versioned policy retrieval, evidence-gated decisions, and basic prompt-injection screening. All policies and metrics are synthetic; this is **not production LLM safety**.

## Features
- Temporal document memory selects the policy version effective on the query date (downtime threshold: 90 min in 2025 → 120 min in 2026).
- Evidence gate maps indicator → policy ID/version/effective date; unsupported or mismatched requests abstain.
- Explicit allowlist of risk indicators; deny prompt-like instructions in user input and untrusted retrieved text.
- FastAPI /assess + /health with Pydantic validation, reproducible tests and offline evaluation.
- Locally: 17 pytest tests passed; 12/12 manually constructed synthetic eval cases produced expected status/version. Those are **functional checks, not real-world model performance**.

## Start

```bash
cd extensions/temporal-rag-guardrails
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m temporal_guard.evaluate
uvicorn temporal_guard.api:app --reload
```

Request: POST http://localhost:8000/assess
```json
{"question":"IT downtime 130 minutes","indicator":"downtime_minutes","value":130,"as_of":"2026-10-01"}
```

## Limits
- No actual LLM inference or model fine-tuning.
- Keyword guardrail is incomplete against adaptive prompt injection.
- Not GraphRAG, not an autonomous agent, no real corporate/bank data.
- Next: real-model integration, held-out human-evaluated document sets, prompt-injection attack suite, monitoring/tracing, security review.
