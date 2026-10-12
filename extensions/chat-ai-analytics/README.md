# Chat AI Analytics Lab — offline prompt evaluation, RAG-adjacent NLP analytics and SQL

**Educational portfolio prototype for an AI Analyst internship in a banking-chat product.** All 100 Russian messages are manually written synthetic fixtures. This is not connected to Sberbank/SBOL, real customers or production traffic.

## Implemented

- Five labelled intent classes: balance, card_block, transfer_status, loan_payment and other.
- Two written prompt instructions and **two deterministic rule-based emulators** of prompt policy A/B. They are NOT actual outputs of two LLMs. A typed ChatCompletion interface plus mocked JSON-response tests illustrate how a real LLM would be integrated.
- Offline accuracy, per-class precision/recall, macro F1 and a five-way confusion matrix.
- Amount extraction by regex, abstention for possible long numeric identifiers, high-priority handling of card-loss wording.
- Reproducible SHA-256 variant assignment for illustrative experiment plumbing, not real user randomization.
- SQLite monitoring with CTEs, correlated subqueries, REGEXP, cumulative windows, simulated latency, mismatches and escalation rate.
- 17 automated unit/contract/SQL tests and GitHub Actions that regenerate JSON reports.

## Results — constrained to hand-authored synthetic data

Offline deterministic rule emulators: **macro F1 A = 0.3245, B = 0.9167**. These cases intentionally contain synonyms handled by B, so they are not independent evidence of improvement in any LLM or real banking chat service.

[Successful CI workflow](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/workflows/chat-ai-analytics.yml)

## Reproduce

```bash
cd extensions/chat-ai-analytics
python -m pip install -r requirements.txt
python -m pytest -q tests/
python -m src.experiment
python -m src.analytics
```

## Responsible online A/B proposal (not yet conducted)

- **Hypothesis:** improved intent classification without worsening unsafe routing or latency.
- **Randomization unit:** anonymous chat session ID with persistent assignment (never individual messages of a single thread).
- **Primary metric:** human-labelled intent routing accuracy; secondary macro F1.
- **Guardrails:** unsafe tool execution, escalations, abstentions, hallucinations, PII leakage and P95 latency.
- **Prelaunch checks:** power/sample-size plan, stable prompts/model version, no peeking, latency limits and rollback criteria.
- **Analysis:** intention-to-treat, uncertainty intervals and segment errors.

## Scope

No live LLM inference in the A/B comparison, no actual online randomized test, no causal treatment effect or statistical significance claim, no production monitoring or customer data. For a separately tested RAG PoC, see [RiskOps](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent).
