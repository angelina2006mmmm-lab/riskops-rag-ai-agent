# LangGraph + LoRA Risk Agent — учебный pet-project

Отдельный учебный модуль в репозитории RiskOps для стажировки Data Scientist (Риски). Все политики и примеры данных **синтетические**.

**LangGraph:** StateGraph с узлами retrieve -> evaluate -> evidence gate -> answer / abstain. Лексический поиск по синтетическим политикам, источники, безопасная оценка через allowlist. Не является автономной LLM-системой.

**LoRA:** `training/lora_smoke.py` демонстрирует настоящее обновление низкоранговых матриц A/B с замороженными весами в PyTorch. `training/train_peft.py` — скрипт PEFT LoRA/SFT для HuggingFaceTB/SmolLM2-135M-Instruct на 4 синтетических примерах; код предоставлен, **полноценное дообучение предобученной LLM пока не подтверждено**.

## Запуск из корня репозитория

```bash
cd extensions/langgraph-lora-risk-agent
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m risk_agent.graph
```

Для LoRA:

```bash
python -m pip install -r requirements-lora.txt
python -m training.lora_smoke
python -m training.train_peft --steps 12 --output artifacts/risk-lora
```

При первом запуске PEFT скачивает предобученную модель, поэтому потребуются интернет, память и время. Последняя команда не проверена в текущей среде.

**Ограничения:** retrieval основан на пересечении слов, а не векторах; граф детерминированный; никаких реальных банковских сведений; датасет не годится для оценки качества fine-tuning.