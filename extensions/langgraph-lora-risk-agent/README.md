# LangGraph + LoRA Risk Agent

Учебный pet-project для стажировки Data Scientist (Риски). Код работает **только на синтетических политиках** и не использует данные Сбера. Проект не является системой принятия реальных риск-решений.

## Что действительно реализовано

- **LangGraph**: `StateGraph` с узлами `retrieve → evaluate → evidence gate → answer/abstain`. Интеграционный тест рассчитан на установленный `langgraph`.
- **Evidence-first**: поиск по фрагментам 5 синтетических документов; источники с ID; численные оценки разрешены только если найден конкретный документ, относящийся к индикатору; иначе система воздерживается от ответа.
- **Guardrails**: ограниченный список показателей, проверка числовых значений (NaN/inf, отрицательные, неверные типы), отказ при несовпадении источника и показателя, отсутствие произвольного выполнения кода.
- **Тесты и оценка**: 18 тестов пройдены локально, 1 тест запуска LangGraph пропущен без внешней зависимости. На 10 **искусственно составленных** случаях получены status accuracy 10/10 и source recall 5/5. Эти метрики **не показывают качество на реальных документах и не отражают качество LLM-генерации**.
- **LoRA/PyTorch**: проверенный smoke-тест обучаемых низкоранговых матриц при замороженном базовом слое; loss в локальном эксперименте 0.228 → 0.091. **Это не fine-tuning предобученной LLM**.
- **PEFT/SFT**: скрипт для адаптации `HuggingFaceTB/SmolLM2-135M-Instruct` на синтетических Q&A. Выполнение этого скрипта с настоящими весами пока **не подтверждено**: требуются загрузка модели и ресурсы.

## Запуск в Ubuntu / WSL

Из корня репозитория:

```bash
cd extensions/langgraph-lora-risk-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_agent.py tests/test_eval.py
python -m scripts.evaluate
python -m risk_agent.graph
```

Отдельно для LoRA:

```bash
python -m pip install -r requirements-lora.txt
python -m training.lora_smoke
python -m training.train_peft --steps 12 --output artifacts/risk-lora
```

Последняя команда **скачивает веса Hugging Face и требует RAM/GPU**; для осмысленных экспериментов необходимы полноценный train/validation dataset, holdout-eval и воспроизводимые результаты.

## Структура

- `risk_agent/graph.py` — граф и независимая проверка той же бизнес-логики
- `tests/` — проверки безопасных и ошибочных сценариев
- `eval/questions.jsonl` — 10 учебных запросов
- `scripts/evaluate.py` — расчёт учебных метрик
- `training/lora_smoke.py` — проверка математики LoRA
- `training/train_peft.py` — опциональный скрипт SFT с PEFT
- `docs/SECURITY_AND_LIMITATIONS.md` — ограничения

Для графа настроен workflow GitHub Actions `langgraph-lora-ci.yml`. Наличие workflow не означает, что он уже успешно отработал: статус см. в Actions.

**Никаких заявлений об автономной работе LLM, GraphRAG, production или о завершённом fine-tuning предобученной модели.**
