# RiskOps RAG + AI Agent

## Engineering readiness for LLM internships (October 2026)

**[LLM Engineering Readiness: Python SOLID/DI, FastAPI, Docker, Langfuse SDK v4, TRL GRPO smoke & GPU-ready example scripts](extensions/llm-engineering-readiness/README.md)**

[CI: 8 tests + Docker image build and running container smoke test](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38305842529). **TRL GRPOTrainer** completed a [four-step CPU smoke test](https://github.com/angelina2006mmmm-lab/riskops-rag-ai-agent/actions/runs/38306359894) on tiny GPT-2; further experiments are exploratory, **not evidence of improved reasoning**. vLLM/Unsloth are **GPU scripts not executed**, while Langfuse uses a test client without SaaS trace ingestion. All content is synthetic.


## Latest: LLM Agent Quality & Safety

[Reproducible agent evaluation and real two-tool LangGraph](extensions/llm-research-lab/quality_safety/README.md) — synthetic retrieval/citation benchmarks, prompt-injection suite and safe monitoring. Not production certification.


## New: LLM Research Lab — reproducible synthetic research

**[LLM Research Lab: graph-based retrieval, RNN/Transformer, toy LoRA-SFT, grouped reward RL, algorithms](extensions/llm-research-lab/README.md)**

This is an educational reproducible set of small experiments. Recent extensions contain small *executed* pretrained-model LoRA SFT/DPO and a one-step TRL GRPO CPU smoke test. These do **not** establish improved real-world reasoning, industrial GraphRAG or production LangGraph integration. Evaluation is strictly synthetic.


## Портфолио LLM / AI Agent / Operational Risk

Три независимых учебных прототипа для демонстрации Python, защиты AI-сервисов, проверки источников и тестирования:

| Проект | Суть | Проверки |
|---|---|---|
| [Temporal RAG & Guardrails](extensions/temporal-rag-guardrails) | Версии синтетических политик, проверка источников, базовые guardrails, FastAPI `/metrics` | 19 локальных тестов, 12 учебных сценариев |
| [RiskOps RAG + Agent](.) | TF-IDF retrieval, ссылки на источники, разрешённый tool calling, детерминированные роли | 46 pytest-тестов |
| [LangGraph + LoRA Risk Agent](extensions/langgraph-lora-risk-agent) | Код `StateGraph`, блокировка неподтверждённых выводов, учебная LoRA-демонстрация | 18 локальных тестов; тест с LangGraph не подтверждён |

**Ограничения:** все документы синтетические, а старые offline-агенты не являются автономными LLM. В [LLM Research Lab](extensions/llm-research-lab) подтверждены отдельные LangGraph/LLM-интеграции и PEFT LoRA/SFT на маленьком предобученном TinyStories-1M; результаты основаны на крошечных учебных выборках. Качество больших LLM, полноценный GRPO для reasoning и продакшн-развёртывание не заявляются.


**Учебный Python-проект для стажировки «LLM и AI-агенты».** Система отвечает на вопросы по локальной базе синтетических политик операционных и ESG-рисков, показывает источники, вызывает разрешённые инструменты и демонстрирует простую ролевую мультиагентную схему.

> Все данные DemoCo и пороги рисков вымышлены. Проект не связан с работодателем, не является экспертной системой и не содержит реальных клиентских сведений.

## Возможности

- **RAG retrieval**: разбиение Markdown по заголовкам, индекс TF-IDF, ранжирование релевантных фрагментов, `source`, `score`, `excerpt`. По желанию — семантический поиск через локальную модель Sentence Transformers.
- **AI Agent с tool calling**: при подключённой настоящей LLM агент может вызвать `search_docs` и `assess_indicator`. Сервис выполняет **только эти заранее разрешённые функции**; инструментов исполнения произвольного Python/shell-кода нет.
- **Режим без ключей**: `/ask` с `mode=offline` показывает найденные выдержки и детерминированный расчёт индикатора. **Это не сгенерированный LLM-ответ**, а воспроизводимое offline-demo.
- **Ролевая мультиагентная демонстрация**: `/multi-agent/brief` запускает исследователя (поиск), риск-аналитика (учебная формула) и проверяющего источники. Роли работают детерминированно — не выдаются за три автономные LLM.
- **FastAPI**: OpenAPI/Swagger, валидация запросов Pydantic, локальный Docker Compose.
- **Тестирование**: pytest, фейковый LLM-клиент для теста реального протокола tool calling, CI через GitHub Actions, маленький retrieval benchmark.

## Стек

Python 3.12 (работает и на совместимых новых версиях), FastAPI, Pydantic, scikit-learn, NumPy, httpx, pytest, Docker Compose, Git/GitHub Actions. Опционально: Sentence Transformers/PyTorch для семантических эмбеддингов и совместимый Chat Completions API для генерации.

## Структура

```text
riskops-rag-ai-agent/
├── riskops/
│   ├── knowledge.py        # Markdown → фрагменты → retrieval
│   ├── tools.py            # безопасные инструменты, синтетические пороги
│   ├── llm.py              # внешний Chat Completions API (не обязателен)
│   ├── agent.py            # offline и реальный LLM tool-calling
│   ├── multi_agent.py      # 3 специализированные детерминированные роли
│   └── api.py              # FastAPI
├── knowledge/              # 5 вымышленных политик рисков
├── data/risk_thresholds.json
├── tests/                  # pytest
├── eval/                   # вопросы для поиска
├── scripts/                # demo.py, evaluate.py
├── docs/                   # архитектура, тест-план, интервью
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-dev.txt
├── requirements-embeddings.txt
├── Dockerfile
└── compose.yaml
```

## Быстрый запуск через Ubuntu (WSL)

Распакуйте ZIP в `C:\Users\USER\Desktop\riskops-rag-ai-agent`, откройте Ubuntu и выполните:

```bash
cd /mnt/c/Users/USER/Desktop/riskops-rag-ai-agent
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m scripts.demo
uvicorn riskops.api:app --reload
```

Откройте **http://127.0.0.1:8000/docs**. Для остановки сервера — Ctrl+C. Если `venv` недоступен, установите `sudo apt install python3-venv` и повторите.

Для запуска через Docker (если Docker установлен и доступен из WSL):

```bash
docker compose up --build -d
# остановить:
docker compose down
```

В Docker Compose API доступен только локально: `127.0.0.1:8000`.

## Примеры запросов

Поиск по политикам:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Какие пороги риска простоя?", "top_k":3}'
```

**Offline RAG** с документальными ссылками, без платных сервисов:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Что делать при простое 135 минут?", "mode":"offline", "indicator":"downtime_minutes", "value":135}'
```

Три специализированные роли:

```bash
curl -s -X POST http://127.0.0.1:8000/multi-agent/brief \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как оценивать выбросы CO2e?", "indicator":"emissions_tons", "value":55}'
```

Другие endpoints: `GET /health`, `GET /documents`, `POST /assess`.

## Как подключить **настоящую LLM**

Режим `llm` отправляет вопрос и найденные выдержки во внешний API, поддерживающий **Chat Completions tool calling**. Потребуется свой API-ключ и разрешение на отправку ваших данных провайдеру. В проекте используются только **вымышленные** политики.

1. Скопируйте `.env.example` в `.env` (этот файл в `.gitignore`).
2. Укажите `OPENAI_API_KEY`, `OPENAI_MODEL`, при необходимости `OPENAI_BASE_URL`. API-ключи не размещайте на GitHub.
3. В Ubuntu загрузите переменные **только в свой терминал**:

   ```bash
   set -a
   source .env
   set +a
   uvicorn riskops.api:app --reload
   ```
4. Вызовите `/ask` с `"mode":"llm"`:

   ```bash
   curl -s -X POST http://127.0.0.1:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"question":"Какие действия при высоком риске простоя?", "mode":"llm"}'
   ```

Протокол поддерживает real tool calling, но **выполнение внешнего API на авторской машине не проверялось** — требуется совместимая модель и ключ. Без ключа сервер корректно вернёт HTTP 400, остальные endpoints продолжат работать.

## Опционально — семантические embeddings

По умолчанию используется локальный TF-IDF (без загрузки моделей из интернета). Для embeddings:

```bash
python -m pip install -r requirements-embeddings.txt
export RETRIEVER=embeddings
uvicorn riskops.api:app --reload
```

Модель Sentence Transformers `paraphrase-multilingual-MiniLM-L12-v2` будет скачана с Hugging Face при первом запуске; это **отдельная необязательная возможность**. Тесты и CI используют TF-IDF и не скачивают модель.

## Тесты и качество

```bash
python -m pytest -q
python -m scripts.evaluate --k 3
```

На машине подготовки проекта прошло **46 unit/API-тестов**. Для пяти небольших синтетических контрольных вопросов получено Recall@3 = 1.00. Это проверяет **поиск источника**, а не достоверность свободно сгенерированных ответов или качество на реальных данных. Результаты могут меняться при редактировании базы знаний.

См. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md), [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md).

## Ограничения и дальнейшие идеи

- Демо работает на **5 учебных Markdown-документах**, не на корпоративном хранилище.
- Ответ offline — выдержка из источника, не генерация нейросети. LLM-режим опциональный.
- Ролевой мультиагентный endpoint показывает разделение обязанностей, но не автономную координацию нескольких LLM.
- Для развития проекта: оценка groundedness, инъекция недоверенных документов в тестах, SQL/vector DB, streaming, ручная валидация ответов, observability и трассировка стоимости запросов.
