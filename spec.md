# ТЕХНИЧЕСКАЯ СПЕЦИФИКАЦИЯ (SPEC.md)

## Продукт: CLI-игра «Переговоры со стартап-инвестором» (Jev + LLM Orchestration)

---

### 1. Концепция и Архитектура Системы

Проект реализует двухконтурный искусственный интеллект для NPC-инвестора в текстовой пошаговой игре:

1. **Детерминированный мозг (Evaluation Engine):** модель класса **Open-Jev-9B** (или совместимый Jev-рантайм), которая принимает формализованное состояние игры (Game State) и вычисляет дискретные вероятностные оценки через заданные критерии (`score`, `choice`, `noul`).
2. **Речевой аппарат (Dialogue Generation):** текстовая LLM (локальная через Ollama/vLLM или облачный API), которая **не принимает решений самостоятельно**, а исключительно переводит сырые численные решения Jev и контекст сделки в художественную реплику персонажа.



```text
               ┌────────────────────────┐
               │    GAME STATE STORE    │
               │ (Баланс, оценка, стек) │
               └───────────┬────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │     JEV EVALUATOR      │
               │  (ZefanCai/Open-Jev)   │
               └───────────┬────────────┘
                           │ evaluation_results (accept, reaction, good_deal)
                           ▼
               ┌────────────────────────┐
               │   DECISION ARBITER     │
               │ (Игровая бизнес-логика)│
               └───────────┬────────────┘
                           │ prompt payload + decision directives
                           ▼
               ┌────────────────────────┐
               │      VOICE LLM         │
               │  (Генерация диалога)   │
               └───────────┬────────────┘
                           │ formatted speech
                           ▼
               ┌────────────────────────┐
               │     CLI INTERFACE      │
               └────────────────────────┘

```

---

### 2. Стек и Требования к Окружению

* **Язык разработки:** Python 3.11+


* **Валидация данных и схемы:** `pydantic>=2.7.0`

* **Консольный UI:** `rich>=13.7.0` (таблицы, цветные панели, форматирование статуса)


* **Инференс-клиенты:**
* `httpx>=0.27.0` (асинхронное взаимодействие с API vLLM/Ollama/OpenAI-compatible)


* `torch`, `transformers`, `accelerate` (при локальном прямом запуске весов `ZefanCai/Open-Jev-9B`)



---

### 3. Спецификация Контрактов Данных (Data Contracts)

#### 3.1. Модель игрового состояния (`GameState`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class PlayerOffer(BaseModel):
    investment: int = Field(gt=0, description="Запрашиваемая сумма инвестиций в €")
    equity: float = Field(gt=0, le=100, description="Предлагаемая доля компании в %")

class InvestorState(BaseModel):
    budget: int = Field(gt=0, description="Максимальный свободный капитал инвестора")
    max_equity: float = Field(gt=0, le=100, description="Желаемая планка доли")
    interest: float = Field(ge=0.0, le=1.0, description="Коэффициент заинтересованности 0..1")
    patience: int = Field(ge=0, description="Остаток раундов терпения до срыва сделки")

class GameState(BaseModel):
    company_valuation: int = Field(gt=0, description="Текущая рыночная оценка компании")
    player_offer: PlayerOffer
    investor: InvestorState
    history: list[str] = Field(default_factory=list)

```

#### 3.2. Схема критериев для Jev Engine (`JevCriteria`)

```python
class JevQueryConfig(BaseModel):
    type: Literal["score", "choice", "noul"]
    instructions: str
    criteria: list[str] | dict[str, str] | None = None

class JevEvaluationSchema(BaseModel):
    accept: JevQueryConfig
    reaction: JevQueryConfig
    good_deal: JevQueryConfig

class JevEvaluationResult(BaseModel):
    accept_score: int = Field(ge=1, le=5, description="1 - Definitely reject, 5 - Definitely accept")
    reaction: Literal["accept", "counter", "reject", "walk_away"]
    good_deal_confidence: float = Field(ge=0.0, le=1.0, description="Вероятность утверждения noul")

```

---

### 4. Взаимодействие с Open-Jev-9B Engine

1. **Формирование промпта для Jev:**
Входное состояние `GameState` сериализуется в чистый JSON без текстовых украшений и передается в контекст Jev:
```json
{
  "valuation": 2000000,
  "player_offer": {"investment": 500000, "equity": 15},
  "investor_profile": {"budget": 700000, "max_equity": 30, "interest": 0.72, "patience": 3},
  "history": ["Investor offered €500k for 30%", "Player rejected"]
}

```


2. **Параллельная оценка по трем осям:**
* `accept` (`score`, шкала 1–5): интегральная вероятность согласия.
* `reaction` (`choice`): определение дискретного действия (`accept` | `counter` | `reject` | `walk_away`).
* `good_deal` (`noul`): бинарная оценка выгоды сделки для NPC.


3. **Детерминированный арбитраж (Guardrail):**
Если `investor.patience <= 1` или предложенная игроком доля опускает оценку ниже допустимого порога инвестора, результат `reaction` форсируется в `walk_away`, исключая галлюцинацию эвристики модели.

---

### 5. Спецификация Voice-LLM (Генератора диалога)

Voice LLM **категорически запрещено** менять статус переговоров или предлагать числа вне рамок, вычисленных системой.

#### Системный промпт для Voice LLM:

```text
Ты — жесткий, прагматичный венчурный инвестор. Твоя задача — озвучить решение, которое УЖЕ ПРИНЯТО аналитическим отделом.
Ты не имеешь права соглашаться, если решение REJECT или COUNTER.
Ты не имеешь права предлагать другие суммы и проценты, кроме указанных в директиве.

ДИРЕКТИВА:
- Действие: {reaction}
- Готовность принять сделку: {accept_score}/5
- Выгодность предложения: {good_deal_status}
- Встречные условия (если action=counter): {counter_offer_investment}€ за {counter_offer_equity}%

Сгенерируй только прямую речь персонажа на русском языке (до 3 предложений). Не добавляй вводных слов автора.

```

---

### 6. Игровой цикл CLI и правила раундов

1. **Фаза инициализации:**
* Отображение вводной карточки проекта: Оценка стартапа, бюджет инвестора, терпение.
* Вывод первого встречного оффера инвестора.


2. **Фаза ввода игрока:**
* Валидация ввода через регулярные выражения / CLI-парсер (сумма в € и доля в %).
* Детекция команд выхода (`exit`, `give up`).


3. **Фаза инференса Jev:**
* Синхронизация `GameState`.


* Запрос к Open-Jev-9B -> получение `JevEvaluationResult`.
* Декремент терпения (`investor.patience -= 1`).


4. **Фаза синтеза ответа:**
* Сборка директивы для Voice LLM.
* Получение текстовой реплики инвестора.


5. **Фаза разрешения раунда:**
* Если `reaction == "accept"`: победа игрока, сделка закрыта.
* Если `reaction == "walk_away"` или `patience == 0`: проигрыш, инвестор покинул переговоры.
* Если `reaction in ("counter", "reject")`: переход к следующему раунду.



---

### 7. Архитектура файлов проекта

```text
startup_deal_game/
├── config.py                 # Переменные окружения (.env: LLM_URL, JEV_MODEL_PATH)
├── schemas.py                # Pydantic-контракты GameState, JevEvaluation, VoicePrompt
├── engine/
│   ├── jev_evaluator.py      # Адаптер вызова Open-Jev-9B (локальный или REST)
│   ├── arbiter.py            # Математика контрпредложений и проверка границ терпения
│   └── voice_llm.py          # Адаптер генератора диалогов (Ollama / vLLM / OpenAI)
├── game/
│   ├── session.py            # Менеджер сессии, история раундов, стейт игры
│   └── cli_renderer.py       # Rich UI: рендеринг диалогов, оценок и панелей
└── main.py                   # Точка входа: интерактивный игровой цикл

```

---

### 8. Критерии приемки (Definition of Done)

1. **Изоляция принятия решений:** Текстовая LLM ни при каких обстоятельствах не переопределяет статус сделки (при `reaction: "reject"` инвестор никогда не говорит «Я согласен»).
2. **Корректный парсинг Jev:** Поля `accept`, `reaction`, `good_deal` извлекаются без ошибок типов данных с жесткой валидацией Pydantic.


3. **Завершаемость игры:** Игра гарантированно завершается успехом (сделка заключена) либо поражением (инвестор ушел) максимум за отведенное число раундов терпения (`patience`).