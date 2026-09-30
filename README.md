# 💼 Open-JEV Negotiator

> **CLI-игра «Переговоры со стартап-инвестором» (Startup Venture Deal or No Deal)**  
> Инновационная пошаговая игра с двухконтурным искусственным интеллектом (Quantitative Open-JEV Brain + Voice LLM Persona + Deterministic Guardrails).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/badge/contracts-pydantic--v2-red.svg)](https://docs.pydantic.dev/)
[![Rich CLI](https://img.shields.io/badge/ui-rich-green.svg)](https://github.com/Textualize/rich)
[![Tests](https://img.shields.io/badge/tests-233%20passed-brightgreen.svg)]()

---

## 📖 Содержание / Table of Contents
- [1. Концепция и Архитектура / Concept & Architecture](#1-концепция-и-архитектура--concept--architecture)
- [2. Быстрый старт / Quickstart](#2-быстрый-старт--quickstart)
- [3. Подключение реальных моделей / Real Models Guide](#3-подключение-реальных-моделей--real-models-guide)
- [4. Правила игры и синтаксис / Game Rules & Syntax](#4-правила-игры-и-синтаксис--game-rules--syntax)
- [5. Запуск тестов / Running Tests](#5-запуск-тестов--running-tests)
- [6. Структура проекта / Project Structure](#6-структура-проекта--project-structure)
- [7. English Summary](#7-english-summary)

---

## 1. Концепция и Архитектура / Concept & Architecture

В традиционных LLM-играх языковые модели страдают от **галлюцинаций согласия**: даже получив указание «отказать», модель под влиянием учтивости может сказать «Ваше предложение интересное, давайте заключим сделку».

**Open-JEV Negotiator** решает эту проблему через **двухконтурную архитектуру (Two-tier Orchestration)** с детерминированным арбитражем:

```mermaid
flowchart TD
    Founder["🧑 Основатель (Игрок)\nОффер: €500,000 за 15%"] --> Parser["CLI Input Parser\n(Regex / Валидация)"]
    Parser --> GameState["💾 GameState Store\n(Оценка, бюджет, история)"]
    
    subgraph Tier1 ["Tier 1: Детерминированный мозг (Evaluation Brain)"]
        GameState --> Jev["🧠 Open-JEV Engine\n(Open-Jev-9B / Mock Evaluator)\nОценка: score (1-5), reaction, good_deal"]
        Jev --> Arbiter["⚖️ Decision Arbiter (Guardrails)\n• Предел терпения (patience <= 1)\n• Потолок оценки (<= 2.5x)\n• Расчет контр-предложения"]
    end
    
    Arbiter --> Directive["📋 Voice Directive\n(Действие, оценка, точные числа)"]
    
    subgraph Tier2 ["Tier 2: Речевой аппарат (Persona Generation)"]
        Directive --> VoiceLLM["🗣️ Voice LLM (Ollama / vLLM / Mock)\n(Жесткий VC инвестор на русском)\nЗапрещено менять условия!"]
    end
    
    VoiceLLM --> UI["🖥️ Rich CLI Renderer\n(Панели диалога, телеметрия JEV, карточки)"]
    UI --> Founder
```

### Ключевые компоненты:
1. **Tier 1: Open-JEV Evaluation Engine (`engine/jev_evaluator.py`)**  
   Оценивает формализованное состояние сделки по трем дискретным осям:
   - `accept` (`score`, 1–5): интегральная готовность принять предложение.
   - `reaction` (`choice`): дискретное решение (`accept`, `counter`, `reject`, `walk_away`).
   - `good_deal` (`noul`, 0.0–1.0): вероятность финансовой привлекательности сделки.

2. **Deterministic Decision Arbiter & Guardrails (`engine/arbiter.py`)**  
   Абсолютный барьер безопасности бизнес-логики:
   - При `patience <= 1` и отсутствии согласия принудительно вызывает `walk_away`.
   - При завышении оценки $\ge 2.5 \times$ или превышении бюджета фонда принудительно блокирует сделку.
   - Вычисляет математически обоснованное встречное предложение инвестора.

3. **Tier 2: Voice LLM Persona (`engine/voice_llm.py`)**  
   Генерирует художественную реплику жесткого венчурного партнера на русском языке, строго подчиняясь директиве арбитра. Модель защищена пост-валидатором `validate_dialogue_guardrails`.

4. **Rich Terminal Presentation (`game/cli_renderer.py`)**  
   Интерактивный терминальный интерфейс с индикаторами терпения (`♥ ♥ ♥`), подробной телеметрией JEV и стилизованными карточками раундов.

---

## 2. Быстрый старт / Quickstart

Проект управляется через современный менеджер пакетов [Astral uv](https://github.com/astral-sh/uv).

### 2.1 Установка зависимостей
```bash
# Синхронизация зависимостей в виртуальное окружение
uv sync
```

### 2.2 Запуск игры в автономном Mock-режиме (без внешних API)
Игра полностью готова к запуску прямо из коробки с встроенным детерминированным эмулятором Open-JEV-9B и Voice LLM:
```bash
uv run python main.py --mock
```

### 2.3 Автоматический тестовый прогон (Non-interactive Test Run)
Позволяет моментально проверить полный игровой цикл от первого питча до закрытия раунда:
```bash
uv run python main.py --test-run --mock
```

### 2.4 Настройка параметров переговоров через аргументы CLI
```bash
# Инвестор с увеличенным бюджетом и строгим терпением
uv run python main.py --mock --valuation 3000000 --budget 900000 --patience 3
```

Флаги CLI:
| Флаг | Описание | Значение по умолчанию |
|---|---|---|
| `--mock` | Принудительный оффлайн-режим (эмуляция JEV и LLM) | `False` (или из `.env`) |
| `--test-run` | Неинтерактивная симуляция сделки | `False` |
| `--valuation` | Рыночная оценка вашего стартапа в € | `2,000,000` |
| `--budget` | Доступный капитал венчурного фонда в € | `700,000` |
| `--patience` / `--rounds` | Количество раундов терпения до ухода инвестора | `4` |
| `--llm-url` | URL эндпоинта Voice LLM (Ollama, vLLM, OpenAI API) | `http://localhost:11434/v1` |
| `--jev-url` | URL сервиса инференса Open-Jev-9B | `http://localhost:8000/eval` |

---

## 3. Подключение реальных моделей / Real Models Guide

Для работы с реальными нейросетями настройте файл окружения `.env`.

### 3.1 Конфигурация `.env`
Создайте файл `.env` в корне проекта (по шаблону):
```env
# Режим работы (False для подключения к реальным сервисам)
MOCK_MODE=False

# Voice LLM (Ollama, vLLM или любой OpenAI-совместимый эндпоинт)
LLM_URL=http://localhost:11434/v1
LLM_API_KEY=dummy
LLM_MODEL=llama3:latest

# Open-JEV-9B Сервис
JEV_URL=http://localhost:8000/eval
JEV_MODEL_PATH=ZefanCai/Open-Jev-9B

# Параметры игры по умолчанию
DEFAULT_COMPANY_VALUATION=2000000
DEFAULT_INVESTOR_BUDGET=700000
DEFAULT_INVESTOR_MAX_EQUITY=30.0
DEFAULT_INVESTOR_INTEREST=0.70
DEFAULT_INVESTOR_PATIENCE=4
```

### 3.2 Запуск Voice LLM через Ollama
1. Установите [Ollama](https://ollama.ai/) и загрузите модель (например, Llama-3 или Mistral):
   ```bash
   ollama run llama3
   ```
2. Сервис будет доступен по адресу `http://localhost:11434/v1`.

### 3.3 Запуск Open-Jev-9B
Модель [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev) разворачивается через совместимый REST microservice или vLLM, принимающий JSON-запрос состояния игры `GameState` и возвращающий `JevEvaluationResult`:
```json
{
  "accept_score": 4,
  "reaction": "counter",
  "good_deal_confidence": 0.72
}
```

---

## 4. Правила игры и синтаксис / Game Rules & Syntax

### 4.1 Математика сделки
- **Имплицитная оценка компании (Implied Valuation):**
  $$\text{Implied Valuation} = \frac{\text{Investment}}{\text{Equity} / 100}$$
- **Коэффициент оценки ($\rho$):**
  $$\rho = \frac{\text{Implied Valuation}}{\text{Company Valuation}}$$
  - $\rho \le 0.75$: Высокая выгода для инвестора — высокая вероятность согласия (`accept`).
  - $0.75 < \rho \le 1.15$: Справедливая рыночная зона — инвестор торгуется (`counter`).
  - $1.15 < \rho \le 2.50$: Завышенная оценка — риск отказа (`reject`).
  - $\rho > 2.50$ или $\text{Investment} > \text{Budget}$: Хищническое предложение — немедленный срыв переговоров (`walk_away`).

### 4.2 Поддерживаемые форматы ввода
CLI-парсер распознает естественный язык, сокращения и валюты:
- `500k 10%` — €500,000 за 10%
- `500000 15` — прямой числовой ввод (€500,000 за 15%)
- `€400,000 for 12.5%` — международный формат
- `1.2m 20%` — обозначение миллионов (€1,200,000 за 20%)
- `300к за 15%` — русскоязычный формат с предлогом
- `450k for 18%` — с англоязычным предлогом
- `250 тыс за 10%` / `1.5 млн 20%` — текстовые множители

### 4.3 Команды управления
- `help` / `?` — показать справку по правилам и синтаксису.
- `status` — повторно вывести карточку стартапа и профиль инвестора.
- `history` — показать хронологию всех ходов текущей сессии.
- `exit` / `quit` / `give up` — сдаться и прервать переговоры.

---

## 5. Запуск тестов / Running Tests

Проект разработан по строгой методологии **Test-Driven Development (TDD)** с полным покрытием контрактов, логики арбитража, адаптеров моделей, UI и CLI-парсера.

```bash
# Запуск полного тестового набора
uv run pytest

# Подробный вывод со всеми тестовыми кейсами
uv run pytest -v

# Проверка конкретного модуля (например, CLI и парсера)
uv run pytest tests/test_main.py -v
```

Текущее состояние: **233 passed** (0 failures, 0 regressions).

---

## 6. Структура проекта / Project Structure

```text
open-jev-negotiator/
├── config.py                 # Pydantic Settings & параметры окружения (.env)
├── schemas.py                # Pydantic v2 контракты (GameState, JevResult, VoiceDirective)
├── engine/
│   ├── arbiter.py            # Детерминированный арбитр: guardrails и контр-офферы
│   ├── jev_evaluator.py      # Адаптер Open-JEV (Mock и HTTP REST клиенты)
│   └── voice_llm.py          # Адаптер Voice LLM (генератор русской речи персонажа)
├── game/
│   ├── session.py            # Координатор игровой сессии и машины состояний
│   └── cli_renderer.py       # Rich UI рендерер (баннеры, панели, сердечки терпения)
├── tests/
│   ├── conftest.py           # Pytest фикстуры
│   ├── test_schemas.py       # Тесты контрактов и валидации данных
│   ├── test_arbiter.py       # Тесты guardrails и арбитража
│   ├── test_jev_evaluator.py # Тесты количественного скоринга Jev
│   ├── test_voice_llm.py     # Тесты речевого LLM и фильтра безопасности
│   ├── test_session.py       # Интеграционные тесты конечного автомата
│   ├── test_cli_renderer.py  # Тесты визуализации в консоли
│   └── test_main.py          # Тесты CLI парсера, аргументов и E2E сценариев
├── main.py                   # Точка входа: REPL цикл и неинтерактивная симуляция
├── pyproject.toml            # Описание проекта и зависимостей uv
└── README.md                 # Документация проекта
```

---

## 7. English Summary

**Open-JEV Negotiator** is a CLI negotiation game showcasing a **two-tier AI architecture**:
1. **Tier 1 (Quantitative Evaluation Brain):** Open-JEV-9B reasoning model evaluates game state along discrete axes (`accept_score`, discrete `reaction`, `good_deal` confidence).
2. **Deterministic Decision Arbiter:** Physical and financial guardrails enforce mathematical bounds, calculate counter-offers, and terminate negotiations when patience expires, completely preventing LLM hallucination of false agreements.
3. **Tier 2 (Voice LLM Dialogue):** Local or remote LLM (via Ollama/vLLM) generates authentic Russian venture capitalist dialogue adhering strictly to arbiter directives.
4. **Rich Terminal UI:** Polished interactive terminal interface with patience hearts, evaluation telemetry, and pitch cards.

### Quick Run:
```bash
uv sync
uv run python main.py --mock
```
