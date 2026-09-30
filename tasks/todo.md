# Todo Checklist: Open-JEV Negotiator

Implementation checklist organized in vertical slices following the [Implementation Plan](plan.md) and technical specification in [spec.md](../spec.md).

---

## Slice 1: Project Setup & Core Data Contracts
**Estimated Size:** S (5 files)  
**Goal:** Initialize project infrastructure, dependency management with `uv`, configuration settings, and complete Pydantic data models.

- [x] **1.1 Configure Project Dependencies & Packaging**
  - Create `pyproject.toml` targeting Python `>=3.11` with dependencies: `pydantic>=2.7.0`, `pydantic-settings>=2.2.0`, `rich>=13.7.0`, `httpx>=0.27.0`, `pytest>=8.0.0`, `pytest-asyncio>=0.23.0`.
  - Files: `pyproject.toml`
  - Verification: `uv sync`

- [x] **1.2 Implement Application Settings & Environment Config**
  - Implement `Settings` in `config.py` using `pydantic_settings.BaseSettings`:
    - `LLM_URL`, `LLM_API_KEY`, `LLM_MODEL`, `JEV_URL`, `JEV_MODEL_PATH`, `MOCK_MODE`.
    - Game defaults: `DEFAULT_COMPANY_VALUATION=2000000`, `DEFAULT_INVESTOR_BUDGET=700000`, `DEFAULT_INVESTOR_MAX_EQUITY=30.0`, `DEFAULT_INVESTOR_INTEREST=0.70`, `DEFAULT_INVESTOR_PATIENCE=4`.
  - Files: `config.py`
  - Verification: `uv run python -c "from config import settings; print(settings.model_dump())"`

- [x] **1.3 Implement Pydantic Domain Contracts & Schemas**
  - Implement schemas in `schemas.py`:
    - `PlayerOffer`: `investment: int > 0`, `equity: float (0, 100]`, computed `implied_valuation`.
    - `InvestorState`: `budget: int > 0`, `max_equity: float (0, 100]`, `interest: float [0, 1]`, `patience: int >= 0`.
    - `GameState`: `company_valuation: int`, `player_offer: PlayerOffer`, `investor: InvestorState`, `history: list[str]`, helper `to_jev_payload() -> dict`.
    - `JevQueryConfig`, `JevEvaluationSchema`, `JevEvaluationResult`: `accept_score: int [1, 5]`, `reaction: Literal["accept", "counter", "reject", "walk_away"]`, `good_deal_confidence: float [0, 1]`.
    - `VoiceDirective`: `reaction`, `accept_score`, `good_deal_status`, `counter_offer_investment`, `counter_offer_equity`.
    - `GameStatus`, `RoundResult`.
  - Files: `schemas.py`
  - Verification: `uv run python -c "import schemas"`

- [x] **1.4 Unit Tests for Schemas & Serialization**
  - Implement test cases in `tests/test_schemas.py` and `tests/conftest.py`:
    - Valid and invalid values for `PlayerOffer` and `InvestorState`.
    - Calculation of `implied_valuation`.
    - JSON serialization matching spec section 4.1 format.
    - `JevEvaluationResult` constraints validation.
  - Files: `tests/conftest.py`, `tests/test_schemas.py`
  - Verification: `uv run pytest tests/test_schemas.py -v`

> [!NOTE]
> **Checkpoint 1:** Core data contracts, environment configuration, and test infrastructure are verified.

---

## Slice 2: Deterministic Decision Arbiter & Guardrails
**Estimated Size:** S (2 files)  
**Goal:** Implement business logic guardrails, counter-offer calculations, and prompt directive construction.

- [x] **2.1 Implement Decision Arbiter Engine**
  - Implement `DecisionArbiter` in `engine/arbiter.py`:
    - `check_guardrails(state: GameState, raw_result: JevEvaluationResult) -> JevEvaluationResult`:
      - If `state.investor.patience <= 1` and `raw_result.reaction != "accept"` -> override to `"walk_away"`.
      - If implied valuation $\ge 2.5 \times \text{valuation}$ or `investment > budget` -> override to `"walk_away"` or `"reject"`.
    - `calculate_counter_offer(state: GameState) -> tuple[int, float]`:
      - Rational investor counter-terms respecting budget ceiling and interest level.
    - `build_voice_directive(state: GameState, evaluation: JevEvaluationResult) -> VoiceDirective`:
      - Builds payload for Voice LLM with exact directive numbers.
  - Files: `engine/__init__.py`, `engine/arbiter.py`
  - Verification: `uv run python -c "from engine.arbiter import DecisionArbiter"`

- [x] **2.2 Unit Tests for Arbiter Guardrails & Math**
  - Implement unit tests in `tests/test_arbiter.py`:
    - Patience guardrail triggers `walk_away` at $\le 1$ patience.
    - Valuation threshold breach triggers `walk_away`.
    - Counter-offer calculates fair equity within bounds $[0, max\_equity]$.
    - Directive generation accurately reflects arbiter decision.
  - Files: `tests/test_arbiter.py`
  - Verification: `uv run pytest tests/test_arbiter.py -v`

> [!NOTE]
> **Checkpoint 2:** Deterministic guardrails and counter-offer calculation pass all unit tests.

---

## Slice 3: Open-JEV Evaluator & Offline Mock
**Estimated Size:** M (2 files)  
**Goal:** Implement the evaluation adapter capable of querying Open-Jev-9B runtime with deterministic offline fallback.

- [x] **3.1 Implement Jev Evaluator (Mock & HTTP Client)**
  - Implement in `engine/jev_evaluator.py`:
    - `BaseJevEvaluator` abstract protocol.
    - `MockJevEvaluator`: Quantitative valuation-ratio heuristic scoring:
      - Ratio $\rho = \text{implied\_valuation} / \text{valuation}$.
      - Computes `accept_score` (1–5), `reaction` (`accept`, `counter`, `reject`, `walk_away`), and `good_deal_confidence` (0.0..1.0).
    - `HttpJevEvaluator`: Async HTTP client sending formatted JSON payload to Open-Jev endpoint with schema parsing.
    - Factory `get_jev_evaluator(settings: Settings) -> BaseJevEvaluator`.
  - Files: `engine/jev_evaluator.py`
  - Verification: `uv run python -c "from engine.jev_evaluator import MockJevEvaluator"`

- [x] **3.2 Unit Tests for Jev Evaluator**
  - Implement unit tests in `tests/test_jev_evaluator.py`:
    - Test `MockJevEvaluator` across good, reasonable, stretched, and absurd offers.
    - Test `HttpJevEvaluator` with mock HTTP responses, strict validation, and error raises.
    - Test output strictly adheres to `JevEvaluationResult`.
  - Files: `tests/test_jev_evaluator.py`
  - Verification: `uv run pytest tests/test_jev_evaluator.py -v`

> [!NOTE]
> **Checkpoint 3:** Jev Evaluator produces validated quantitative evaluation results offline and over HTTP.

---

## Slice 4: Voice LLM Generator & Prompt Guardrails
**Estimated Size:** M (2 files)  
**Goal:** Implement Russian persona dialogue generation strictly adhering to Arbiter directives, with safety validation.

- [x] **4.1 Implement Voice LLM Generator (Mock & HTTP Client)**
  - Implement in `engine/voice_llm.py`:
    - Exact Russian system prompt template from spec section 5.
    - `BaseVoiceLLM` abstract protocol.
    - `MockVoiceLLM`: High-conviction pre-authored Russian venture capitalist responses for each reaction type (`accept`, `counter`, `reject`, `walk_away`) with dynamic parameter insertion.
    - `HttpVoiceLLM`: Asynchronous client targeting Ollama/vLLM/OpenAI `/chat/completions`.
    - `validate_dialogue_guardrails(text: str, directive: VoiceDirective) -> str`: Safety filter verifying no contradictory commitments occur.
    - Factory `get_voice_llm(settings: Settings) -> BaseVoiceLLM`.
  - Files: `engine/voice_llm.py`
  - Verification: `uv run python -c "from engine.voice_llm import MockVoiceLLM"`

- [x] **4.2 Unit Tests for Voice LLM**
  - Implement unit tests in `tests/test_voice_llm.py`:
    - Verify prompt formatting incorporates directive values accurately.
    - Verify `MockVoiceLLM` yields valid Russian dialogue for all actions.
    - Verify safety filter replaces contradictory text if an LLM hallucinates acceptance on reject.
  - Files: `tests/test_voice_llm.py`
  - Verification: `uv run pytest tests/test_voice_llm.py -v`

> [!NOTE]
> **Checkpoint 4:** Dialogue generation operates strictly under directive bounds without hallucinating agreement.

---

## Slice 5: Game Session State Machine
**Estimated Size:** M (3 files)  
**Goal:** Coordinate round lifecycle, patience decrements, evaluation, arbiter decisions, and win/loss resolution.

- [x] **5.1 Implement GameSession State Machine**
  - Implement `GameSession` in `game/session.py`:
    - Initialize game state with founder and investor defaults.
    - Opening investor stance and initial counter-offer.
    - `async def step(player_offer: PlayerOffer) -> RoundResult`:
      1. Register player offer.
      2. Query Jev evaluation.
      3. Run Arbiter guardrails and counter-offer calculation.
      4. Decrement patience: `investor.patience -= 1`.
      5. Generate Voice LLM dialogue line.
      6. Determine game outcome:
         - `reaction == "accept"` -> `GameStatus.WON`
         - `reaction == "walk_away"` or `patience == 0` -> `GameStatus.LOST`
         - else -> `GameStatus.IN_PROGRESS`
      7. Record round in history.
  - Files: `game/__init__.py`, `game/session.py`
  - Verification: `uv run python -c "from game.session import GameSession"`

- [x] **5.2 Unit & Integration Tests for Game Session**
  - Implement tests in `tests/test_session.py`:
    - Successful negotiation flow ending in victory.
    - Failed negotiation due to patience exhaustion (max rounds).
    - Failed negotiation due to predatory offer (instant walk away).
    - Proper history accumulation across multiple rounds.
  - Files: `tests/test_session.py`
  - Verification: `uv run pytest tests/test_session.py -v`

> [!NOTE]
> **Checkpoint 5:** Full negotiation state machine runs deterministically across turns and transitions to final states.

---

## Slice 6: Rich Terminal UI Renderer
**Estimated Size:** M (2 files)  
**Goal:** Implement rich console presentation with tables, panels, colored speech bubbles, patience bars, and telemetry.

- [x] **6.1 Implement CliRenderer**
  - Implement in `game/cli_renderer.py`:
    - `render_banner()`: Startup Deal Negotiation header & visual logo.
    - `render_pitch_card(company_valuation, investor_budget)`: Startup card panel.
    - `render_investor_panel(investor_state)`: Investor capital, interest percentage, patience hearts (`[red]♥ ♥ ♥[/red]`).
    - `render_dialogue(speaker: str, text: str)`: Styled panels (Cyan for Founder, Yellow/Magenta for Investor).
    - `render_eval_telemetry(eval_result: JevEvaluationResult)`: Formatted telemetry panel displaying score, reaction, and confidence.
    - `render_game_over(status: GameStatus, session: GameSession)`: Victory fanfare or Defeat walk-away screen.
    - `render_help()`: Command instructions and input syntax.
  - Files: `game/cli_renderer.py`
  - Verification: `uv run python -c "from game.cli_renderer import CliRenderer"`

- [x] **6.2 Unit Tests for CLI Renderer**
  - Implement tests in `tests/test_cli_renderer.py`:
    - Verify all render methods execute without exceptions on mock console.
    - Verify patience visual indicator renders correct number of hearts/bars.
    - Verify dialogue panels format player and investor distinctly.
  - Files: `tests/test_cli_renderer.py`
  - Verification: `uv run pytest tests/test_cli_renderer.py -v`

> [!NOTE]
> **Checkpoint 6:** Rich UI components render all game panels, dialogue, and status indicators cleanly.

---

## Slice 7: CLI Parser & End-to-End Game Integration
**Estimated Size:** M (3 files)  
**Goal:** Implement natural language offer parsing, REPL game loop, CLI arguments, and complete documentation.

- [x] **7.1 Implement Player Input Parser & REPL Loop**
  - Implement in `main.py`:
    - `parse_player_offer(raw_input: str) -> PlayerOffer | str`: Regex matching expressions such as `500k 10%`, `500000 15`, `€400,000 for 12.5%`, `300k за 20%`, `exit`, `quit`, `give up`, `help`, `status`.
    - CLI arguments via `argparse`: `--mock`, `--llm-url`, `--jev-url`, `--budget`, `--valuation`, `--rounds`, `--test-run`.
    - Async interactive game loop connecting `GameSession` and `CliRenderer`.
  - Files: `main.py`
  - Verification: `uv run python main.py --help`

- [x] **7.2 CLI & End-to-End Integration Tests**
  - Implement tests in `tests/test_main.py`:
    - Test regex parsing of various input formats and edge-case inputs.
    - Test `--test-run` flag running an automated simulated session non-interactively.
  - Files: `tests/test_main.py`
  - Verification: `uv run pytest tests/test_main.py -v`

- [x] **7.3 Documentation & Run Guide**
  - Create `README.md` with:
    - Quickstart with `uv run python main.py --mock`.
    - Real model execution guide (Ollama, vLLM, Open-Jev-9B).
    - Game rules, negotiation tactics, and input format examples.
    - Architecture diagram and component summary.
  - Files: `README.md`
  - Verification: Inspect `README.md` for clarity and completeness.

> [!NOTE]
> **Final Checkpoint:** Full test suite passes (`uv run pytest`) and game runs seamlessly in interactive CLI and automated test run modes.
