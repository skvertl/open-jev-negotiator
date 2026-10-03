"""GameSession state machine managing negotiation rounds, arbiter rules, and outcomes."""
from config import Settings, settings
from engine.arbiter import DecisionArbiter
from engine.jev_evaluator import BaseJevEvaluator, get_jev_evaluator
from engine.voice_llm import BaseVoiceLLM, get_voice_llm
from schemas import (
    GameState,
    GameStatus,
    InvestorState,
    JevEvaluationResult,
    PlayerOffer,
    RoundResult,
    VoiceDirective,
)


def _build_default_game_state(
    cfg: Settings,
    company_valuation: int | None = None,
    investor_budget: int | None = None,
    investor_max_equity: float | None = None,
    investor_interest: float | None = None,
    investor_patience: int | None = None,
) -> GameState:
    """Construct a GameState with custom parameters or defaults from Settings."""
    val = company_valuation if company_valuation is not None else cfg.DEFAULT_COMPANY_VALUATION
    budget = investor_budget if investor_budget is not None else cfg.DEFAULT_INVESTOR_BUDGET
    max_equity = (
        investor_max_equity
        if investor_max_equity is not None
        else cfg.DEFAULT_INVESTOR_MAX_EQUITY
    )
    interest = (
        investor_interest
        if investor_interest is not None
        else cfg.DEFAULT_INVESTOR_INTEREST
    )
    patience = (
        investor_patience
        if investor_patience is not None
        else cfg.DEFAULT_INVESTOR_PATIENCE
    )

    initial_investment = min(500_000, budget)
    initial_equity = round(min(20.0, max_equity), 1)

    return GameState(
        company_valuation=val,
        player_offer=PlayerOffer(investment=initial_investment, equity=initial_equity),
        investor=InvestorState(
            budget=budget,
            max_equity=max_equity,
            interest=interest,
            patience=patience,
        ),
        history=[],
    )


class GameSession:
    """Manages the full lifecycle and state machine of a startup negotiation session."""

    def __init__(
        self,
        state: GameState | None = None,
        evaluator: BaseJevEvaluator | None = None,
        arbiter: DecisionArbiter | None = None,
        voice_llm: BaseVoiceLLM | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        cfg = settings_obj or settings
        self._state = state if state is not None else _build_default_game_state(cfg)
        self._evaluator = evaluator or get_jev_evaluator(cfg)
        self._arbiter = arbiter or DecisionArbiter()
        self._voice_llm = voice_llm or get_voice_llm(cfg)
        self._round_number: int = 0
        self._status: GameStatus = GameStatus.IN_PROGRESS
        self._round_results: list[RoundResult] = []
        if self._state.last_counter_offer is None:
            self._state.last_counter_offer = self._arbiter.calculate_counter_offer(self._state)

    @classmethod
    def create_session(
        cls,
        company_valuation: int | None = None,
        investor_budget: int | None = None,
        investor_max_equity: float | None = None,
        investor_interest: float | None = None,
        investor_patience: int | None = None,
        evaluator: BaseJevEvaluator | None = None,
        arbiter: DecisionArbiter | None = None,
        voice_llm: BaseVoiceLLM | None = None,
        settings_obj: Settings | None = None,
    ) -> "GameSession":
        """Factory method to construct a GameSession with custom parameters or defaults."""
        cfg = settings_obj or settings
        state = _build_default_game_state(
            cfg=cfg,
            company_valuation=company_valuation,
            investor_budget=investor_budget,
            investor_max_equity=investor_max_equity,
            investor_interest=investor_interest,
            investor_patience=investor_patience,
        )
        return cls(
            state=state,
            evaluator=evaluator or get_jev_evaluator(cfg),
            arbiter=arbiter or DecisionArbiter(),
            voice_llm=voice_llm or get_voice_llm(cfg),
            settings_obj=cfg,
        )

    @property
    def state(self) -> GameState:
        """Current game state snapshot."""
        return self._state

    @property
    def status(self) -> GameStatus:
        """Current game outcome status."""
        return self._status

    @property
    def round_number(self) -> int:
        """Current round index (0 before first step)."""
        return self._round_number

    @property
    def is_active(self) -> bool:
        """True if game is in progress and can accept steps."""
        return self._status == GameStatus.IN_PROGRESS

    @property
    def is_game_over(self) -> bool:
        """True if game has concluded (won, lost, or quit)."""
        return self._status != GameStatus.IN_PROGRESS

    @property
    def history(self) -> list[str]:
        """Chronological list of negotiation round summaries."""
        return list(self._state.history)

    @property
    def round_results(self) -> list[RoundResult]:
        """Historical list of all completed round results."""
        return list(self._round_results)

    @property
    def patience(self) -> int:
        """Current remaining investor patience."""
        return self._state.investor.patience

    @property
    def company_valuation(self) -> int:
        """Market valuation of the startup."""
        return self._state.company_valuation

    @property
    def investor(self) -> InvestorState:
        """Current investor traits and budget."""
        return self._state.investor

    def get_opening_stance(self) -> tuple[tuple[int, float], str]:
        """Generate initial investor opening terms and speech line without advancing round."""
        counter_inv, counter_eq = self._arbiter.calculate_counter_offer(self._state)
        speech = (
            f"Приветствую. Я готов рассмотреть инвестиции до {counter_inv}€ "
            f"за {counter_eq}% вашей компании. Жду ваше встречное предложение."
        )
        return (counter_inv, counter_eq), speech

    async def step(self, player_offer: PlayerOffer) -> RoundResult:
        """Execute a single negotiation round step."""
        if self.is_game_over:
            raise ValueError(
                f"Cannot step in finished game (current status: {self._status.value})"
            )

        self._round_number += 1
        self._state.player_offer = player_offer

        # 1. Query quantitative Jev evaluation
        raw_eval = await self._evaluator.evaluate(self._state)

        # 2. Run deterministic Arbiter guardrails and counter-offer arbitration
        evaluation, directive = self._arbiter.arbitrate(self._state, raw_eval)

        # Update last counter offer if investor countered
        if evaluation.reaction == "counter":
            if (
                directive.counter_offer_investment is not None
                and directive.counter_offer_equity is not None
            ):
                self._state.last_counter_offer = (
                    directive.counter_offer_investment,
                    directive.counter_offer_equity,
                )

        # 3. Decrement investor patience
        self._state.investor.patience = max(0, self._state.investor.patience - 1)

        # 4. Generate dialogue speech line with Voice LLM
        speech = await self._voice_llm.generate_speech(
            directive, history=self._state.history
        )

        # 5. Determine game outcome status
        if evaluation.reaction == "accept":
            self._status = GameStatus.WON
        elif evaluation.reaction == "walk_away" or self._state.investor.patience <= 0:
            self._status = GameStatus.LOST
        else:
            self._status = GameStatus.IN_PROGRESS

        # 6. Record round in history
        entry = (
            f"Раунд {self._round_number}: Игрок предложил {player_offer.investment}€ "
            f"за {player_offer.equity}%. "
            f"Решение инвестора: {evaluation.reaction}. Реплика: {speech}"
        )
        self._state.history.append(entry)

        # 7. Package and record RoundResult
        result = RoundResult(
            round_number=self._round_number,
            player_offer=player_offer,
            evaluation=evaluation,
            directive=directive,
            investor_dialogue=speech,
            game_status=self._status,
            investor_state=self._state.investor.model_copy(),
        )
        self._round_results.append(result)
        return result

    def give_up(self, status: GameStatus = GameStatus.QUIT) -> RoundResult:
        """Surrender and terminate negotiations."""
        if self.is_game_over:
            raise ValueError(
                f"Cannot give up in finished game (current status: {self._status.value})"
            )

        self._status = status
        self._round_number += 1
        dummy_eval = JevEvaluationResult(
            accept_score=1,
            reaction="walk_away",
            good_deal_confidence=0.0,
        )
        dummy_directive = VoiceDirective(
            reaction="walk_away",
            accept_score=1,
            good_deal_status="Переговоры прерваны",
        )
        speech = "Основатель прервал переговоры. Сделка не состоялась."
        entry = (
            f"Раунд {self._round_number}: Переговоры завершены досрочно игроком "
            f"(статус: {self._status.value})."
        )
        self._state.history.append(entry)

        result = RoundResult(
            round_number=self._round_number,
            player_offer=self._state.player_offer,
            evaluation=dummy_eval,
            directive=dummy_directive,
            investor_dialogue=speech,
            game_status=self._status,
            investor_state=self._state.investor.model_copy(),
        )
        self._round_results.append(result)
        return result


__all__ = ["GameSession"]
