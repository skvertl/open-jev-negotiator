"""Unit and integration tests for GameSession state machine."""
import pytest
from config import settings
from engine.arbiter import DecisionArbiter
from engine.jev_evaluator import BaseJevEvaluator, MockJevEvaluator
from engine.voice_llm import BaseVoiceLLM, MockVoiceLLM
from game.session import GameSession
from schemas import (
    GameState,
    GameStatus,
    InvestorState,
    JevEvaluationResult,
    PlayerOffer,
    RoundResult,
    VoiceDirective,
)


class StubJevEvaluator(BaseJevEvaluator):
    """Stub evaluator returning predetermined evaluation results."""

    def __init__(self, response: JevEvaluationResult) -> None:
        self.response = response
        self.calls: list[GameState] = []

    async def evaluate(self, state: GameState) -> JevEvaluationResult:
        self.calls.append(state.model_copy(deep=True))
        return self.response


class StubVoiceLLM(BaseVoiceLLM):
    """Stub voice generator returning predetermined speech."""

    def __init__(self, speech: str = "Реплика инвестора") -> None:
        self.speech = speech
        self.directives: list[VoiceDirective] = []

    async def generate_speech(
        self, directive: VoiceDirective, history: list[str] | None = None
    ) -> str:
        self.directives.append(directive.model_copy(deep=True))
        return self.speech


@pytest.fixture
def default_session() -> GameSession:
    """Session initialized with default settings and mock components."""
    return GameSession()


@pytest.fixture
def custom_game_state() -> GameState:
    """Custom game state fixture."""
    return GameState(
        company_valuation=3_000_000,
        player_offer=PlayerOffer(investment=600_000, equity=15.0),
        investor=InvestorState(
            budget=900_000,
            max_equity=35.0,
            interest=0.80,
            patience=5,
        ),
        history=[],
    )


class TestSessionInitialization:
    """Tests verifying default and custom GameSession initialization."""

    def test_default_initialization(self, default_session: GameSession):
        assert default_session.status == GameStatus.IN_PROGRESS
        assert default_session.round_number == 0
        assert default_session.is_active is True
        assert default_session.is_game_over is False
        assert default_session.history == []
        assert default_session.round_results == []
        assert default_session.patience == settings.DEFAULT_INVESTOR_PATIENCE
        assert default_session.company_valuation == settings.DEFAULT_COMPANY_VALUATION
        assert default_session.investor.budget == settings.DEFAULT_INVESTOR_BUDGET
        assert default_session.investor.max_equity == settings.DEFAULT_INVESTOR_MAX_EQUITY
        assert default_session.investor.interest == settings.DEFAULT_INVESTOR_INTEREST

    def test_default_initialization_clamps_equity_to_max_equity(self):
        low_max_equity_settings = settings.model_copy(
            update={"DEFAULT_INVESTOR_MAX_EQUITY": 15.0}
        )
        session = GameSession(settings_obj=low_max_equity_settings)
        assert session.state.player_offer.equity == 15.0

    def test_custom_initialization_with_state(self, custom_game_state: GameState):
        arbiter = DecisionArbiter()
        evaluator = MockJevEvaluator()
        voice_llm = MockVoiceLLM()

        session = GameSession(
            state=custom_game_state,
            evaluator=evaluator,
            arbiter=arbiter,
            voice_llm=voice_llm,
        )

        assert session.status == GameStatus.IN_PROGRESS
        assert session.round_number == 0
        assert session.is_active is True
        assert session.is_game_over is False
        assert session.company_valuation == 3_000_000
        assert session.investor.budget == 900_000
        assert session.patience == 5
        assert session.investor.max_equity == 35.0
        assert session.investor.interest == 0.80

    def test_create_session_factory_with_overrides(self):
        session = GameSession.create_session(
            company_valuation=5_000_000,
            investor_budget=1_000_000,
            investor_max_equity=25.0,
            investor_interest=0.65,
            investor_patience=6,
        )

        assert session.company_valuation == 5_000_000
        assert session.investor.budget == 1_000_000
        assert session.investor.max_equity == 25.0
        assert session.investor.interest == 0.65
        assert session.patience == 6
        assert session.status == GameStatus.IN_PROGRESS
        assert session.round_number == 0


class TestOpeningStance:
    """Tests for generating opening investor stance and terms."""

    def test_get_opening_stance_returns_valid_terms_and_speech(self, default_session: GameSession):
        terms, speech = default_session.get_opening_stance()

        assert isinstance(terms, tuple)
        assert len(terms) == 2
        inv, eq = terms
        assert isinstance(inv, int)
        assert isinstance(eq, float)
        assert inv > 0
        assert 0.0 < eq <= 100.0

        assert isinstance(speech, str)
        assert len(speech.strip()) > 0
        assert str(inv) in speech or f"{inv:,}" in speech

        # Calling opening stance must not advance round or alter patience/status
        assert default_session.round_number == 0
        assert default_session.patience == settings.DEFAULT_INVESTOR_PATIENCE
        assert default_session.status == GameStatus.IN_PROGRESS

    def test_opening_stance_matches_arbiter_calculation(self, custom_game_state: GameState):
        arbiter = DecisionArbiter()
        session = GameSession(state=custom_game_state, arbiter=arbiter)

        expected_terms = arbiter.calculate_counter_offer(custom_game_state)
        terms, speech = session.get_opening_stance()

        assert terms == expected_terms


class TestRoundStepExecution:
    """Tests executing negotiation rounds and state machine transitions."""

    @pytest.mark.asyncio
    async def test_patience_decrements_by_one_each_round(self):
        session = GameSession.create_session(investor_patience=3)
        assert session.patience == 3

        offer = PlayerOffer(investment=300_000, equity=15.0)
        res1 = await session.step(offer)
        assert session.round_number == 1
        assert session.patience == 2
        assert res1.investor_state.patience == 2

        res2 = await session.step(offer)
        assert session.round_number == 2
        assert session.patience == 1
        assert res2.investor_state.patience == 1

    @pytest.mark.asyncio
    async def test_round_invokes_evaluator_arbiter_and_voice(self):
        eval_result = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.6,
        )
        stub_eval = StubJevEvaluator(eval_result)
        stub_voice = StubVoiceLLM("Мой встречный оффер")
        arbiter = DecisionArbiter()

        session = GameSession(
            evaluator=stub_eval,
            arbiter=arbiter,
            voice_llm=stub_voice,
        )

        offer = PlayerOffer(investment=400_000, equity=15.0)
        result = await session.step(offer)

        assert len(stub_eval.calls) == 1
        assert stub_eval.calls[0].player_offer == offer
        assert len(stub_voice.directives) == 1
        assert stub_voice.directives[0].reaction == "counter"
        assert result.investor_dialogue == "Мой встречный оффер"
        assert result.directive.reaction == "counter"
        assert result.evaluation.reaction == "counter"

    @pytest.mark.asyncio
    async def test_step_delegates_to_arbiter_arbitrate(self, monkeypatch):
        arbitrate_called = False
        original_arbitrate = DecisionArbiter.arbitrate

        def spy_arbitrate(self, state, raw_result):
            nonlocal arbitrate_called
            arbitrate_called = True
            return original_arbitrate(self, state, raw_result)

        monkeypatch.setattr(DecisionArbiter, "arbitrate", spy_arbitrate)

        session = GameSession.create_session()
        await session.step(PlayerOffer(investment=300_000, equity=15.0))
        assert arbitrate_called is True

    @pytest.mark.asyncio
    async def test_round_history_records_offer_decision_speech(self):
        session = GameSession.create_session(investor_patience=3)
        offer = PlayerOffer(investment=350_000, equity=18.0)

        result = await session.step(offer)

        assert len(session.history) == 1
        entry = session.history[0]
        assert "350000" in entry or "350 000" in entry
        assert "18" in entry
        assert result.evaluation.reaction in entry
        assert result.investor_dialogue in entry

    @pytest.mark.asyncio
    async def test_reaction_accept_transitions_to_won(self):
        eval_result = JevEvaluationResult(
            accept_score=5,
            reaction="accept",
            good_deal_confidence=0.9,
        )
        stub_eval = StubJevEvaluator(eval_result)
        session = GameSession(evaluator=stub_eval)

        offer = PlayerOffer(investment=400_000, equity=25.0)
        res = await session.step(offer)

        assert res.game_status == GameStatus.WON
        assert session.status == GameStatus.WON
        assert session.is_active is False
        assert session.is_game_over is True

    @pytest.mark.asyncio
    async def test_reaction_walk_away_transitions_to_lost(self):
        eval_result = JevEvaluationResult(
            accept_score=1,
            reaction="walk_away",
            good_deal_confidence=0.05,
        )
        stub_eval = StubJevEvaluator(eval_result)
        session = GameSession(evaluator=stub_eval)

        offer = PlayerOffer(investment=400_000, equity=5.0)
        res = await session.step(offer)

        assert res.game_status == GameStatus.LOST
        assert session.status == GameStatus.LOST
        assert session.is_active is False
        assert session.is_game_over is True

    @pytest.mark.asyncio
    async def test_patience_exhaustion_transitions_to_lost(self):
        # Starting with patience 1: after 1 non-accept step, patience becomes 0 and triggers walk_away
        eval_result = JevEvaluationResult(
            accept_score=2,
            reaction="counter",
            good_deal_confidence=0.4,
        )
        stub_eval = StubJevEvaluator(eval_result)
        session = GameSession.create_session(investor_patience=1, evaluator=stub_eval)

        offer = PlayerOffer(investment=400_000, equity=15.0)
        res = await session.step(offer)

        assert session.patience == 0
        assert res.evaluation.reaction == "walk_away"
        assert res.game_status == GameStatus.LOST
        assert session.status == GameStatus.LOST
        assert session.is_game_over is True

    @pytest.mark.asyncio
    async def test_reaction_counter_with_patience_remains_in_progress(self):
        eval_result = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.6,
        )
        stub_eval = StubJevEvaluator(eval_result)
        session = GameSession.create_session(investor_patience=4, evaluator=stub_eval)

        offer = PlayerOffer(investment=400_000, equity=15.0)
        res = await session.step(offer)

        assert res.game_status == GameStatus.IN_PROGRESS
        assert session.status == GameStatus.IN_PROGRESS
        assert session.is_active is True
        assert session.is_game_over is False
        assert session.patience == 3

    @pytest.mark.asyncio
    async def test_reaction_reject_with_patience_remains_in_progress(self):
        eval_result = JevEvaluationResult(
            accept_score=2,
            reaction="reject",
            good_deal_confidence=0.3,
        )
        stub_eval = StubJevEvaluator(eval_result)
        session = GameSession.create_session(investor_patience=3, evaluator=stub_eval)

        offer = PlayerOffer(investment=400_000, equity=10.0)
        res = await session.step(offer)

        assert res.game_status == GameStatus.IN_PROGRESS
        assert session.status == GameStatus.IN_PROGRESS
        assert session.is_active is True
        assert session.is_game_over is False
        assert session.patience == 2


class TestTerminalStateGuardrails:
    """Tests preventing steps after game has concluded."""

    @pytest.mark.asyncio
    async def test_cannot_step_when_game_won(self):
        eval_result = JevEvaluationResult(
            accept_score=5,
            reaction="accept",
            good_deal_confidence=0.9,
        )
        session = GameSession(evaluator=StubJevEvaluator(eval_result))
        await session.step(PlayerOffer(investment=200_000, equity=20.0))
        assert session.is_game_over is True

        with pytest.raises(ValueError, match="Cannot step"):
            await session.step(PlayerOffer(investment=200_000, equity=20.0))

    @pytest.mark.asyncio
    async def test_cannot_step_when_game_lost(self):
        eval_result = JevEvaluationResult(
            accept_score=1,
            reaction="walk_away",
            good_deal_confidence=0.1,
        )
        session = GameSession(evaluator=StubJevEvaluator(eval_result))
        await session.step(PlayerOffer(investment=600_000, equity=5.0))
        assert session.is_game_over is True

        with pytest.raises(ValueError, match="Cannot step"):
            await session.step(PlayerOffer(investment=600_000, equity=5.0))

    @pytest.mark.asyncio
    async def test_cannot_step_after_give_up(self):
        session = GameSession()
        session.give_up()
        assert session.is_game_over is True

        with pytest.raises(ValueError, match="Cannot step"):
            await session.step(PlayerOffer(investment=300_000, equity=15.0))


class TestGiveUpSurrender:
    """Tests for give_up surrender action."""

    def test_give_up_transitions_to_quit_or_lost(self):
        session = GameSession()
        assert session.is_active is True

        res = session.give_up()
        assert session.status in (GameStatus.LOST, GameStatus.QUIT)
        assert session.is_active is False
        assert session.is_game_over is True
        assert res.game_status == session.status
        assert len(session.history) == 1

    def test_give_up_with_explicit_status(self):
        session = GameSession()
        res = session.give_up(status=GameStatus.LOST)
        assert session.status == GameStatus.LOST
        assert res.game_status == GameStatus.LOST

    def test_cannot_give_up_when_already_game_over(self):
        session = GameSession()
        session.give_up()

        with pytest.raises(ValueError, match="finished game"):
            session.give_up()

    def test_immediate_give_up_round_indexing(self):
        session = GameSession()
        res = session.give_up()
        assert res.round_number == 1
        assert session.round_number == 1
        assert [r.round_number for r in session.round_results] == [1]

    @pytest.mark.asyncio
    async def test_mid_session_give_up_round_indexing(self):
        eval_result = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.6,
        )
        session = GameSession.create_session(
            investor_patience=5,
            evaluator=StubJevEvaluator(eval_result),
        )

        # Execute 2 valid steps
        r1 = await session.step(PlayerOffer(investment=300_000, equity=15.0))
        assert r1.round_number == 1
        assert session.round_number == 1

        r2 = await session.step(PlayerOffer(investment=350_000, equity=16.0))
        assert r2.round_number == 2
        assert session.round_number == 2

        # Mid-session surrender
        res_quit = session.give_up()

        # Verify round_number matches latest round_result.round_number
        assert res_quit.round_number == 3
        assert session.round_number == 3
        assert session.round_number == session.round_results[-1].round_number

        # Verify all round numbers in session.round_results are unique and strictly monotonic [1, 2, 3]
        round_numbers = [r.round_number for r in session.round_results]
        assert round_numbers == [1, 2, 3]
        assert len(round_numbers) == len(set(round_numbers))
        assert all(round_numbers[i] < round_numbers[i + 1] for i in range(len(round_numbers) - 1))


class TestEndToEndNegotiationFlows:
    """Integration negotiation flow scenarios."""

    @pytest.mark.asyncio
    async def test_successful_negotiation_flow_ending_in_victory(self):
        # Using real MockJevEvaluator and MockVoiceLLM
        session = GameSession.create_session(
            company_valuation=2_000_000,
            investor_budget=700_000,
            investor_max_equity=30.0,
            investor_interest=0.75,
            investor_patience=4,
        )

        terms, speech = session.get_opening_stance()
        assert terms[0] <= 700_000

        # Turn 1: Player offers 400k for 12% (implied valuation ~3.33M vs 2M) -> counter
        r1 = await session.step(PlayerOffer(investment=400_000, equity=12.0))
        assert r1.game_status == GameStatus.IN_PROGRESS
        assert r1.evaluation.reaction == "counter"
        assert session.patience == 3

        # Turn 2: Player offers very attractive deal: 400k for 30% (implied valuation 1.33M <= 0.75 ratio) -> accept
        r2 = await session.step(PlayerOffer(investment=400_000, equity=30.0))
        assert r2.game_status == GameStatus.WON
        assert session.status == GameStatus.WON
        assert session.is_game_over is True
        assert len(session.history) == 2

    @pytest.mark.asyncio
    async def test_failed_negotiation_due_to_patience_exhaustion(self):
        session = GameSession.create_session(
            company_valuation=2_000_000,
            investor_budget=700_000,
            investor_interest=0.70,
            investor_patience=3,
        )

        # 3 rounds of stretched offers (implied valuation ~2.5M, ratio 1.25 -> counter)
        offer = PlayerOffer(investment=500_000, equity=20.0)

        r1 = await session.step(offer)
        assert r1.game_status == GameStatus.IN_PROGRESS
        assert session.patience == 2

        r2 = await session.step(offer)
        assert r2.game_status == GameStatus.IN_PROGRESS
        assert session.patience == 1

        # Round 3: patience was 1, so arbiter patience guardrail forces walk_away, patience -> 0
        r3 = await session.step(offer)
        assert r3.game_status == GameStatus.LOST
        assert r3.evaluation.reaction == "walk_away"
        assert session.status == GameStatus.LOST
        assert session.is_game_over is True
        assert session.patience == 0
        assert session.round_number == 3

    @pytest.mark.asyncio
    async def test_failed_negotiation_due_to_predatory_offer(self):
        session = GameSession.create_session(
            company_valuation=2_000_000,
            investor_budget=700_000,
            investor_patience=4,
        )

        # 10M for 1% (valuation 1 Billion >> 2.5x threshold)
        predatory_offer = PlayerOffer(investment=10_000_000, equity=1.0)
        res = await session.step(predatory_offer)

        assert res.game_status == GameStatus.LOST
        assert res.evaluation.reaction == "walk_away"
        assert session.status == GameStatus.LOST
        assert session.is_game_over is True

    @pytest.mark.asyncio
    async def test_proper_history_accumulation_across_multiple_rounds(self):
        session = GameSession.create_session(investor_patience=4)

        for i in range(1, 3):
            offer = PlayerOffer(investment=300_000 + i * 50_000, equity=15.0)
            res = await session.step(offer)
            assert len(session.history) == i
            assert len(session.round_results) == i
            assert session.round_results[-1] == res
            assert f"Раунд {i}" in session.history[-1]
