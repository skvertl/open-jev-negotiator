"""Unit tests for Decision Arbiter and Guardrails."""
import pytest
from schemas import (
    GameState,
    InvestorState,
    JevEvaluationResult,
    PlayerOffer,
    VoiceDirective,
)
from engine.arbiter import DecisionArbiter


@pytest.fixture
def arbiter() -> DecisionArbiter:
    return DecisionArbiter()


@pytest.fixture
def base_investor_state() -> InvestorState:
    return InvestorState(
        budget=700_000,
        max_equity=30.0,
        interest=0.70,
        patience=4,
    )


@pytest.fixture
def base_game_state(base_investor_state: InvestorState) -> GameState:
    return GameState(
        company_valuation=2_000_000,
        player_offer=PlayerOffer(investment=500_000, equity=20.0),
        investor=base_investor_state,
        history=[],
    )


class TestPatienceGuardrail:
    """Tests ensuring patience exhaustion strictly forces walk_away unless already accepted."""

    @pytest.mark.parametrize("raw_reaction", ["counter", "reject", "walk_away"])
    def test_patience_one_forces_walk_away_on_non_accept(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
        raw_reaction: str,
    ):
        base_game_state.investor.patience = 1
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction=raw_reaction,  # type: ignore
            good_deal_confidence=0.5,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "walk_away"

    def test_patience_zero_forces_walk_away(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.investor.patience = 0
        raw_eval = JevEvaluationResult(
            accept_score=2,
            reaction="counter",
            good_deal_confidence=0.4,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "walk_away"

    def test_patience_one_preserves_accept(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.investor.patience = 1
        raw_eval = JevEvaluationResult(
            accept_score=5,
            reaction="accept",
            good_deal_confidence=0.9,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "accept"

    @pytest.mark.parametrize("patience", [2, 3, 4, 5])
    def test_patience_greater_than_one_does_not_override_counter(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
        patience: int,
    ):
        base_game_state.investor.patience = patience
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.6,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "counter"


class TestValuationAndBudgetGuardrails:
    """Tests ensuring valuation breaches and budget limits force walk_away or reject."""

    def test_valuation_breach_forces_walk_away(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # 500k for 5% -> implied valuation €10,000,000 (5.0x of €2,000,000 >= 2.5x)
        base_game_state.player_offer = PlayerOffer(investment=500_000, equity=5.0)
        raw_eval = JevEvaluationResult(
            accept_score=4,
            reaction="accept",
            good_deal_confidence=0.8,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "walk_away"

    def test_exact_valuation_threshold_boundary_forces_walk_away(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # 500k for 10% -> implied valuation €5,000,000 (exactly 2.5x of €2,000,000)
        base_game_state.player_offer = PlayerOffer(investment=500_000, equity=10.0)
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.5,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "walk_away"

    def test_valuation_below_threshold_does_not_force_walk_away(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # 490k for 10% -> implied valuation €4,900,000 (2.45x of €2,000,000 < 2.5x)
        base_game_state.player_offer = PlayerOffer(investment=490_000, equity=10.0)
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.5,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "counter"

    def test_investment_exceeding_budget_forces_reject_when_patience_normal(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # 800k > budget 700k, implied valuation = 800k / 0.5 = 1.6M (< 2.5x 2M)
        base_game_state.player_offer = PlayerOffer(investment=800_000, equity=50.0)
        raw_eval = JevEvaluationResult(
            accept_score=4,
            reaction="accept",
            good_deal_confidence=0.8,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "reject"

    def test_investment_exceeding_budget_forces_walk_away_when_patience_low(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.player_offer = PlayerOffer(investment=800_000, equity=50.0)
        base_game_state.investor.patience = 1
        raw_eval = JevEvaluationResult(
            accept_score=4,
            reaction="accept",
            good_deal_confidence=0.8,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "walk_away"

    def test_investment_within_budget_is_preserved(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.player_offer = PlayerOffer(investment=700_000, equity=30.0)
        raw_eval = JevEvaluationResult(
            accept_score=4,
            reaction="accept",
            good_deal_confidence=0.8,
        )

        guarded = arbiter.check_guardrails(base_game_state, raw_eval)
        assert guarded.reaction == "accept"


class TestCounterOfferCalculation:
    """Tests for rational counter-offer terms within budget and equity bounds."""

    def test_counter_offer_respects_budget_ceiling(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # Player asks for 800k, budget is 700k
        base_game_state.player_offer = PlayerOffer(investment=800_000, equity=25.0)
        inv, eq = arbiter.calculate_counter_offer(base_game_state)
        assert inv <= base_game_state.investor.budget
        assert inv == 700_000

    def test_counter_offer_preserves_investment_when_within_budget(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.player_offer = PlayerOffer(investment=450_000, equity=15.0)
        inv, _ = arbiter.calculate_counter_offer(base_game_state)
        assert inv == 450_000

    def test_counter_offer_clamps_equity_to_max_equity(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # Low interest (0.3) -> formula equity would exceed max_equity (30.0)
        base_game_state.investor.interest = 0.30
        base_game_state.player_offer = PlayerOffer(investment=500_000, equity=10.0)
        _, eq = arbiter.calculate_counter_offer(base_game_state)
        assert eq <= base_game_state.investor.max_equity
        assert eq == 30.0

    def test_counter_offer_demands_at_least_25_percent_bump_over_player_offer(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # Offer 10% equity with high interest (1.0) and high valuation
        # formula_equity = 100k / (2M * 1.0) * 100 = 5.0%
        # But 10% * 1.25 = 12.5%
        base_game_state.investor.interest = 1.0
        base_game_state.player_offer = PlayerOffer(investment=100_000, equity=10.0)
        _, eq = arbiter.calculate_counter_offer(base_game_state)
        assert eq >= 12.5
        assert eq == pytest.approx(12.5, abs=0.1)

    def test_counter_offer_equity_rounded_to_one_decimal(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.player_offer = PlayerOffer(investment=333_333, equity=11.1)
        _, eq = arbiter.calculate_counter_offer(base_game_state)
        assert eq == round(eq, 1)

    def test_counter_offer_equity_strictly_positive(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.player_offer = PlayerOffer(investment=1_000, equity=0.1)
        _, eq = arbiter.calculate_counter_offer(base_game_state)
        assert eq > 0.0


class TestBuildVoiceDirective:
    """Tests for VoiceDirective payload construction."""

    def test_good_deal_status_high_confidence(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        evaluation = JevEvaluationResult(
            accept_score=4,
            reaction="accept",
            good_deal_confidence=0.85,
        )
        directive = arbiter.build_voice_directive(base_game_state, evaluation)
        assert directive.reaction == "accept"
        assert directive.accept_score == 4
        assert directive.good_deal_status == "выгодная сделка"
        assert directive.counter_offer_investment is None
        assert directive.counter_offer_equity is None

    def test_good_deal_status_low_confidence(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        evaluation = JevEvaluationResult(
            accept_score=1,
            reaction="reject",
            good_deal_confidence=0.20,
        )
        directive = arbiter.build_voice_directive(base_game_state, evaluation)
        assert directive.reaction == "reject"
        assert directive.accept_score == 1
        assert directive.good_deal_status == "невыгодная сделка"
        assert directive.counter_offer_investment is None
        assert directive.counter_offer_equity is None

    def test_good_deal_status_threshold_boundary(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        evaluation = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.50,
        )
        directive = arbiter.build_voice_directive(
            base_game_state, evaluation, counter_offer=(500_000, 25.0)
        )
        assert directive.good_deal_status == "выгодная сделка"
        assert directive.counter_offer_investment == 500_000
        assert directive.counter_offer_equity == 25.0

    def test_counter_reaction_clears_terms_when_reaction_not_counter(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        evaluation = JevEvaluationResult(
            accept_score=2,
            reaction="reject",
            good_deal_confidence=0.30,
        )
        # Even if counter-offer tuple is provided, reject directive must not carry it
        directive = arbiter.build_voice_directive(
            base_game_state, evaluation, counter_offer=(500_000, 25.0)
        )
        assert directive.reaction == "reject"
        assert directive.counter_offer_investment is None
        assert directive.counter_offer_equity is None

    def test_counter_reaction_auto_computes_counter_offer_when_none(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        evaluation = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.55,
        )
        directive = arbiter.build_voice_directive(base_game_state, evaluation, counter_offer=None)
        assert directive.reaction == "counter"
        assert directive.counter_offer_investment is not None
        assert directive.counter_offer_equity is not None
        assert directive.counter_offer_investment <= base_game_state.investor.budget


class TestArbitrateOrchestration:
    """Integration tests for arbiter.arbitrate orchestrating guardrails and directive."""

    def test_arbitrate_accept_flow(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        raw_eval = JevEvaluationResult(
            accept_score=5,
            reaction="accept",
            good_deal_confidence=0.90,
        )
        evaluation, directive = arbiter.arbitrate(base_game_state, raw_eval)
        assert evaluation.reaction == "accept"
        assert directive.reaction == "accept"
        assert directive.accept_score == 5
        assert directive.good_deal_status == "выгодная сделка"
        assert directive.counter_offer_investment is None

    def test_arbitrate_counter_flow_produces_directive_with_terms(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.60,
        )
        evaluation, directive = arbiter.arbitrate(base_game_state, raw_eval)
        assert evaluation.reaction == "counter"
        assert directive.reaction == "counter"
        assert directive.counter_offer_investment is not None
        assert directive.counter_offer_equity is not None
        assert directive.counter_offer_equity <= base_game_state.investor.max_equity

    def test_arbitrate_low_patience_overrides_to_walk_away(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.investor.patience = 1
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.60,
        )
        evaluation, directive = arbiter.arbitrate(base_game_state, raw_eval)
        assert evaluation.reaction == "walk_away"
        assert directive.reaction == "walk_away"
        assert directive.counter_offer_investment is None
        assert directive.counter_offer_equity is None

    def test_arbitrate_predatory_offer_overrides_to_walk_away(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # 600k for 5% -> implied 12M >= 2.5x 2M
        base_game_state.player_offer = PlayerOffer(investment=600_000, equity=5.0)
        raw_eval = JevEvaluationResult(
            accept_score=4,
            reaction="accept",
            good_deal_confidence=0.85,
        )
        evaluation, directive = arbiter.arbitrate(base_game_state, raw_eval)
        assert evaluation.reaction == "walk_away"
        assert directive.reaction == "walk_away"
        assert directive.counter_offer_investment is None

    def test_arbitrate_agreement_on_counter_offer_terms_forces_accept(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        base_game_state.last_counter_offer = (500_000, 30.0)
        base_game_state.player_offer = PlayerOffer(investment=500_000, equity=30.0)
        raw_eval = JevEvaluationResult(
            accept_score=2,
            reaction="reject",
            good_deal_confidence=0.25,
        )
        evaluation, directive = arbiter.arbitrate(base_game_state, raw_eval)
        assert evaluation.reaction == "accept"
        assert evaluation.accept_score == 5
        assert evaluation.good_deal_confidence >= 0.95
        assert directive.reaction == "accept"

    def test_arbitrate_better_offer_than_counter_forces_accept(
        self,
        arbiter: DecisionArbiter,
        base_game_state: GameState,
    ):
        # 400k for 30% is even more advantageous than requested 500k for 30%
        base_game_state.last_counter_offer = (500_000, 30.0)
        base_game_state.player_offer = PlayerOffer(investment=400_000, equity=30.0)
        raw_eval = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=0.30,
        )
        evaluation, directive = arbiter.arbitrate(base_game_state, raw_eval)
        assert evaluation.reaction == "accept"
        assert evaluation.accept_score == 5
        assert directive.reaction == "accept"
