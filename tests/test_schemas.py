"""Unit tests for Pydantic domain models, serialization, and configuration."""
import pytest
from pydantic import ValidationError
from schemas import (
    PlayerOffer,
    InvestorState,
    GameState,
    JevQueryConfig,
    JevEvaluationSchema,
    JevEvaluationResult,
    VoiceDirective,
    GameStatus,
    RoundResult,
)
from config import Settings, settings


class TestPlayerOffer:
    def test_valid_player_offer(self):
        offer = PlayerOffer(investment=500_000, equity=20.0)
        assert offer.investment == 500_000
        assert offer.equity == 20.0

    def test_implied_valuation_calculation(self):
        # 500,000 / (25 / 100) = 2,000,000
        offer = PlayerOffer(investment=500_000, equity=25.0)
        assert offer.implied_valuation == pytest.approx(2_000_000.0)

        # 100,000 / (10 / 100) = 1,000,000
        offer2 = PlayerOffer(investment=100_000, equity=10.0)
        assert offer2.implied_valuation == pytest.approx(1_000_000.0)

    @pytest.mark.parametrize("invalid_investment", [0, -1, -500_000])
    def test_invalid_investment(self, invalid_investment):
        with pytest.raises(ValidationError):
            PlayerOffer(investment=invalid_investment, equity=20.0)

    @pytest.mark.parametrize("invalid_equity", [0.0, -5.0, 100.1, 150.0])
    def test_invalid_equity(self, invalid_equity):
        with pytest.raises(ValidationError):
            PlayerOffer(investment=500_000, equity=invalid_equity)


class TestInvestorState:
    def test_valid_investor_state(self):
        state = InvestorState(
            budget=700_000,
            max_equity=30.0,
            interest=0.70,
            patience=4,
        )
        assert state.budget == 700_000
        assert state.max_equity == 30.0
        assert state.interest == 0.70
        assert state.patience == 4

    @pytest.mark.parametrize("invalid_budget", [0, -1, -100_000])
    def test_invalid_budget(self, invalid_budget):
        with pytest.raises(ValidationError):
            InvestorState(budget=invalid_budget, max_equity=30.0, interest=0.5, patience=3)

    @pytest.mark.parametrize("invalid_max_equity", [0.0, -1.0, 100.1, 105.0])
    def test_invalid_max_equity(self, invalid_max_equity):
        with pytest.raises(ValidationError):
            InvestorState(budget=500_000, max_equity=invalid_max_equity, interest=0.5, patience=3)

    @pytest.mark.parametrize("invalid_interest", [-0.1, 1.1, -1.0, 2.0])
    def test_invalid_interest(self, invalid_interest):
        with pytest.raises(ValidationError):
            InvestorState(budget=500_000, max_equity=30.0, interest=invalid_interest, patience=3)

    @pytest.mark.parametrize("invalid_patience", [-1, -5])
    def test_invalid_patience(self, invalid_patience):
        with pytest.raises(ValidationError):
            InvestorState(budget=500_000, max_equity=30.0, interest=0.5, patience=invalid_patience)

    def test_boundary_patience_zero(self):
        state = InvestorState(budget=500_000, max_equity=30.0, interest=0.5, patience=0)
        assert state.patience == 0


class TestGameState:
    def test_game_state_to_jev_payload(self, sample_game_state: GameState):
        payload = sample_game_state.to_jev_payload()
        assert payload == {
            "valuation": 2_000_000,
            "player_offer": {"investment": 500_000, "equity": 15.0},
            "investor_profile": {
                "budget": 700_000,
                "max_equity": 30.0,
                "interest": 0.72,
                "patience": 3,
            },
            "history": ["Investor offered €500k for 30%", "Player rejected"],
        }

    def test_game_state_default_history(self, sample_player_offer, sample_investor_state):
        gs = GameState(
            company_valuation=2_000_000,
            player_offer=sample_player_offer,
            investor=sample_investor_state,
        )
        assert gs.history == []
        payload = gs.to_jev_payload()
        assert payload["history"] == []

    def test_invalid_company_valuation(self, sample_player_offer, sample_investor_state):
        with pytest.raises(ValidationError):
            GameState(
                company_valuation=0,
                player_offer=sample_player_offer,
                investor=sample_investor_state,
            )


class TestJevSchemas:
    def test_valid_jev_query_config(self):
        config = JevQueryConfig(
            type="score",
            instructions="Evaluate accept probability",
            criteria=["1: Definitely reject", "5: Definitely accept"],
        )
        assert config.type == "score"

    def test_invalid_jev_query_config_type(self):
        with pytest.raises(ValidationError):
            JevQueryConfig(
                type="unsupported",  # type: ignore
                instructions="Test",
            )

    def test_valid_jev_evaluation_schema(self):
        schema = JevEvaluationSchema(
            accept=JevQueryConfig(type="score", instructions="Score 1-5"),
            reaction=JevQueryConfig(type="choice", instructions="Choose action"),
            good_deal=JevQueryConfig(type="noul", instructions="Is it good?"),
        )
        assert schema.accept.type == "score"
        assert schema.reaction.type == "choice"
        assert schema.good_deal.type == "noul"

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_valid_jev_evaluation_result_scores(self, score):
        res = JevEvaluationResult(
            accept_score=score,
            reaction="accept",
            good_deal_confidence=0.8,
        )
        assert res.accept_score == score

    @pytest.mark.parametrize("invalid_score", [0, 6, -1, 10])
    def test_invalid_jev_evaluation_result_scores(self, invalid_score):
        with pytest.raises(ValidationError):
            JevEvaluationResult(
                accept_score=invalid_score,
                reaction="accept",
                good_deal_confidence=0.8,
            )

    @pytest.mark.parametrize("reaction", ["accept", "counter", "reject", "walk_away"])
    def test_valid_reactions(self, reaction):
        res = JevEvaluationResult(
            accept_score=3,
            reaction=reaction,
            good_deal_confidence=0.5,
        )
        assert res.reaction == reaction

    def test_invalid_reaction(self):
        with pytest.raises(ValidationError):
            JevEvaluationResult(
                accept_score=3,
                reaction="invalid_reaction",  # type: ignore
                good_deal_confidence=0.5,
            )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_valid_good_deal_confidence(self, confidence):
        res = JevEvaluationResult(
            accept_score=3,
            reaction="counter",
            good_deal_confidence=confidence,
        )
        assert res.good_deal_confidence == confidence

    @pytest.mark.parametrize("invalid_conf", [-0.01, 1.01, -1.0, 2.0])
    def test_invalid_good_deal_confidence(self, invalid_conf):
        with pytest.raises(ValidationError):
            JevEvaluationResult(
                accept_score=3,
                reaction="counter",
                good_deal_confidence=invalid_conf,
            )


class TestVoiceDirective:
    def test_valid_voice_directive(self):
        directive = VoiceDirective(
            reaction="counter",
            accept_score=3,
            good_deal_status="Сомнительно",
            counter_offer_investment=500_000,
            counter_offer_equity=25.0,
        )
        assert directive.reaction == "counter"
        assert directive.counter_offer_investment == 500_000
        assert directive.counter_offer_equity == 25.0

    def test_voice_directive_optional_counters(self):
        directive = VoiceDirective(
            reaction="reject",
            accept_score=1,
            good_deal_status="Невыгодно",
        )
        assert directive.counter_offer_investment is None
        assert directive.counter_offer_equity is None


class TestGameStatusAndRoundResult:
    def test_game_status_enum_values(self):
        assert GameStatus.IN_PROGRESS.value == "in_progress"
        assert GameStatus.WON.value == "won"
        assert GameStatus.LOST.value == "lost"

    def test_round_result_container(
        self,
        sample_player_offer,
        sample_jev_evaluation_result,
        sample_voice_directive,
        sample_investor_state,
    ):
        result = RoundResult(
            round_number=1,
            player_offer=sample_player_offer,
            evaluation=sample_jev_evaluation_result,
            directive=sample_voice_directive,
            investor_dialogue="Предложение интересное, но давай 25%.",
            game_status=GameStatus.IN_PROGRESS,
            investor_state=sample_investor_state,
        )
        assert result.round_number == 1
        assert result.game_status == GameStatus.IN_PROGRESS
        assert result.player_offer.investment == 500_000


class TestConfigSettings:
    def test_settings_defaults(self):
        test_settings = Settings()
        assert test_settings.LLM_URL == "http://localhost:11434/v1"
        assert test_settings.LLM_API_KEY == "dummy"
        assert test_settings.LLM_MODEL == "llama3:latest"
        assert test_settings.JEV_URL == "http://localhost:8000/eval"
        assert test_settings.JEV_MODEL_PATH == "ZefanCai/Open-Jev-9B"
        assert test_settings.MOCK_MODE is True
        assert test_settings.DEFAULT_COMPANY_VALUATION == 2_000_000
        assert test_settings.DEFAULT_INVESTOR_BUDGET == 700_000
        assert test_settings.DEFAULT_INVESTOR_MAX_EQUITY == 30.0
        assert test_settings.DEFAULT_INVESTOR_INTEREST == 0.70
        assert test_settings.DEFAULT_INVESTOR_PATIENCE == 4

    def test_singleton_settings_instance(self):
        assert settings is not None
        assert settings.DEFAULT_COMPANY_VALUATION == 2_000_000
