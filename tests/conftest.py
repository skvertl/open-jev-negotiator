"""Pytest configuration and shared test fixtures."""
import pytest
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


@pytest.fixture
def sample_player_offer() -> PlayerOffer:
    return PlayerOffer(investment=500_000, equity=15.0)


@pytest.fixture
def sample_investor_state() -> InvestorState:
    return InvestorState(
        budget=700_000,
        max_equity=30.0,
        interest=0.72,
        patience=3,
    )


@pytest.fixture
def sample_game_state(
    sample_player_offer: PlayerOffer, sample_investor_state: InvestorState
) -> GameState:
    return GameState(
        company_valuation=2_000_000,
        player_offer=sample_player_offer,
        investor=sample_investor_state,
        history=["Investor offered €500k for 30%", "Player rejected"],
    )


@pytest.fixture
def sample_jev_evaluation_result() -> JevEvaluationResult:
    return JevEvaluationResult(
        accept_score=3,
        reaction="counter",
        good_deal_confidence=0.65,
    )


@pytest.fixture
def sample_voice_directive() -> VoiceDirective:
    return VoiceDirective(
        reaction="counter",
        accept_score=3,
        good_deal_status="Сомнительно",
        counter_offer_investment=500_000,
        counter_offer_equity=25.0,
    )


@pytest.fixture(autouse=True)
def default_mock_settings(monkeypatch):
    """Ensure tests run in mock mode by default for speed and isolation."""
    monkeypatch.setattr("config.settings.MOCK_MODE", True)
    monkeypatch.setattr("config.settings.JEV_BACKEND", "mock")
