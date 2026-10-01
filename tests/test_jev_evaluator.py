"""Unit tests for Open-JEV Evaluator (Mock and HTTP client)."""
import httpx
import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock, patch

from config import Settings
from engine.jev_evaluator import (
    BaseJevEvaluator,
    MockJevEvaluator,
    HttpJevEvaluator,
    OllamaJevEvaluator,
    get_jev_evaluator,
    DEFAULT_JEV_SCHEMA,
)
from schemas import GameState, InvestorState, JevEvaluationResult, PlayerOffer


@pytest.fixture
def standard_investor() -> InvestorState:
    return InvestorState(
        budget=700_000,
        max_equity=30.0,
        interest=0.70,
        patience=4,
    )


@pytest.fixture
def make_game_state(standard_investor: InvestorState):
    def _maker(
        investment: int,
        equity: float,
        company_valuation: int = 2_000_000,
        investor: InvestorState | None = None,
    ) -> GameState:
        return GameState(
            company_valuation=company_valuation,
            player_offer=PlayerOffer(investment=investment, equity=equity),
            investor=investor or standard_investor.model_copy(),
            history=[],
        )

    return _maker


# -----------------------------------------------------------------------------
# 1. MockJevEvaluator Tests
# -----------------------------------------------------------------------------


async def test_mock_evaluator_generous_offer(make_game_state):
    """Generous offer: 200k for 20% -> implied 1M < company 2M, high interest.

    Should yield accept_score in 4-5, reaction "accept", high good_deal_confidence (>0.7).
    """
    evaluator = MockJevEvaluator()
    investor = InvestorState(budget=700_000, max_equity=30.0, interest=0.85, patience=4)
    state = make_game_state(investment=200_000, equity=20.0, investor=investor)

    result = await evaluator.evaluate(state)

    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score in (4, 5)
    assert result.reaction == "accept"
    assert result.good_deal_confidence > 0.70


async def test_mock_evaluator_fair_offer(make_game_state):
    """Fair / negotiable offer: 500k for 25% -> implied 2M == company 2M.

    Should yield reaction "counter" or "accept", moderate accept_score (3-4).
    """
    evaluator = MockJevEvaluator()
    state = make_game_state(investment=500_000, equity=25.0)

    result = await evaluator.evaluate(state)

    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score in (3, 4)
    assert result.reaction in ("counter", "accept")
    assert 0.40 <= result.good_deal_confidence <= 0.75


async def test_mock_evaluator_overvalued_offer(make_game_state):
    """Overvalued / tough offer: 700k for 10% -> implied 7M >> company 2M.

    Should yield reaction "reject" or "walk_away", accept_score (1-2), low confidence (<0.3).
    """
    evaluator = MockJevEvaluator()
    state = make_game_state(investment=700_000, equity=10.0)

    result = await evaluator.evaluate(state)

    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score in (1, 2)
    assert result.reaction in ("reject", "walk_away")
    assert result.good_deal_confidence < 0.30


async def test_mock_evaluator_overbudget_offer(make_game_state):
    """Overbudget offer: investment (800k) > investor budget (700k).

    Should yield low score (1-2), reaction "reject" or "walk_away", low confidence.
    """
    evaluator = MockJevEvaluator()
    state = make_game_state(investment=800_000, equity=35.0)

    result = await evaluator.evaluate(state)

    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score in (1, 2)
    assert result.reaction in ("reject", "walk_away")
    assert result.good_deal_confidence < 0.40


async def test_mock_evaluator_severely_overbudget_offer(make_game_state):
    """Severely overbudget offer: 1.5M when budget is 700k -> lowest score 1."""
    evaluator = MockJevEvaluator()
    state = make_game_state(investment=1_500_000, equity=50.0)

    result = await evaluator.evaluate(state)

    assert result.accept_score == 1
    assert result.reaction in ("reject", "walk_away")
    assert result.good_deal_confidence <= 0.20


async def test_mock_evaluator_interest_influence(make_game_state):
    """Higher investor interest should improve evaluation metrics for the same offer."""
    evaluator = MockJevEvaluator()
    low_investor = InvestorState(budget=700_000, max_equity=30.0, interest=0.20, patience=4)
    high_investor = InvestorState(budget=700_000, max_equity=30.0, interest=0.90, patience=4)

    state_low = make_game_state(investment=400_000, equity=20.0, investor=low_investor)
    state_high = make_game_state(investment=400_000, equity=20.0, investor=high_investor)

    res_low = await evaluator.evaluate(state_low)
    res_high = await evaluator.evaluate(state_high)

    assert res_high.accept_score >= res_low.accept_score
    assert res_high.good_deal_confidence >= res_low.good_deal_confidence


async def test_mock_evaluator_matrix_always_valid(make_game_state):
    """Across various combinations of valuation, investment, and equity, output is always valid."""
    evaluator = MockJevEvaluator()
    investments = [50_000, 200_000, 500_000, 700_000, 1_000_000]
    equities = [5.0, 15.0, 25.0, 50.0]

    for inv in investments:
        for eq in equities:
            state = make_game_state(investment=inv, equity=eq)
            res = await evaluator.evaluate(state)
            assert isinstance(res, JevEvaluationResult)
            assert 1 <= res.accept_score <= 5
            assert res.reaction in ("accept", "counter", "reject", "walk_away")
            assert 0.0 <= res.good_deal_confidence <= 1.0


# -----------------------------------------------------------------------------
# 2. HttpJevEvaluator Tests
# -----------------------------------------------------------------------------


async def test_http_evaluator_successful_direct_response(make_game_state):
    """Successful REST call with direct JevEvaluationResult schema."""
    mock_response_data = {
        "accept_score": 4,
        "reaction": "accept",
        "good_deal_confidence": 0.85,
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_resp.raise_for_status = AsyncMock() if hasattr(mock_resp.raise_for_status, "__await__") else lambda: None
    mock_client.post.return_value = mock_resp

    evaluator = HttpJevEvaluator(
        jev_url="http://test-jev:8000/eval",
        client=mock_client,
    )
    state = make_game_state(investment=300_000, equity=20.0)

    result = await evaluator.evaluate(state)

    assert result.accept_score == 4
    assert result.reaction == "accept"
    assert result.good_deal_confidence == 0.85

    # Verify call parameters
    mock_client.post.assert_awaited_once()
    called_url, called_kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert called_url == "http://test-jev:8000/eval"
    json_payload = called_kwargs["json"]
    assert json_payload["valuation"] == 2_000_000
    assert json_payload["player_offer"]["investment"] == 300_000
    assert "schema" in json_payload
    assert "accept" in json_payload["schema"]


async def test_http_evaluator_successful_nested_response(make_game_state):
    """Successful REST call with nested axes results."""
    mock_response_data = {
        "results": {
            "accept": {"score": 3},
            "reaction": {"choice": "counter"},
            "good_deal": {"confidence": 0.62},
        }
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    evaluator = HttpJevEvaluator(
        jev_url="http://test-jev:8000/eval",
        client=mock_client,
    )
    state = make_game_state(investment=500_000, equity=25.0)

    result = await evaluator.evaluate(state)

    assert result.accept_score == 3
    assert result.reaction == "counter"
    assert result.good_deal_confidence == 0.62


# --- parse_response Unit Tests ---


def test_parse_response_malformed_empty_dict_raises():
    """Empty dictionary payload fails strict validation without inventing defaults."""
    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({})


def test_parse_response_missing_fields_raises():
    """Payloads with missing required fields raise ValidationError."""
    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept": None})

    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept": 4, "reaction": "accept"})  # missing good_deal

    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"reaction": "accept", "good_deal": 0.8})  # missing accept


def test_parse_response_invalid_reaction_raises():
    """Invalid reaction value fails Pydantic Literal validation."""
    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept": 4, "reaction": "invalid", "good_deal": 0.8})


def test_parse_response_out_of_bounds_score_raises():
    """Accept score outside [1, 5] raises ValidationError instead of silent clamping."""
    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept_score": 6, "reaction": "accept", "good_deal_confidence": 0.8})

    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept_score": 0, "reaction": "accept", "good_deal_confidence": 0.8})


def test_parse_response_out_of_bounds_confidence_raises():
    """Confidence outside [0.0, 1.0] raises ValidationError instead of silent clamping."""
    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept_score": 4, "reaction": "accept", "good_deal_confidence": 1.5})

    with pytest.raises((ValidationError, ValueError)):
        HttpJevEvaluator.parse_response({"accept_score": 4, "reaction": "accept", "good_deal_confidence": -0.1})


def test_parse_response_valid_flat_format():
    """Valid flat format with accept_score, reaction, good_deal_confidence."""
    result = HttpJevEvaluator.parse_response({
        "accept_score": 4,
        "reaction": "accept",
        "good_deal_confidence": 0.8,
    })
    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score == 4
    assert result.reaction == "accept"
    assert result.good_deal_confidence == 0.8


def test_parse_response_valid_alias_format():
    """Valid alias format with accept, reaction, good_deal."""
    result = HttpJevEvaluator.parse_response({
        "accept": 4,
        "reaction": "accept",
        "good_deal": 0.8,
    })
    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score == 4
    assert result.reaction == "accept"
    assert result.good_deal_confidence == 0.8


def test_parse_response_valid_nested_format():
    """Valid nested format under 'results' with score, choice, and confidence."""
    result = HttpJevEvaluator.parse_response({
        "results": {
            "accept": {"score": 3},
            "reaction": {"choice": "counter"},
            "good_deal": {"confidence": 0.62},
        }
    })
    assert isinstance(result, JevEvaluationResult)
    assert result.accept_score == 3
    assert result.reaction == "counter"
    assert result.good_deal_confidence == 0.62


def test_parse_response_non_dict_raises():
    """Non-dict input or non-dict target raises ValueError."""
    with pytest.raises(ValueError):
        HttpJevEvaluator.parse_response("not a dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HttpJevEvaluator.parse_response({"results": "invalid string target"})


# --- evaluate() Unit Tests ---


async def test_http_evaluator_timeout_raises(make_game_state):
    """Network timeout on evaluate raises httpx.HTTPError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")

    evaluator = HttpJevEvaluator(
        jev_url="http://test-jev:8000/eval",
        client=mock_client,
    )
    state = make_game_state(investment=500_000, equity=20.0)

    with pytest.raises(httpx.HTTPError):
        await evaluator.evaluate(state)


async def test_http_evaluator_http_500_raises(make_game_state):
    """HTTP 500 error on evaluate raises httpx.HTTPStatusError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    req = httpx.Request("POST", "http://test-jev:8000/eval")
    resp = httpx.Response(status_code=500, request=req)
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "Server Error", request=req, response=resp
    )

    evaluator = HttpJevEvaluator(
        jev_url="http://test-jev:8000/eval",
        client=mock_client,
    )
    state = make_game_state(investment=500_000, equity=25.0)

    with pytest.raises(httpx.HTTPStatusError):
        await evaluator.evaluate(state)


async def test_http_evaluator_malformed_response_raises(make_game_state):
    """Malformed response from endpoint raises ValidationError / ValueError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {}
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    evaluator = HttpJevEvaluator(
        jev_url="http://test-jev:8000/eval",
        client=mock_client,
    )
    state = make_game_state(investment=500_000, equity=25.0)

    with pytest.raises((ValidationError, ValueError)):
        await evaluator.evaluate(state)


async def test_http_evaluator_default_client_lifecycle(make_game_state):
    """When client is not injected, HttpJevEvaluator uses httpx.AsyncClient context manager."""
    state = make_game_state(investment=300_000, equity=20.0)
    mock_response = httpx.Response(
        status_code=200,
        json={"accept_score": 4, "reaction": "accept", "good_deal_confidence": 0.8},
        request=httpx.Request("POST", "http://test-jev:8000/eval"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        evaluator = HttpJevEvaluator(jev_url="http://test-jev:8000/eval")
        result = await evaluator.evaluate(state)
        assert result.reaction == "accept"
        mock_post.assert_awaited_once()


def test_http_evaluator_decoupled_from_mock():
    """HttpJevEvaluator is decoupled from MockJevEvaluator and has no fallback attributes."""
    evaluator = HttpJevEvaluator()
    assert not hasattr(evaluator, "_mock_evaluator")
    assert not hasattr(evaluator, "fallback_to_mock")


# -----------------------------------------------------------------------------
# 3. Factory Function & Base Evaluator Tests
# -----------------------------------------------------------------------------


def test_factory_returns_mock_when_mock_mode_true():
    """get_jev_evaluator returns MockJevEvaluator when MOCK_MODE=True."""
    settings = Settings(MOCK_MODE=True)
    evaluator = get_jev_evaluator(settings)
    assert isinstance(evaluator, MockJevEvaluator)
    assert isinstance(evaluator, BaseJevEvaluator)


def test_factory_returns_http_when_backend_http():
    """get_jev_evaluator returns HttpJevEvaluator when JEV_BACKEND='http' and MOCK_MODE=False."""
    settings = Settings(MOCK_MODE=False, JEV_BACKEND="http", JEV_URL="http://remote-jev:8000/eval")
    evaluator = get_jev_evaluator(settings)
    assert isinstance(evaluator, HttpJevEvaluator)
    assert isinstance(evaluator, BaseJevEvaluator)
    assert evaluator.jev_url == "http://remote-jev:8000/eval"


def test_factory_returns_ollama_when_backend_ollama():
    """get_jev_evaluator returns OllamaJevEvaluator when JEV_BACKEND='ollama' and MOCK_MODE=False."""
    settings = Settings(MOCK_MODE=False, JEV_BACKEND="ollama")
    evaluator = get_jev_evaluator(settings)
    assert isinstance(evaluator, OllamaJevEvaluator)
    assert isinstance(evaluator, BaseJevEvaluator)


def test_factory_uses_default_settings_when_none_provided():
    """get_jev_evaluator uses global settings by default."""
    evaluator = get_jev_evaluator()
    assert isinstance(evaluator, BaseJevEvaluator)


async def test_ollama_evaluator_evaluates_correctly(make_game_state):
    """OllamaJevEvaluator sends chat request and strictly parses JSON response."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {
            "content": '{"accept_score": 4, "reaction": "accept", "good_deal_confidence": 0.82}'
        }
    }
    mock_resp.raise_for_status = lambda: None
    mock_client.post.return_value = mock_resp

    evaluator = OllamaJevEvaluator(client=mock_client)
    state = make_game_state(investment=300_000, equity=20.0)

    result = await evaluator.evaluate(state)
    assert result.accept_score == 4
    assert result.reaction == "accept"
    assert result.good_deal_confidence == 0.82
