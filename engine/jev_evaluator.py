"""Open-JEV Evaluator engine: Mock and HTTP clients for quantitative deal evaluations."""
from abc import ABC, abstractmethod
from typing import Any
import httpx

from config import Settings, settings
from schemas import (
    GameState,
    JevEvaluationResult,
    JevEvaluationSchema,
    JevQueryConfig,
    ReactionType,
)

DEFAULT_JEV_SCHEMA = JevEvaluationSchema(
    accept=JevQueryConfig(
        type="score",
        instructions="Rate readiness to accept the investment offer on a 1-5 scale.",
    ),
    reaction=JevQueryConfig(
        type="choice",
        instructions="Determine the strategic action in response to the offer.",
        criteria=["accept", "counter", "reject", "walk_away"],
    ),
    good_deal=JevQueryConfig(
        type="noul",
        instructions="Evaluate if the deal is financially attractive and advantageous to the investor.",
    ),
)


class BaseJevEvaluator(ABC):
    """Abstract base class / protocol for Open-JEV evaluation engines."""

    @abstractmethod
    async def evaluate(self, state: GameState) -> JevEvaluationResult:
        """Evaluate game state and return quantitative JevEvaluationResult."""
        pass


class MockJevEvaluator(BaseJevEvaluator):
    """Deterministic offline mock evaluator simulating Open-Jev-9B quantitative scoring."""

    async def evaluate(self, state: GameState) -> JevEvaluationResult:
        """Evaluate offer based on valuation ratio, budget ceiling, and investor traits."""
        offer = state.player_offer
        investor = state.investor
        company_val = max(state.company_valuation, 1)
        investment = offer.investment
        budget = max(investor.budget, 1)
        interest = max(0.0, min(1.0, investor.interest))
        patience = investor.patience

        implied_val = offer.implied_valuation
        ratio = implied_val / company_val

        # Budget constraint check
        if investment > budget:
            overbudget_factor = investment / budget
            accept_score = 1 if overbudget_factor > 1.25 else 2
            reaction: ReactionType = (
                "walk_away"
                if (patience <= 1 or overbudget_factor > 1.5)
                else "reject"
            )
            confidence = max(0.01, min(0.35, round(0.20 / overbudget_factor, 2)))
            return JevEvaluationResult(
                accept_score=accept_score,
                reaction=reaction,
                good_deal_confidence=confidence,
            )

        # Within budget: evaluate based on valuation ratio and investor interest
        if ratio <= 0.75:
            # Very generous / discounted offer
            accept_score = 5 if interest >= 0.60 else 4
            reaction = "accept"
            confidence = min(
                0.98,
                max(0.72, round(0.70 + 0.25 * interest + 0.1 * (0.75 - ratio), 2)),
            )
        elif ratio <= 1.15:
            # Fair / negotiable offer
            accept_score = 4 if interest >= 0.80 else 3
            reaction = "accept" if interest >= 0.85 else "counter"
            confidence = min(
                0.75,
                max(0.40, round(0.48 + 0.20 * interest - 0.10 * (ratio - 0.75), 2)),
            )
        elif ratio <= 1.80:
            # Stretched offer
            accept_score = 3 if interest >= 0.80 else 2
            reaction = "counter" if interest >= 0.50 else "reject"
            confidence = min(
                0.45,
                max(0.20, round(0.35 + 0.15 * interest - 0.15 * (ratio - 1.15), 2)),
            )
        elif ratio <= 2.50:
            # Unfavorable / overvalued offer
            accept_score = 2 if interest >= 0.75 else 1
            reaction = "reject"
            confidence = min(
                0.28,
                max(0.05, round(0.22 - 0.10 * (ratio - 1.80), 2)),
            )
        else:
            # Absurd valuation (>= 2.5x)
            accept_score = 1
            reaction = "walk_away" if (patience <= 2 or ratio >= 3.0) else "reject"
            confidence = max(0.02, min(0.20, round(0.25 / ratio, 2)))

        # Safety clamps
        accept_score = max(1, min(5, accept_score))
        confidence = max(0.0, min(1.0, confidence))

        return JevEvaluationResult(
            accept_score=accept_score,
            reaction=reaction,
            good_deal_confidence=confidence,
        )


class HttpJevEvaluator(BaseJevEvaluator):
    """Asynchronous HTTP client calling remote Open-Jev-9B runtime endpoint."""

    def __init__(
        self,
        jev_url: str = "http://localhost:8000/eval",
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
        schema: JevEvaluationSchema | None = None,
    ) -> None:
        self.jev_url = jev_url
        self.timeout = timeout
        self._client = client
        self.schema = schema or DEFAULT_JEV_SCHEMA

    def build_payload(self, state: GameState) -> dict[str, Any]:
        """Build JSON payload conforming to Spec section 3.2 and 4.1."""
        return {
            **state.to_jev_payload(),
            "schema": self.schema.model_dump(exclude_none=True),
        }

    @staticmethod
    def parse_response(data: dict[str, Any]) -> JevEvaluationResult:
        """Parse raw response dict into strictly validated JevEvaluationResult.

        Raises:
            ValueError: If response or extracted target is not a dictionary.
            pydantic.ValidationError: If required fields are missing or values are out of bounds.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response from JEV, got {type(data).__name__}: {data}")

        target = data.get("results") or data.get("evaluation") or data
        if not isinstance(target, dict):
            raise ValueError(f"Expected dict in evaluation results, got {type(target).__name__}: {target}")

        extracted: dict[str, Any] = {}

        # 1. Accept score (1-5) - support accept_score or accept
        if "accept_score" in target:
            val = target["accept_score"]
            extracted["accept_score"] = val.get("score") if isinstance(val, dict) else val
        elif "accept" in target:
            val = target["accept"]
            extracted["accept_score"] = val.get("score") if isinstance(val, dict) else val

        # 2. Reaction choice - support reaction
        if "reaction" in target:
            val = target["reaction"]
            extracted["reaction"] = val.get("choice") if isinstance(val, dict) else val

        # 3. Good deal confidence (0.0 .. 1.0) - support good_deal_confidence or good_deal
        if "good_deal_confidence" in target:
            val = target["good_deal_confidence"]
            if isinstance(val, dict):
                extracted["good_deal_confidence"] = val.get("confidence", val.get("score"))
            else:
                extracted["good_deal_confidence"] = val
        elif "good_deal" in target:
            val = target["good_deal"]
            if isinstance(val, dict):
                extracted["good_deal_confidence"] = val.get("confidence", val.get("score"))
            else:
                extracted["good_deal_confidence"] = val

        return JevEvaluationResult.model_validate(extracted)

    async def evaluate(self, state: GameState) -> JevEvaluationResult:
        """Post game state to Open-Jev endpoint and parse response."""
        payload = self.build_payload(state)
        if self._client is not None:
            response = await self._client.post(
                self.jev_url,
                json=payload,
                timeout=self.timeout,
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.jev_url,
                    json=payload,
                )
        response.raise_for_status()
        data = response.json()
        return self.parse_response(data)


def get_jev_evaluator(settings_obj: Settings | None = None) -> BaseJevEvaluator:
    """Factory creating configured Jev evaluator based on application settings."""
    cfg = settings_obj or settings
    if cfg.MOCK_MODE:
        return MockJevEvaluator()
    return HttpJevEvaluator(
        jev_url=cfg.JEV_URL,
    )


__all__ = [
    "BaseJevEvaluator",
    "MockJevEvaluator",
    "HttpJevEvaluator",
    "DEFAULT_JEV_SCHEMA",
    "get_jev_evaluator",
]
