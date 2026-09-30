"""Core domain data contracts and schemas for Open-JEV Negotiator."""
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

ReactionType = Literal["accept", "counter", "reject", "walk_away"]


class PlayerOffer(BaseModel):
    """Founder's investment offer for a percentage of equity."""

    investment: int = Field(gt=0, description="Запрашиваемая сумма инвестиций в €")
    equity: float = Field(gt=0, le=100, description="Предлагаемая доля компании в %")

    @property
    def implied_valuation(self) -> float:
        """Calculate the implied company valuation: investment / (equity / 100)."""
        return self.investment / (self.equity / 100.0)


class InvestorState(BaseModel):
    """Current state and traits of the NPC investor."""

    budget: int = Field(gt=0, description="Максимальный свободный капитал инвестора")
    max_equity: float = Field(gt=0, le=100, description="Желаемая планка доли")
    interest: float = Field(ge=0.0, le=1.0, description="Коэффициент заинтересованности 0..1")
    patience: int = Field(ge=0, description="Остаток раундов терпения до срыва сделки")


class GameState(BaseModel):
    """Complete game snapshot at the current turn."""

    company_valuation: int = Field(gt=0, description="Текущая рыночная оценка компании")
    player_offer: PlayerOffer
    investor: InvestorState
    history: list[str] = Field(default_factory=list)

    def to_jev_payload(self) -> dict[str, Any]:
        """Serialize state to standard JSON payload for Open-JEV engine matching spec 4.1."""
        return {
            "valuation": self.company_valuation,
            "player_offer": {
                "investment": self.player_offer.investment,
                "equity": self.player_offer.equity,
            },
            "investor_profile": {
                "budget": self.investor.budget,
                "max_equity": self.investor.max_equity,
                "interest": self.investor.interest,
                "patience": self.investor.patience,
            },
            "history": list(self.history),
        }


class JevQueryConfig(BaseModel):
    """Configuration for an individual Jev query axis."""

    type: Literal["score", "choice", "noul"]
    instructions: str
    criteria: list[str] | dict[str, str] | None = None


class JevEvaluationSchema(BaseModel):
    """Composite schema containing all three Jev evaluation axes."""

    accept: JevQueryConfig
    reaction: JevQueryConfig
    good_deal: JevQueryConfig


class JevEvaluationResult(BaseModel):
    """Raw quantitative evaluation output from Open-JEV."""

    accept_score: int = Field(ge=1, le=5, description="1 - Definitely reject, 5 - Definitely accept")
    reaction: ReactionType
    good_deal_confidence: float = Field(ge=0.0, le=1.0, description="Вероятность утверждения noul")


class VoiceDirective(BaseModel):
    """Strict directive passed to Voice LLM governing character dialogue."""

    reaction: ReactionType
    accept_score: int = Field(ge=1, le=5)
    good_deal_status: str
    counter_offer_investment: int | None = None
    counter_offer_equity: float | None = None


class GameStatus(str, Enum):
    """Terminal and ongoing statuses of the negotiation game."""

    IN_PROGRESS = "in_progress"
    WON = "won"
    LOST = "lost"
    QUIT = "quit"


class RoundResult(BaseModel):
    """Container recording the result of a single negotiation turn."""

    round_number: int = Field(ge=1)
    player_offer: PlayerOffer
    evaluation: JevEvaluationResult
    directive: VoiceDirective
    investor_dialogue: str
    game_status: GameStatus
    investor_state: InvestorState


__all__ = [
    "ReactionType",
    "PlayerOffer",
    "InvestorState",
    "GameState",
    "JevQueryConfig",
    "JevEvaluationSchema",
    "JevEvaluationResult",
    "VoiceDirective",
    "GameStatus",
    "RoundResult",
]
