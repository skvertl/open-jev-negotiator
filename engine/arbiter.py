"""Decision Arbiter module enforcing deterministic game rules and guardrails."""
from schemas import GameState, JevEvaluationResult, VoiceDirective


class DecisionArbiter:
    """Arbiter that enforces physical and business guardrails over raw evaluations."""

    def __init__(
        self,
        valuation_threshold_multiplier: float = 2.5,
        counter_equity_bump_ratio: float = 1.25,
    ) -> None:
        self.valuation_threshold_multiplier = valuation_threshold_multiplier
        self.counter_equity_bump_ratio = counter_equity_bump_ratio

    def check_guardrails(
        self, state: GameState, raw_result: JevEvaluationResult
    ) -> JevEvaluationResult:
        """Apply deterministic guardrails to prevent absurd, impossible, or contradictory deals.

        Guardrails:
        1. Predatory Valuation: If player implied valuation >= threshold (default 2.5x company valuation),
           force reaction to 'walk_away'.
        2. Budget Ceiling: If requested investment > investor budget, investor cannot fund it.
           If patience <= 1 or already walk_away, force 'walk_away', otherwise force 'reject'.
        3. Agreement on Terms: If player's offer meets or improves upon the investor's active counter-offer
           (within budget and max_equity constraints), force reaction to 'accept'.
        4. Patience Exhaustion: If patience <= 1 and reaction != 'accept', force reaction to 'walk_away'.
        """
        # Guardrail 1: Implied valuation exceeds acceptable multiple of company valuation
        if (
            state.player_offer.implied_valuation
            >= self.valuation_threshold_multiplier * state.company_valuation
        ):
            return raw_result.model_copy(update={"reaction": "walk_away"})

        # Guardrail 2: Investment request exceeds investor's available budget
        if state.player_offer.investment > state.investor.budget:
            new_reaction = (
                "walk_away"
                if (state.investor.patience <= 1 or raw_result.reaction == "walk_away")
                else "reject"
            )
            return raw_result.model_copy(update={"reaction": new_reaction})

        # Guardrail 3: Agreement on terms (founder accepted or improved upon counter-offer)
        if state.last_counter_offer is not None:
            target_inv, target_eq = state.last_counter_offer
            if (
                state.player_offer.investment <= target_inv
                and state.player_offer.equity >= target_eq
                and state.player_offer.equity <= state.investor.max_equity
                and state.player_offer.investment <= state.investor.budget
            ):
                return raw_result.model_copy(
                    update={
                        "reaction": "accept",
                        "accept_score": 5,
                        "good_deal_confidence": max(raw_result.good_deal_confidence, 0.95),
                    }
                )

        # Guardrail 4: Low patience forces walk_away unless already an accepted deal
        if state.investor.patience <= 1 and raw_result.reaction != "accept":
            return raw_result.model_copy(update={"reaction": "walk_away"})

        return raw_result

    def calculate_counter_offer(self, state: GameState) -> tuple[int, float]:
        """Calculate rational, advantageous counter-offer terms for the investor.

        Returns:
            tuple[int, float]: (counter_investment, counter_equity)
        """
        counter_investment = min(state.player_offer.investment, state.investor.budget)

        # Protect against division by zero
        interest = max(state.investor.interest, 0.01)
        company_val = max(state.company_valuation, 1)

        # Formula equity based on investor expectation and interest discount
        formula_equity = (counter_investment / (company_val * interest)) * 100.0

        # Minimum bump over founder's offer (default +25%)
        bumped_equity = state.player_offer.equity * self.counter_equity_bump_ratio

        target_equity = max(formula_equity, bumped_equity)

        # Clamp within (0, max_equity]
        clamped_equity = min(target_equity, state.investor.max_equity)
        clamped_equity = max(clamped_equity, 0.1)

        counter_equity = round(clamped_equity, 1)
        counter_equity = min(counter_equity, state.investor.max_equity)

        return counter_investment, counter_equity

    def build_voice_directive(
        self,
        state: GameState,
        evaluation: JevEvaluationResult,
        counter_offer: tuple[int, float] | None = None,
    ) -> VoiceDirective:
        """Build VoiceDirective payload passed to Voice LLM."""
        good_deal_status = (
            "выгодная сделка" if evaluation.good_deal_confidence >= 0.5 else "невыгодная сделка"
        )

        counter_investment: int | None = None
        counter_equity: float | None = None

        if evaluation.reaction == "counter":
            if counter_offer is not None:
                counter_investment, counter_equity = counter_offer
            else:
                counter_investment, counter_equity = self.calculate_counter_offer(state)

        return VoiceDirective(
            reaction=evaluation.reaction,
            accept_score=evaluation.accept_score,
            good_deal_status=good_deal_status,
            counter_offer_investment=counter_investment,
            counter_offer_equity=counter_equity,
        )

    def arbitrate(
        self, state: GameState, raw_result: JevEvaluationResult
    ) -> tuple[JevEvaluationResult, VoiceDirective]:
        """Apply guardrails to raw evaluation and generate authoritative voice directive."""
        guarded_eval = self.check_guardrails(state, raw_result)
        counter_offer = (
            self.calculate_counter_offer(state) if guarded_eval.reaction == "counter" else None
        )
        directive = self.build_voice_directive(state, guarded_eval, counter_offer)
        return guarded_eval, directive
