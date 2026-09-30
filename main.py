"""Main CLI entrypoint and interactive game loop for Open-JEV Negotiator."""
import argparse
import asyncio
import re
import sys
from typing import Literal

from config import Settings, settings
from game.cli_renderer import CliRenderer
from game.session import GameSession
from schemas import GameStatus, PlayerOffer

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SPECIAL_COMMANDS: dict[str, str] = {
    "exit": "exit",
    "quit": "quit",
    "q": "quit",
    "give up": "give up",
    "giveup": "give up",
    "give_up": "give up",
    "help": "help",
    "h": "help",
    "?": "help",
    "status": "status",
    "history": "history",
}

OFFER_REGEX = re.compile(
    r"""
    ^\s*
    (?:[€$£₽]|eur|usd|rub)?\s*
    (?P<inv_sign>[-+])?\s*
    (?P<inv_val>
        \d{1,3}(?:,\d{3})+       # e.g. 400,000 or 1,200,000
        |
        \d+(?:[.,]\d+)?          # e.g. 500000, 1.2, 1,5
    )\s*
    (?P<inv_mult>
        k|к|тыс(?:\.|яч)?|тысяч[а-я]*
        |
        m|м|млн|миллион[а-я]*
    )?\s*
    (?:[€$£₽]|eur|usd|rub)?\s*
    (?:for|за|at|на|доля|:|,|\s+[-—–]\s+|\s+)\s*
    (?P<eq_sign>[-+])?\s*
    (?P<eq_val>\d+(?:[.,]\d+)?)\s*
    %?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_player_offer(raw_input: str) -> PlayerOffer | str:
    """Parse player raw text into a validated PlayerOffer or special command string.

    Args:
        raw_input: Natural text typed by the player (e.g., '500k 10%', 'exit', 'help').

    Returns:
        PlayerOffer if valid offer numbers found, or normalized command string.

    Raises:
        ValueError: If input format is malformed or numerical bounds are violated.
    """
    clean = raw_input.strip()
    if not clean:
        raise ValueError("Ввод не может быть пустым. Введите предложение или команду.")

    # 1. Check special commands
    lowered = clean.lower()
    if lowered in SPECIAL_COMMANDS:
        return SPECIAL_COMMANDS[lowered]

    # 2. Match offer pattern
    match = OFFER_REGEX.match(clean)
    if not match:
        raise ValueError(
            f"Не удалось распознать предложение: '{raw_input}'. "
            "Используйте формат: '500k 10%', '400000 15' или 'help'."
        )

    inv_val_str = match.group("inv_val")
    inv_mult_str = match.group("inv_mult")
    inv_sign = match.group("inv_sign")
    eq_val_str = match.group("eq_val")
    eq_sign = match.group("eq_sign")

    # Clean investment value
    if "," in inv_val_str and not inv_mult_str and re.match(r"^\d{1,3}(?:,\d{3})+$", inv_val_str):
        # Thousand separator
        clean_inv_str = inv_val_str.replace(",", "")
    else:
        # Decimal comma
        clean_inv_str = inv_val_str.replace(",", ".")

    try:
        base_inv = float(clean_inv_str)
    except ValueError as exc:
        raise ValueError(f"Некорректная сумма инвестиций: {inv_val_str}") from exc

    multiplier = 1
    if inv_mult_str:
        mult_lower = inv_mult_str.lower()
        if mult_lower.startswith(("k", "к", "тыс")):
            multiplier = 1_000
        elif mult_lower.startswith(("m", "м", "млн", "мил")):
            multiplier = 1_000_000

    investment = int(round(base_inv * multiplier))
    if inv_sign == "-":
        investment = -investment

    # Clean equity value
    try:
        equity = float(eq_val_str.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Некорректная доля: {eq_val_str}") from exc

    if eq_sign == "-":
        equity = -equity

    # Validate mathematical bounds
    if investment <= 0:
        raise ValueError(f"Сумма инвестиций должна быть больше 0€ (получено: {investment}€).")
    if equity <= 0 or equity > 100:
        raise ValueError(
            f"Доля компании должна быть в диапазоне (0, 100]% (получено: {equity}%)."
        )

    return PlayerOffer(investment=investment, equity=round(equity, 2))


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Define and parse CLI arguments for Open-JEV Negotiator.

    Args:
        args: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        argparse.Namespace with parsed CLI configuration.
    """
    parser = argparse.ArgumentParser(
        description="Open-JEV Negotiator: Startup Venture Deal or No Deal CLI Game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force offline mock mode for quantitative evaluation and voice dialogue",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=settings.LLM_URL,
        help="Voice LLM endpoint URL (Ollama, vLLM, OpenAI-compatible)",
    )
    parser.add_argument(
        "--jev-url",
        type=str,
        default=settings.JEV_URL,
        help="Open-JEV evaluation engine endpoint URL",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=settings.DEFAULT_INVESTOR_BUDGET,
        help="Initial investor budget in EUR",
    )
    parser.add_argument(
        "--valuation",
        type=int,
        default=settings.DEFAULT_COMPANY_VALUATION,
        help="Initial company market valuation in EUR",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of negotiation rounds before investor walks away",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Alias for --patience",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        default=False,
        help="Execute non-interactive automated simulation to verify pipeline",
    )

    parsed = parser.parse_args(args)

    # Harmonize --rounds and --patience
    if parsed.rounds is not None and parsed.patience is None:
        parsed.patience = parsed.rounds
    elif parsed.patience is not None and parsed.rounds is None:
        parsed.rounds = parsed.patience

    return parsed


async def run_test_simulation(session: GameSession, renderer: CliRenderer) -> int:
    """Run an automated non-interactive simulation to test game pipeline.

    Args:
        session: Pre-configured GameSession.
        renderer: CliRenderer instance.

    Returns:
        Exit code: 0 on WON, 1 on LOST.
    """
    renderer.render_banner()
    renderer.render_pitch_card(
        company_valuation=session.company_valuation,
        investor_budget=session.investor.budget,
        patience=session.patience,
    )
    renderer.render_investor_panel(session.investor)

    opening_terms, opening_speech = session.get_opening_stance()
    renderer.render_dialogue("Investor", opening_speech)

    # Simulated founder progression: starts aggressive, makes concessions, achieves agreement
    simulated_script = [
        PlayerOffer(investment=min(500_000, session.investor.budget), equity=15.0),
        PlayerOffer(investment=min(450_000, session.investor.budget), equity=20.0),
        PlayerOffer(
            investment=min(300_000, session.investor.budget),
            equity=min(25.0, session.investor.max_equity),
        ),
    ]

    offer_idx = 0
    while session.is_active:
        if offer_idx < len(simulated_script):
            offer = simulated_script[offer_idx]
            offer_idx += 1
        else:
            # Fallback to generous terms to close
            offer = PlayerOffer(
                investment=min(300_000, session.investor.budget),
                equity=min(30.0, session.investor.max_equity),
            )
        renderer.render_dialogue(
            "Founder",
            f"Предлагаю €{offer.investment:,} за {offer.equity:.1f}% доли компании.",
            is_player=True,
        )

        round_res = await session.step(offer)
        renderer.render_eval_telemetry(round_res.evaluation)
        renderer.render_dialogue("Investor", round_res.investor_dialogue)
        renderer.render_round_summary(round_res)

    renderer.render_game_over(session.status, session)
    return 0 if session.status == GameStatus.WON else 1


async def run_game(
    session: GameSession,
    renderer: CliRenderer,
    interactive: bool = True,
) -> int:
    """Run interactive REPL game loop for startup negotiation.

    Args:
        session: Active GameSession instance.
        renderer: CliRenderer instance.
        interactive: If False, delegates to run_test_simulation.

    Returns:
        Exit code: 0 on WON, 1 on LOST or QUIT.
    """
    if not interactive:
        return await run_test_simulation(session, renderer)

    renderer.render_banner()
    renderer.render_pitch_card(
        company_valuation=session.company_valuation,
        investor_budget=session.investor.budget,
        patience=session.patience,
    )
    renderer.render_investor_panel(session.investor)

    opening_terms, opening_speech = session.get_opening_stance()
    renderer.render_dialogue("Investor", opening_speech)

    while session.is_active:
        try:
            renderer.console.print(
                f"\n[bold green]Раунд {session.round_number + 1} | "
                f"Ваше предложение (напр. '500k 15%', 'help', 'exit') > [/bold green]",
                end="",
            )
            raw_input = input()
        except (EOFError, KeyboardInterrupt):
            session.give_up(status=GameStatus.QUIT)
            break

        try:
            parsed = parse_player_offer(raw_input)
        except ValueError as err:
            renderer.console.print(f"[bold red]⚠️  Ошибка: {err}[/bold red]")
            continue

        if isinstance(parsed, str):
            cmd = parsed.lower()
            if cmd in ("exit", "quit", "give up"):
                session.give_up(status=GameStatus.QUIT)
                break
            elif cmd == "help":
                renderer.render_help()
                continue
            elif cmd == "status":
                renderer.render_pitch_card(
                    company_valuation=session.company_valuation,
                    investor_budget=session.investor.budget,
                    patience=session.patience,
                )
                renderer.render_investor_panel(session.investor)
                continue
            elif cmd == "history":
                if not session.history:
                    renderer.console.print("[dim]История переговоров пока пуста.[/dim]")
                else:
                    for h in session.history:
                        renderer.console.print(f"[dim cyan]• {h}[/dim cyan]")
                continue

        # Valid PlayerOffer received
        renderer.render_dialogue(
            "Founder",
            f"Предлагаю €{parsed.investment:,} за {parsed.equity:.1f}% доли компании.",
            is_player=True,
        )

        round_res = await session.step(parsed)
        renderer.render_eval_telemetry(round_res.evaluation)
        renderer.render_dialogue("Investor", round_res.investor_dialogue)
        renderer.render_round_summary(round_res)

    renderer.render_game_over(session.status, session)
    return 0 if session.status == GameStatus.WON else 1


async def async_main(args: list[str] | None = None) -> int:
    """Asynchronous orchestrator for CLI execution."""
    parsed_args = parse_args(args)

    effective_patience = (
        parsed_args.patience
        if parsed_args.patience is not None
        else settings.DEFAULT_INVESTOR_PATIENCE
    )

    app_settings = Settings(
        MOCK_MODE=True if parsed_args.mock else settings.MOCK_MODE,
        LLM_URL=parsed_args.llm_url or settings.LLM_URL,
        JEV_URL=parsed_args.jev_url or settings.JEV_URL,
        DEFAULT_COMPANY_VALUATION=parsed_args.valuation or settings.DEFAULT_COMPANY_VALUATION,
        DEFAULT_INVESTOR_BUDGET=parsed_args.budget or settings.DEFAULT_INVESTOR_BUDGET,
        DEFAULT_INVESTOR_PATIENCE=effective_patience,
    )

    renderer = CliRenderer()
    session = GameSession.create_session(
        company_valuation=app_settings.DEFAULT_COMPANY_VALUATION,
        investor_budget=app_settings.DEFAULT_INVESTOR_BUDGET,
        investor_patience=app_settings.DEFAULT_INVESTOR_PATIENCE,
        settings_obj=app_settings,
    )

    if parsed_args.test_run:
        return await run_test_simulation(session, renderer)
    return await run_game(session, renderer, interactive=True)


def main(args: list[str] | None = None) -> None:
    """Synchronous entrypoint for CLI command execution."""
    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
