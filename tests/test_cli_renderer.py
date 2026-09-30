"""Unit tests for CliRenderer rich console interface."""
import io
import pytest
from rich.console import Console
from rich.panel import Panel

from game.cli_renderer import CliRenderer
from schemas import (
    GameStatus,
    InvestorState,
    JevEvaluationResult,
    PlayerOffer,
    RoundResult,
    VoiceDirective,
)
from game.session import GameSession


@pytest.fixture
def string_console() -> tuple[Console, io.StringIO]:
    """Provide a Rich console writing to an in-memory string buffer."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=100)
    return console, buf


@pytest.fixture
def renderer(string_console: tuple[Console, io.StringIO]) -> CliRenderer:
    """Provide a CliRenderer bound to the in-memory string console."""
    console, _ = string_console
    return CliRenderer(console=console)


class TestCliRendererInit:
    """Tests for CliRenderer initialization and console binding."""

    def test_default_console_initialization(self):
        r = CliRenderer()
        assert isinstance(r.console, Console)

    def test_custom_console_initialization(self, string_console: tuple[Console, io.StringIO]):
        console, _ = string_console
        r = CliRenderer(console=console)
        assert r.console is console


class TestRenderBanner:
    """Tests for banner and logo rendering."""

    def test_render_banner_outputs_title(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_banner()

        output = buf.getvalue()
        assert "OPEN-JEV" in output.upper() or "NEGOTIATOR" in output.upper()
        assert isinstance(panel, Panel)


class TestRenderPitchCard:
    """Tests for startup pitch card rendering."""

    def test_render_pitch_card_with_currency_and_patience(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_pitch_card(
            valuation=2_000_000,
            budget=700_000,
            patience=4,
        )

        output = buf.getvalue()
        assert "€" in output
        assert "2,000,000" in output or "2 000 000" in output
        assert "700,000" in output or "700 000" in output
        assert "4" in output
        assert isinstance(panel, Panel)

    def test_render_pitch_card_with_alias_keywords(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_pitch_card(
            company_valuation=3_500_000,
            investor_budget=900_000,
        )

        output = buf.getvalue()
        assert "€" in output
        assert "3,500,000" in output or "3 500 000" in output
        assert "900,000" in output or "900 000" in output
        assert isinstance(panel, Panel)


class TestRenderInvestorPanel:
    """Tests for investor status panel with hearts and interest."""

    def test_render_investor_panel_hearts_and_interest(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        investor = InvestorState(
            budget=750_000,
            max_equity=30.0,
            interest=0.75,
            patience=4,
        )

        panel = renderer.render_investor_panel(investor)
        output = buf.getvalue()

        # Output must contain 4 hearts for patience=4
        assert output.count("♥") == 4
        assert "75%" in output
        assert "750,000" in output or "750 000" in output
        assert "30" in output
        assert isinstance(panel, Panel)

    def test_render_investor_panel_zero_patience(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        investor = InvestorState(
            budget=500_000,
            max_equity=20.0,
            interest=0.50,
            patience=0,
        )

        renderer.render_investor_panel(investor)
        output = buf.getvalue()

        assert output.count("♥") == 0
        assert "50%" in output


class TestRenderDialogue:
    """Tests for player and investor speech styling."""

    def test_render_player_dialogue_cyan_styling(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_dialogue(
            speaker="Основатель",
            text="Предлагаю 500 000€ за 15%.",
            is_player=True,
        )

        output = buf.getvalue()
        assert "Основатель" in output
        assert "500 000€" in output or "500 000" in output
        assert isinstance(panel, Panel)
        assert panel.border_style == "cyan"

    def test_render_investor_dialogue_styling(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_dialogue(
            speaker="Инвестор",
            text="Это предложение меня не устраивает.",
            is_player=False,
        )

        output = buf.getvalue()
        assert "Инвестор" in output
        assert "не устраивает" in output
        assert isinstance(panel, Panel)
        assert panel.border_style in ("yellow", "magenta")

    def test_render_dialogue_auto_detects_player(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        panel = renderer.render_dialogue(speaker="Founder", text="Let's make a deal")
        assert panel.border_style == "cyan"


class TestRenderEvalTelemetry:
    """Tests for Open-JEV evaluation metrics display."""

    def test_render_eval_telemetry_fields(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        eval_result = JevEvaluationResult(
            accept_score=4,
            reaction="counter",
            good_deal_confidence=0.85,
        )

        panel = renderer.render_eval_telemetry(eval_result)
        output = buf.getvalue()

        assert "4" in output
        assert "counter" in output
        assert "85%" in output or "0.85" in output
        assert isinstance(panel, Panel)

    @pytest.mark.parametrize("reaction", ["accept", "counter", "reject", "walk_away"])
    def test_render_eval_telemetry_all_reactions(
        self, reaction: str, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        eval_result = JevEvaluationResult(
            accept_score=3,
            reaction=reaction,  # type: ignore[arg-type]
            good_deal_confidence=0.50,
        )
        panel = renderer.render_eval_telemetry(eval_result)
        assert isinstance(panel, Panel)


class TestRenderRoundSummary:
    """Tests for negotiation round outcomes summary."""

    def test_render_round_summary(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        round_res = RoundResult(
            round_number=2,
            player_offer=PlayerOffer(investment=450_000, equity=15.0),
            evaluation=JevEvaluationResult(
                accept_score=3,
                reaction="counter",
                good_deal_confidence=0.65,
            ),
            directive=VoiceDirective(
                reaction="counter",
                accept_score=3,
                good_deal_status="Приемлемо",
                counter_offer_investment=450_000,
                counter_offer_equity=20.0,
            ),
            investor_dialogue="Я готов дать 450к, но за 20%.",
            game_status=GameStatus.IN_PROGRESS,
            investor_state=InvestorState(
                budget=700_000,
                max_equity=30.0,
                interest=0.70,
                patience=2,
            ),
        )

        panel = renderer.render_round_summary(round_res)
        output = buf.getvalue()

        assert "2" in output
        assert "450,000" in output or "450 000" in output
        assert "15" in output
        assert "counter" in output
        assert isinstance(panel, Panel)


class TestRenderGameOver:
    """Tests for victory and defeat end game banners."""

    def test_render_game_over_won(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_game_over(GameStatus.WON)
        output = buf.getvalue()

        assert any(word in output.upper() for word in ["ПОБЕДА", "СДЕЛКА", "VICTORY", "WON"])
        assert panel.border_style == "green"

    def test_render_game_over_lost(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_game_over(GameStatus.LOST)
        output = buf.getvalue()

        assert any(word in output.upper() for word in ["ПОРАЖЕНИЕ", "ПРОВАЛ", "DEFEAT", "LOST", "WALK"])
        assert panel.border_style == "red"

    def test_render_game_over_quit(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_game_over(GameStatus.QUIT)
        output = buf.getvalue()

        assert any(word in output.upper() for word in ["ВЫХОД", "ПРЕРВАН", "QUIT", "SURRENDER"])
        assert panel.border_style in ("yellow", "red")

    def test_render_game_over_with_session(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        session = GameSession.create_session(investor_patience=3)
        session.give_up()

        panel = renderer.render_game_over(session.status, session=session)
        output = buf.getvalue()

        assert "1" in output  # 1 round played
        assert isinstance(panel, Panel)


class TestRenderHelp:
    """Tests for help instructions and command syntax rendering."""

    def test_render_help_content(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        panel = renderer.render_help()
        output = buf.getvalue()

        assert "500k 10%" in output
        assert "exit" in output or "quit" in output
        assert isinstance(panel, Panel)

    def test_render_help_includes_rules_text(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        renderer.render_help()
        output = buf.getvalue()

        assert "Правила" in output or "переговоров" in output


class TestRendererEdgeCases:
    """Tests for edge cases and default console execution."""

    def test_render_pitch_card_with_none_values(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        renderer.render_pitch_card(valuation=None, budget=None)
        output = buf.getvalue()
        assert "Н/Д" in output

    def test_render_round_summary_with_zero_patience(
        self, renderer: CliRenderer, string_console: tuple[Console, io.StringIO]
    ):
        _, buf = string_console
        round_res = RoundResult(
            round_number=1,
            player_offer=PlayerOffer(investment=300_000, equity=10.0),
            evaluation=JevEvaluationResult(
                accept_score=1,
                reaction="walk_away",
                good_deal_confidence=0.1,
            ),
            directive=VoiceDirective(
                reaction="walk_away",
                accept_score=1,
                good_deal_status="Неприемлемо",
            ),
            investor_dialogue="Ухожу.",
            game_status=GameStatus.LOST,
            investor_state=InvestorState(
                budget=500_000,
                max_equity=20.0,
                interest=0.4,
                patience=0,
            ),
        )

        renderer.render_round_summary(round_res)
        output = buf.getvalue()
        assert "(0)" in output

    def test_render_dialogue_speaker_case_variants(
        self, renderer: CliRenderer
    ):
        panel1 = renderer.render_dialogue(speaker="игрок", text="Привет")
        assert panel1.border_style == "cyan"

        panel2 = renderer.render_dialogue(speaker="Аркадий (Инвестор)", text="Здравствуйте")
        assert panel2.border_style == "yellow"
