import sys
from typing import Any
from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from schemas import (
    GameStatus,
    InvestorState,
    JevEvaluationResult,
    RoundResult,
)

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class CliRenderer:
    """Renders formatted Rich terminal UI components for startup negotiations."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the renderer with an optional custom Rich Console."""
        self.console: Console = console if console is not None else Console()

    def _print_and_return(self, renderable: Panel) -> Panel:
        """Helper to print a renderable to the bound console and return it."""
        self.console.print(renderable)
        return renderable

    def render_banner(self) -> Panel:
        """Render the top game banner with logo and title."""
        banner_ascii = (
            "  ___  ____  _____ _   _        _ _______      __\n"
            " / _ \\|  _ \\| ____| \\ | |      | | ____\\ \\    / /\n"
            "| | | | |_) |  _| |  \\| |______| |  _|  \\ \\  / / \n"
            "| |_| |  __/| |___| |\\  |______| | |___  \\ \\/ /  \n"
            " \\___/|_|   |_____|_| \\_|      |_|_____|  \\__/   \n"
        )
        subtitle = (
            "[bold white]NEGOTIATOR[/bold white] • "
            "[bold cyan]STARTUP VENTURE DEAL OR NO DEAL[/bold cyan]\n"
            "[dim]Powered by Open-Jev-9B Quantitative Decision Engine[/dim]"
        )
        content = f"[bold green]{banner_ascii}[/bold green]\n{subtitle}"
        panel = Panel(
            Align.center(content),
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_pitch_card(
        self,
        valuation: int | None = None,
        budget: int | None = None,
        patience: int | None = None,
        *,
        company_valuation: int | None = None,
        investor_budget: int | None = None,
    ) -> Panel:
        """Render startup pitch overview card with valuation, budget, and patience."""
        val = valuation if valuation is not None else company_valuation
        bud = budget if budget is not None else investor_budget
        val_str = f"€{val:,}" if val is not None else "Н/Д"
        bud_str = f"€{bud:,}" if bud is not None else "Н/Д"

        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Параметр", style="bold white", width=30)
        table.add_column("Значение", style="bold cyan")

        table.add_row("🚀 Рыночная оценка стартапа:", f"[bold cyan]{val_str}[/bold cyan]")
        table.add_row("💼 Бюджет венчурного фонда:", f"[bold yellow]{bud_str}[/bold yellow]")

        if patience is not None:
            hearts = "♥ " * patience if patience > 0 else "[dim]исчерпано[/dim]"
            table.add_row(
                "⏳ Запас терпения инвестора:",
                f"[bold red]{hearts.strip()}[/bold red] ({patience} раундов)",
            )

        panel = Panel(
            table,
            title="[bold cyan]📋 Карточка Питча / Startup Pitch Overview[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_investor_panel(self, investor_state: InvestorState) -> Panel:
        """Render investor state panel including capital, interest rate, and patience hearts."""
        hearts_str = "♥ " * investor_state.patience if investor_state.patience > 0 else ""
        hearts_display = (
            f"[bold red]{hearts_str.strip()}[/bold red]"
            if investor_state.patience > 0
            else "[dim](исчерпано)[/dim]"
        )
        interest_pct = f"{int(investor_state.interest * 100)}%"
        budget_str = f"€{investor_state.budget:,}"
        max_eq_str = f"{investor_state.max_equity:.1f}%"

        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Метрика", style="bold white", width=28)
        table.add_column("Значение", style="bold")

        table.add_row("💰 Свободный капитал:", f"[bold green]{budget_str}[/bold green]")
        table.add_row("📊 Максимальная доля:", f"[bold yellow]{max_eq_str}[/bold yellow]")
        table.add_row("🔥 Заинтересованность:", f"[bold magenta]{interest_pct}[/bold magenta]")
        table.add_row("❤️ Терпение инвестора:", hearts_display)

        panel = Panel(
            table,
            title="[bold yellow]💼 Профиль Инвестора / Investor Stance[/bold yellow]",
            border_style="yellow",
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_dialogue(
        self,
        speaker: str,
        text: str,
        is_player: bool = False,
    ) -> Panel:
        """Render dialogue speech bubble styled distinctly for Player and Investor."""
        is_founder = is_player or speaker.strip().lower() in (
            "player",
            "founder",
            "основатель",
            "игрок",
        )

        if is_founder:
            border_color = "cyan"
            title_text = f"[bold cyan]🧑 {speaker} (Founder)[/bold cyan]"
            content_style = "white"
        else:
            border_color = "yellow"
            title_text = f"[bold yellow]💼 {speaker} (Investor)[/bold yellow]"
            content_style = "italic white"

        panel = Panel(
            Text(text, style=content_style),
            title=title_text,
            border_style=border_color,
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_eval_telemetry(self, eval_result: JevEvaluationResult) -> Panel:
        """Render Open-JEV quantitative evaluation metrics (score, reaction, confidence)."""
        reaction_colors = {
            "accept": "bold green",
            "counter": "bold yellow",
            "reject": "bold red",
            "walk_away": "bold dark_red",
        }
        reaction_color = reaction_colors.get(eval_result.reaction, "bold white")

        score_stars = "★" * eval_result.accept_score + "☆" * (5 - eval_result.accept_score)
        confidence_pct = f"{int(eval_result.good_deal_confidence * 100)}%"

        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Ось оценки", style="bold white", width=28)
        table.add_column("Показатель", style="bold")

        table.add_row(
            "🎯 Accept Score (1-5):",
            f"[bold cyan]{eval_result.accept_score}/5[/bold cyan] [yellow]{score_stars}[/yellow]",
        )
        table.add_row(
            "⚡ Реакция (Reaction):",
            f"[{reaction_color}]{eval_result.reaction}[/{reaction_color}]",
        )
        table.add_row(
            "📈 Good Deal Confidence:",
            f"[bold magenta]{confidence_pct}[/bold magenta] ({eval_result.good_deal_confidence:.2f})",
        )

        panel = Panel(
            table,
            title="[bold blue]🧠 Open-JEV 9B Evaluation Telemetry[/bold blue]",
            border_style="blue",
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_round_summary(self, round_result: RoundResult) -> Panel:
        """Render structured summary of a completed negotiation round."""
        offer = round_result.player_offer
        eval_res = round_result.evaluation
        inv = round_result.investor_state

        hearts_str = "♥ " * inv.patience if inv.patience > 0 else "[dim](0)[/dim]"

        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Показатель", style="bold white", width=25)
        table.add_column("Данные", style="bold")

        table.add_row(
            "Оффер основателя:",
            f"[bold cyan]€{offer.investment:,}[/bold cyan] за [bold cyan]{offer.equity:.1f}%[/bold cyan] "
            f"[dim](оценка: €{int(offer.implied_valuation):,})[/dim]",
        )
        table.add_row("Вердикт Jev:", f"[bold]{eval_res.reaction}[/bold]")
        table.add_row("Статус игры:", f"[bold]{round_result.game_status.value}[/bold]")
        table.add_row("Остаток терпения:", f"[bold red]{hearts_str.strip()}[/bold red]")

        panel = Panel(
            table,
            title=f"[bold]🏁 Итоги Раунда {round_result.round_number}[/bold]",
            border_style="blue",
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_game_over(
        self,
        status: GameStatus,
        session: Any | None = None,
    ) -> Panel:
        """Render end game announcement banner with outcome fanfare or walk-away notice."""
        rounds_played = session.round_number if session is not None else 0
        stats_line = f"\n[dim]Сыграно раундов: {rounds_played}[/dim]" if rounds_played > 0 else ""

        if status == GameStatus.WON:
            border_style = "green"
            title = "[bold green]🏆 ПОБЕДА! СДЕЛКА ЗАКЛЮЧЕНА / DEAL CLOSED[/bold green]"
            msg = (
                "[bold green]Поздравляем! Вы успешно договорились с венчурным инвестором![/bold green]\n"
                "Инвестиционный раунд успешно закрыт. Компания получила необходимое финансирование."
                f"{stats_line}"
            )
        elif status == GameStatus.QUIT:
            border_style = "yellow"
            title = "[bold yellow]🚪 ПЕРЕГОВОРЫ ПРЕРВАНЫ / QUIT[/bold yellow]"
            msg = (
                "[bold yellow]Вы решили прервать переговоры и покинуть стол.[/bold yellow]\n"
                "Сделка не состоялась. Возможно, в другой раз условия будут благоприятнее."
                f"{stats_line}"
            )
        else:
            border_style = "red"
            title = "[bold red]💀 ПОРАЖЕНИЕ! ИНВЕСТОР ПОКИНУЛ ВСТРЕЧУ / WALK AWAY[/bold red]"
            msg = (
                "[bold red]Инвестор встал из-за стола переговоров и ушел.[/bold red]\n"
                "Лимит терпения исчерпан либо условия оказались неприемлемыми."
                f"{stats_line}"
            )

        panel = Panel(
            Align.center(msg),
            title=title,
            border_style=border_style,
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)

    def render_help(self) -> Panel:
        """Render command instructions and offer syntax guide."""
        table = Table(box=None, expand=True, show_header=True)
        table.add_column("Команда / Синтаксис", style="bold cyan", width=30)
        table.add_column("Описание", style="white")

        table.add_row("500k 10%", "Предложить €500,000 за 10% доли компании")
        table.add_row("€400,000 for 12.5%", "Альтернативный формат ввода оффера")
        table.add_row("300k за 20%", "Русскоязычный синтаксис ввода оффера")
        table.add_row("500000 15", "Числовой ввод без буквенных сокращений")
        table.add_row("status", "Показать текущий статус переговоров и карточку")
        table.add_row("help", "Показать данную справку по командам")
        table.add_row("exit / quit / give up", "Прервать переговоры и выйти из игры")

        guide_text = (
            "[bold white]Правила переговоров:[/bold white]\n"
            "• Инвестор оценивает обоснованность запрашиваемой оценки компании.\n"
            "• С каждым раундом терпение инвестора ([red]♥[/red]) уменьшается на 1.\n"
            "• Если терпение падает до 0, инвестор уходит и раунд проигран.\n\n"
        )

        content = Text.from_markup(guide_text)

        panel = Panel(
            Group(content, table),
            title="[bold cyan]ℹ️ Справка и Команды / Commands Guide[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
            padding=(1, 2),
        )
        return self._print_and_return(panel)


__all__ = ["CliRenderer"]
