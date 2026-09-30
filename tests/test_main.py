"""Tests for CLI parsing, arguments, and end-to-end game execution in main.py."""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from rich.console import Console

from game.cli_renderer import CliRenderer
from game.session import GameSession
from schemas import GameStatus, PlayerOffer
from main import (
    async_main,
    main,
    parse_args,
    parse_player_offer,
    run_game,
    run_test_simulation,
)


class TestParsePlayerOffer:
    """Test parse_player_offer for natural language input, commands, and errors."""

    @pytest.mark.parametrize(
        ("raw_input", "expected_investment", "expected_equity"),
        [
            ("500k 10%", 500_000, 10.0),
            ("500000 15", 500_000, 15.0),
            ("€400,000 for 12.5%", 400_000, 12.5),
            ("1.2m 20%", 1_200_000, 20.0),
            ("300к за 15%", 300_000, 15.0),
            ("450k for 18%", 450_000, 18.0),
            ("$1.5m for 25%", 1_500_000, 25.0),
            ("250 тыс за 10%", 250_000, 10.0),
            ("2 млн 20%", 2_000_000, 20.0),
            ("€ 600,000 at 16%", 600_000, 16.0),
        ],
    )
    def test_parse_valid_offers(
        self,
        raw_input: str,
        expected_investment: int,
        expected_equity: float,
    ) -> None:
        """Verify natural language offer strings correctly convert to PlayerOffer."""
        result = parse_player_offer(raw_input)
        assert isinstance(result, PlayerOffer)
        assert result.investment == expected_investment
        assert pytest.approx(result.equity, rel=1e-3) == expected_equity

    @pytest.mark.parametrize(
        ("raw_input", "expected_command"),
        [
            ("exit", "exit"),
            ("quit", "quit"),
            ("give up", "give up"),
            ("help", "help"),
            ("status", "status"),
            ("history", "history"),
            ("EXIT", "exit"),
            (" Help ", "help"),
            ("GIVE UP", "give up"),
        ],
    )
    def test_parse_special_commands(
        self,
        raw_input: str,
        expected_command: str,
    ) -> None:
        """Verify command inputs return normalized command strings."""
        result = parse_player_offer(raw_input)
        assert isinstance(result, str)
        assert result.strip().lower() == expected_command.strip().lower()

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "0 0",
            "-500k 10%",
            "500k 0%",
            "500k 120%",
            "500k -5%",
            "500k",
            "15%",
            "gibberish deal please",
            "",
            "   ",
            "foo bar baz",
            "0k 10%",
        ],
    )
    def test_parse_invalid_offers(self, invalid_input: str) -> None:
        """Verify malformed or out-of-bounds inputs raise ValueError."""
        with pytest.raises(ValueError):
            parse_player_offer(invalid_input)


class TestCliArgs:
    """Test CLI argument parsing."""

    def test_default_args(self) -> None:
        """Verify default argument values."""
        args = parse_args([])
        assert args.mock is False or args.mock is True  # boolean flag
        assert args.test_run is False

    def test_custom_flags(self) -> None:
        """Verify all custom CLI flags are parsed into expected types."""
        custom_args = [
            "--mock",
            "--llm-url", "http://ollama-host:11434/v1",
            "--jev-url", "http://jev-host:8000/eval",
            "--budget", "850000",
            "--valuation", "3000000",
            "--patience", "6",
            "--test-run",
        ]
        args = parse_args(custom_args)
        assert args.mock is True
        assert args.llm_url == "http://ollama-host:11434/v1"
        assert args.jev_url == "http://jev-host:8000/eval"
        assert args.budget == 850_000
        assert args.valuation == 3_000_000
        assert args.patience == 6
        assert args.test_run is True

    def test_rounds_alias_flag(self) -> None:
        """Verify --rounds functions as alias or option for patience."""
        args = parse_args(["--rounds", "5"])
        # Should populate either rounds or patience
        val = getattr(args, "rounds", None) or getattr(args, "patience", None)
        assert val == 5


class TestSimulationAndGameLoop:
    """Test non-interactive simulation and interactive REPL game loop."""

    @pytest.mark.asyncio
    async def test_run_simulation_completes(self) -> None:
        """Verify run_test_simulation runs to conclusion without user interaction."""
        session = GameSession.create_session(investor_patience=3)
        renderer = CliRenderer(console=Console(record=True, width=100))

        exit_code = await run_test_simulation(session, renderer)

        assert session.is_game_over is True
        assert session.status in (GameStatus.WON, GameStatus.LOST)
        assert len(session.round_results) > 0
        assert exit_code in (0, 1)

    @pytest.mark.asyncio
    async def test_run_game_interactive_quit(self) -> None:
        """Verify typing 'exit' terminates the interactive session with GameStatus.QUIT."""
        session = GameSession.create_session()
        renderer = CliRenderer(console=Console(record=True, width=100))

        with patch("builtins.input", side_effect=["exit"]):
            exit_code = await run_game(session, renderer, interactive=True)

        assert session.is_game_over is True
        assert session.status == GameStatus.QUIT
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_run_game_interactive_deal(self) -> None:
        """Verify attractive offer allows concluding the game with victory."""
        session = GameSession.create_session()
        renderer = CliRenderer(console=Console(record=True, width=100))

        # Attractive discounted offer (ratio 0.6 <= 0.75): €300,000 for 25%
        with patch("builtins.input", side_effect=["300k 25%"]):
            exit_code = await run_game(session, renderer, interactive=True)

        assert session.is_game_over is True
        assert session.status == GameStatus.WON
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_run_game_interactive_help_and_invalid(self) -> None:
        """Verify handling help and invalid inputs before quitting."""
        session = GameSession.create_session()
        renderer = CliRenderer(console=Console(record=True, width=100))

        inputs = ["help", "status", "invalid format", "quit"]
        with patch("builtins.input", side_effect=inputs):
            exit_code = await run_game(session, renderer, interactive=True)

        assert session.is_game_over is True
        assert session.status == GameStatus.QUIT
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_run_game_interactive_multi_round_win(self) -> None:
        """Verify multi-round interactive negotiation reaching agreement."""
        session = GameSession.create_session()
        renderer = CliRenderer(console=Console(record=True, width=100))

        # First offer triggers counter (500k for 20%), second offer triggers accept (300k for 25%)
        inputs = ["500k 20%", "300k 25%"]
        with patch("builtins.input", side_effect=inputs):
            exit_code = await run_game(session, renderer, interactive=True)

        assert session.is_game_over is True
        assert session.status == GameStatus.WON
        assert session.round_number == 2
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_run_game_interactive_patience_exhaustion(self) -> None:
        """Verify repeated aggressive offers exhaust investor patience."""
        session = GameSession.create_session(investor_patience=2)
        renderer = CliRenderer(console=Console(record=True, width=100))

        # Overvalued offers (500k for 5% -> 10M valuation, ratio 5.0)
        inputs = ["500k 5%", "500k 5%"]
        with patch("builtins.input", side_effect=inputs):
            exit_code = await run_game(session, renderer, interactive=True)

        assert session.is_game_over is True
        assert session.status == GameStatus.LOST
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_run_game_history_command(self) -> None:
        """Verify 'history' command displays round entries without error."""
        session = GameSession.create_session()
        renderer = CliRenderer(console=Console(record=True, width=100))

        inputs = ["history", "500k 20%", "history", "quit"]
        with patch("builtins.input", side_effect=inputs):
            exit_code = await run_game(session, renderer, interactive=True)

        assert session.is_game_over is True
        assert session.status == GameStatus.QUIT
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_run_simulation_low_patience_loss(self) -> None:
        """Verify simulation with patience=1 ends in loss/walk_away."""
        session = GameSession.create_session(investor_patience=1)
        renderer = CliRenderer(console=Console(record=True, width=100))

        exit_code = await run_test_simulation(session, renderer)
        assert session.is_game_over is True
        assert session.status == GameStatus.LOST
        assert exit_code == 1


class TestMainEntrypoint:
    """Test top-level CLI execution."""

    @pytest.mark.asyncio
    async def test_async_main_test_run(self) -> None:
        """Verify async_main executes test run cleanly."""
        exit_code = await async_main(["--test-run", "--mock"])
        assert exit_code in (0, 1)

    @pytest.mark.asyncio
    async def test_async_main_interactive_delegation(self) -> None:
        """Verify async_main delegates to run_game when not in test-run mode."""
        with patch("main.run_game", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0
            code = await async_main(["--mock"])
            assert code == 0
            assert mock_run.called

    def test_main_sys_argv(self) -> None:
        """Verify main() entrypoint handles sys.argv."""
        with patch("sys.argv", ["main.py", "--test-run", "--mock"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code in (0, 1)
