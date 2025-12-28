from typer.testing import CliRunner

from pymort.cli import app


def test_cli_help_for_spec_aliases() -> None:
    runner = CliRunner()
    for args in (
        ["forecast", "--help"],
        ["price-bond", "--help"],
        ["hedge", "--help"],
        ["fit", "--help"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0
