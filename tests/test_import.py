from typer.testing import CliRunner

from spider4ssc_zeroshot import __version__
from spider4ssc_zeroshot.cli import app


def test_version_is_available():
    assert __version__ == "0.1.0"


def test_cli_help_runs():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Spider4SSC" in result.output
