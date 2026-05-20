from __future__ import annotations

import typer

app = typer.Typer(help="Spider4SSC zero-shot reproducibility commands.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Spider4SSC zero-shot reproducibility commands."""
