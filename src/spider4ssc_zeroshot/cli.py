from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from spider4ssc_zeroshot.config import load_experiment_config
from spider4ssc_zeroshot.data import ensure_dataset, write_manifest

app = typer.Typer(help="Spider4SSC zero-shot reproducibility commands.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Spider4SSC zero-shot reproducibility commands."""


@app.command("prepare-data")
def prepare_data(
    source: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False),
    ] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    config: Annotated[Path, typer.Option()] = Path("configs/experiment.yaml"),
) -> None:
    experiment = load_experiment_config(config)
    output_path = output or experiment.dataset.local_path
    ensure_dataset(
        output_path,
        source=source,
        url=experiment.dataset.url,
        split=experiment.dataset.split,
        test_file=experiment.dataset.test_file,
        test_db_dir=experiment.dataset.test_db_dir,
        archive_sha256=experiment.dataset.archive_sha256,
    )
    write_manifest(output_path, output_path.parent / "Spider4SSC.manifest.json")
    typer.echo(f"Prepared Spider4SSC at {output_path}")
