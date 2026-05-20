from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from spider4ssc_zeroshot.config import load_experiment_config, load_model_groups
from spider4ssc_zeroshot.data import (
    ensure_dataset,
    load_split,
    normalize_examples_for_language,
    write_manifest,
)
from spider4ssc_zeroshot.evaluate import evaluate_predictions
from spider4ssc_zeroshot.prompting import PromptTemplate, render_prompt
from spider4ssc_zeroshot.report import collect_scores, write_reports
from spider4ssc_zeroshot.run_generation import GenerationRequest, run_generation
from spider4ssc_zeroshot.schema_serialization import serialize_example_schema
from spider4ssc_zeroshot.vllm_client import VllmClient, VllmClientConfig

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
    if source is None and experiment.dataset.archive_sha256 is None:
        raise typer.BadParameter(
            "dataset.archive_sha256 is required for remote downloads; "
            "pass --source to copy a local Spider4SSC tree",
            param_hint="--config",
        )
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


@app.command("generate")
def generate(
    run_id: str,
    language: str,
    config: Annotated[Path, typer.Option()] = Path("configs/experiment.yaml"),
    models: Annotated[Path, typer.Option()] = Path("configs/models.yaml"),
    model_group: Annotated[str, typer.Option()] = "main",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    experiment = load_experiment_config(config)
    if language not in experiment.experiment.languages:
        raise typer.BadParameter(f"Unsupported configured language: {language}")

    model = load_model_groups(models)[model_group][run_id]
    dataset_root = experiment.dataset.local_path
    split_file = (
        experiment.dataset.test_file if experiment.dataset.split == "test" else None
    )
    examples = normalize_examples_for_language(
        load_split(
            dataset_root,
            experiment.dataset.split,
            split_file=split_file,
        ),
        language,
    )
    if limit is not None:
        examples = examples[:limit]

    template = PromptTemplate.from_path(language, experiment.experiment.prompt_files[language])
    requests: list[GenerationRequest] = []
    for example in examples:
        schema = serialize_example_schema(dataset_root, example, language)
        prompt = render_prompt(template, schema=schema, question=example["question"])
        requests.append(
            GenerationRequest(
                example_id=example["example_id"],
                split=example["split"],
                language=language,
                db_id=example["db_id"],
                question=example["question"],
                gold_sql=example["sql"],
                prompt=prompt.text,
            )
        )

    endpoint = experiment.endpoint
    client = VllmClient(
        VllmClientConfig(
            base_url=endpoint.base_url,
            api_key=os.getenv(endpoint.api_key_env, "token-abc123"),
            readiness_timeout_seconds=endpoint.readiness_timeout_seconds,
            request_timeout_seconds=endpoint.request_timeout_seconds,
            max_retries=endpoint.max_retries,
            retry_sleep_seconds=endpoint.retry_sleep_seconds,
        )
    )
    client.wait_until_ready(model.model_id)
    output_file = experiment.experiment.output_root / run_id / language / "predictions.jsonl"
    run_generation(
        requests=requests,
        client=client,
        model_id=model.model_id,
        decoding=experiment.decoding.model_dump(),
        output_file=output_file,
    )
    typer.echo(f"Wrote predictions to {output_file}")


@app.command("evaluate")
def evaluate(
    run_id: str,
    language: str,
    config: Annotated[Path, typer.Option()] = Path("configs/experiment.yaml"),
) -> None:
    experiment = load_experiment_config(config)
    run_dir = experiment.experiment.output_root / run_id / language
    score = evaluate_predictions(
        predictions_file=run_dir / "predictions.jsonl",
        dataset_root=experiment.dataset.local_path,
        run_id=run_id,
        language=language,
        output_file=run_dir / "scores.json",
    )
    typer.echo(f"{run_id} {language} execution_accuracy={score['execution_accuracy']:.4f}")


@app.command("report")
def report(
    runs_dir: Annotated[Path, typer.Option()] = Path("runs/test"),
    output_dir: Annotated[Path, typer.Option()] = Path("reports"),
) -> None:
    frame = collect_scores(runs_dir)
    write_reports(frame, output_dir)
    typer.echo(f"Wrote reports to {output_dir}")
