from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from spider4ssc_zeroshot.config import load_experiment_config, load_model_groups
from spider4ssc_zeroshot.data import (
    ensure_dataset,
    load_split,
    normalize_examples_for_language,
    validate_split,
    write_manifest,
)
from spider4ssc_zeroshot.env import load_dotenv
from spider4ssc_zeroshot.evaluate import evaluate_predictions
from spider4ssc_zeroshot.openai_client import OpenAIChatClient, OpenAIChatClientConfig
from spider4ssc_zeroshot.pipeline_validation import validate_pipeline
from spider4ssc_zeroshot.prompting import PromptTemplate, render_prompt
from spider4ssc_zeroshot.report import collect_scores, write_reports
from spider4ssc_zeroshot.run_generation import GenerationRequest, run_generation
from spider4ssc_zeroshot.schema_extraction import extract_neo4j_schemas
from spider4ssc_zeroshot.schema_serialization import (
    schema_provenance_for_example,
    serialize_example_schema,
)
from spider4ssc_zeroshot.vllm_client import VllmClient, VllmClientConfig

app = typer.Typer(help="Spider4SSC zero-shot reproducibility commands.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Spider4SSC zero-shot reproducibility commands."""


def _effective_split(experiment_split: str, split: str | None) -> str:
    try:
        return validate_split(split or experiment_split)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--split") from error


def _effective_schema_mode(experiment_schema_mode: str, schema_mode: str | None) -> str:
    effective = schema_mode or experiment_schema_mode
    if effective not in {"strict", "fallback"}:
        raise typer.BadParameter(
            "schema_mode must be one of: strict, fallback",
            param_hint="--schema-mode",
        )
    return effective


def _api_key_for_provider(endpoint_api_key_env: str, provider: str) -> str:
    load_dotenv()
    api_key = os.getenv(endpoint_api_key_env)
    if provider == "openai":
        if not api_key:
            raise typer.BadParameter(
                f"{endpoint_api_key_env} is required for OpenAI runs",
                param_hint=endpoint_api_key_env,
            )
        return api_key
    return api_key or "token-abc123"


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
    split: Annotated[str | None, typer.Option()] = None,
    schema_mode: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    experiment = load_experiment_config(config)
    if language not in experiment.experiment.languages:
        raise typer.BadParameter(f"Unsupported configured language: {language}")
    effective_split = _effective_split(experiment.dataset.split, split)
    effective_schema_mode = _effective_schema_mode(
        experiment.experiment.schema_mode,
        schema_mode,
    )

    model = load_model_groups(models)[model_group][run_id]
    dataset_root = experiment.dataset.local_path
    examples = normalize_examples_for_language(
        load_split(
            dataset_root,
            effective_split,
            test_file=experiment.dataset.test_file,
        ),
        language,
    )
    if limit is not None:
        examples = examples[:limit]

    template = PromptTemplate.from_path(language, experiment.experiment.prompt_files[language])
    requests: list[GenerationRequest] = []
    for example in examples:
        schema = serialize_example_schema(
            dataset_root,
            example,
            language,
            schema_mode=effective_schema_mode,
            test_db_dir=experiment.dataset.test_db_dir,
        )
        schema_provenance = schema_provenance_for_example(
            dataset_root,
            example,
            language,
            schema_mode=effective_schema_mode,
            test_db_dir=experiment.dataset.test_db_dir,
        )
        prompt = render_prompt(template, schema=schema, question=example["question"])
        requests.append(
            GenerationRequest(
                example_id=example["example_id"],
                split=example["split"],
                language=language,
                model_provider=model.provider,
                db_id=example["db_id"],
                question=example["question"],
                gold_sql=example["sql"],
                gold_query=example["gold_query"],
                schema_mode=effective_schema_mode,
                schema_provenance=schema_provenance,
                prompt=prompt.text,
            )
        )

    endpoint = experiment.endpoint
    api_key = _api_key_for_provider(endpoint.api_key_env, model.provider)
    if model.provider == "openai":
        client = OpenAIChatClient(
            OpenAIChatClientConfig(
                base_url=endpoint.base_url,
                api_key=api_key,
                request_timeout_seconds=endpoint.request_timeout_seconds,
                max_retries=endpoint.max_retries,
                retry_sleep_seconds=endpoint.retry_sleep_seconds,
            )
        )
    else:
        client = VllmClient(
            VllmClientConfig(
                base_url=endpoint.base_url,
                api_key=api_key,
                readiness_timeout_seconds=endpoint.readiness_timeout_seconds,
                request_timeout_seconds=endpoint.request_timeout_seconds,
                max_retries=endpoint.max_retries,
                retry_sleep_seconds=endpoint.retry_sleep_seconds,
            )
        )
    client.wait_until_ready(model.model_id)
    output_file = (
        experiment.experiment.output_root
        / effective_split
        / run_id
        / language
        / "predictions.jsonl"
    )
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
    split: Annotated[str | None, typer.Option()] = None,
    schema_mode: Annotated[str | None, typer.Option()] = None,
) -> None:
    experiment = load_experiment_config(config)
    effective_split = _effective_split(experiment.dataset.split, split)
    effective_schema_mode = _effective_schema_mode(
        experiment.experiment.schema_mode,
        schema_mode,
    )
    run_dir = experiment.experiment.output_root / effective_split / run_id / language
    score = evaluate_predictions(
        predictions_file=run_dir / "predictions.jsonl",
        dataset_root=experiment.dataset.local_path,
        run_id=run_id,
        language=language,
        output_file=run_dir / "scores.json",
        split=effective_split,
        schema_mode=effective_schema_mode,
        test_db_dir=experiment.dataset.test_db_dir,
    )
    typer.echo(
        f"{effective_split} {run_id} {language} "
        f"execution_accuracy={score['execution_accuracy']:.4f}"
    )


@app.command("report")
def report(
    split: Annotated[str, typer.Option()] = "test",
    runs_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path, typer.Option()] = Path("reports"),
) -> None:
    effective_split = _effective_split("test", split)
    effective_runs_dir = runs_dir or (Path("runs") / effective_split)
    frame = collect_scores(effective_runs_dir)
    write_reports(frame, output_dir, split=effective_split)
    typer.echo(f"Wrote reports to {output_dir}")


@app.command("validate-pipeline")
def validate_pipeline_command(
    config: Annotated[Path, typer.Option()] = Path("configs/experiment.yaml"),
    split: Annotated[str | None, typer.Option()] = None,
    schema_mode: Annotated[str | None, typer.Option()] = None,
    enforce_expected_counts: Annotated[bool, typer.Option()] = True,
) -> None:
    experiment = load_experiment_config(config)
    effective_split = _effective_split(experiment.dataset.split, split)
    effective_schema_mode = _effective_schema_mode(
        experiment.experiment.schema_mode,
        schema_mode,
    )
    try:
        result = validate_pipeline(
            experiment.dataset.local_path,
            split=effective_split,
            schema_mode=effective_schema_mode,
            test_file=experiment.dataset.test_file,
            test_db_dir=experiment.dataset.test_db_dir,
            enforce_expected_counts=enforce_expected_counts,
        )
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("extract-neo4j-schemas")
def extract_neo4j_schemas_command(
    config: Annotated[Path, typer.Option()] = Path("configs/experiment.yaml"),
    split: Annotated[str | None, typer.Option()] = None,
    neo4j_root: Annotated[Path, typer.Option()] = Path("docker/neo4j-root"),
    import_subfolder: Annotated[str | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
    wipe: Annotated[bool, typer.Option()] = True,
) -> None:
    experiment = load_experiment_config(config)
    effective_split = _effective_split(experiment.dataset.split, split)
    result = extract_neo4j_schemas(
        experiment.dataset.local_path,
        split=effective_split,
        neo4j_root=neo4j_root,
        test_file=experiment.dataset.test_file,
        test_db_dir=experiment.dataset.test_db_dir,
        import_subfolder=import_subfolder,
        overwrite=overwrite,
        wipe=wipe,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
