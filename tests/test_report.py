import json
from pathlib import Path

import pandas as pd

from spider4ssc_zeroshot.report import (
    LANGUAGE_ORDER,
    LANGUAGE_PALETTE,
    collect_scores,
    write_reports,
)


def _write_score(path: Path, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score = {
        "run_id": "run-b",
        "model_id": "test-model",
        "model_revision": "rev",
        "split": "test",
        "language": "cypher",
        "schema": "compact",
        "schema_mode": "strict",
        "n_examples": 2,
        "execution_accuracy": 0.25,
        "n_execution_errors": 0,
        "n_empty_predictions": 1,
        "evaluator": "fake",
    }
    score.update(overrides)
    path.write_text(json.dumps(score), encoding="utf-8")


def test_language_palette_matches_ut5_ssc_scheme():
    assert LANGUAGE_PALETTE == {
        "sql": "#B3A369",
        "sparql": "#367040",
        "cypher": "#5C7596",
    }


def test_language_order_matches_paper_scheme():
    assert LANGUAGE_ORDER == ["sparql", "sql", "cypher"]


def test_collect_scores_orders_runs_and_languages(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _write_score(
        runs_dir / "run-b" / "cypher" / "scores.json",
        run_id="run-b",
        language="cypher",
    )
    _write_score(
        runs_dir / "run-a" / "sql" / "scores.json",
        run_id="run-a",
        language="sql",
        execution_accuracy=1.0,
        n_empty_predictions=0,
    )

    frame = collect_scores(runs_dir)

    assert list(frame["run_id"]) == ["run-a", "run-b"]
    assert list(frame["language"].astype(str)) == ["sql", "cypher"]


def test_collect_scores_empty_frame_has_report_columns(tmp_path: Path):
    frame = collect_scores(tmp_path / "missing")

    assert list(frame.columns) == [
        "run_id",
        "model_id",
        "split",
        "language",
        "schema_mode",
        "execution_accuracy",
        "n_examples",
        "n_empty_predictions",
    ]
    assert frame.empty


def test_write_reports_writes_csv_markdown_latex_and_failures(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "run_id": "run-a",
                "model_id": "model-a",
                "split": "dev",
                "language": "sql",
                "schema_mode": "strict",
                "execution_accuracy": 1.0,
                "n_examples": 2,
                "n_empty_predictions": 0,
            },
            {
                "run_id": "run-b",
                "model_id": "model-b",
                "split": "dev",
                "language": "sparql",
                "schema_mode": "fallback",
                "execution_accuracy": 0.5,
                "n_examples": 2,
                "n_empty_predictions": 1,
            },
        ]
    )
    output_dir = tmp_path / "reports"

    write_reports(frame, output_dir, split="dev")

    csv_text = (output_dir / "dev_main_matrix.csv").read_text(encoding="utf-8")
    md_text = (output_dir / "dev_main_matrix.md").read_text(encoding="utf-8")
    tex_text = (output_dir / "dev_main_matrix.tex").read_text(encoding="utf-8")
    failures = pd.read_csv(output_dir / "dev_main_matrix_failures.csv")

    assert "run-a,model-a,dev,sql,strict,1.0,2,0" in csv_text
    assert "| run_id" in md_text
    assert "run-b" in md_text
    assert "\\begin{tabular}" in tex_text
    assert list(failures["run_id"]) == ["run-b"]


def test_write_reports_writes_execution_and_empty_prediction_plots(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "run_id": "model-a",
                "model_id": "model/a",
                "split": "dev",
                "language": "sql",
                "schema_mode": "strict",
                "execution_accuracy": 0.75,
                "n_examples": 100,
                "n_empty_predictions": 0,
            },
            {
                "run_id": "model-a",
                "model_id": "model/a",
                "split": "dev",
                "language": "sparql",
                "schema_mode": "strict",
                "execution_accuracy": 0.25,
                "n_examples": 100,
                "n_empty_predictions": 5,
            },
            {
                "run_id": "model-a",
                "model_id": "model/a",
                "split": "dev",
                "language": "cypher",
                "schema_mode": "strict",
                "execution_accuracy": 0.5,
                "n_examples": 100,
                "n_empty_predictions": 10,
            },
            {
                "run_id": "model-b",
                "model_id": "model/b",
                "split": "dev",
                "language": "sql",
                "schema_mode": "strict",
                "execution_accuracy": 0.8,
                "n_examples": 100,
                "n_empty_predictions": 1,
            },
            {
                "run_id": "model-b",
                "model_id": "model/b",
                "split": "dev",
                "language": "sparql",
                "schema_mode": "strict",
                "execution_accuracy": 0.1,
                "n_examples": 100,
                "n_empty_predictions": 2,
            },
            {
                "run_id": "model-b",
                "model_id": "model/b",
                "split": "dev",
                "language": "cypher",
                "schema_mode": "strict",
                "execution_accuracy": 0.3,
                "n_examples": 100,
                "n_empty_predictions": 3,
            },
        ]
    )
    output_dir = tmp_path / "reports"

    write_reports(frame, output_dir, split="dev")

    for stem in (
        "dev_execution_accuracy_by_model",
        "dev_empty_prediction_rate_by_model",
    ):
        for suffix in (".pdf", ".png"):
            plot_file = output_dir / f"{stem}{suffix}"
            assert plot_file.exists()
            assert plot_file.stat().st_size > 0


def test_write_reports_plots_missing_language_combinations_without_zero_fill(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "run_id": "model-a",
                "model_id": "model/a",
                "split": "dev",
                "language": "sql",
                "schema_mode": "strict",
                "execution_accuracy": 0.75,
                "n_examples": 100,
                "n_empty_predictions": 0,
            },
            {
                "run_id": "model-b",
                "model_id": "model/b",
                "split": "dev",
                "language": "cypher",
                "schema_mode": "strict",
                "execution_accuracy": 0.3,
                "n_examples": 100,
                "n_empty_predictions": 3,
            },
        ]
    )
    output_dir = tmp_path / "reports"

    write_reports(frame, output_dir, split="dev")

    assert (output_dir / "dev_execution_accuracy_by_model.pdf").stat().st_size > 0
    csv_text = (output_dir / "dev_main_matrix.csv").read_text(encoding="utf-8")
    assert "model-a,model/a,dev,sql,strict,0.75,100,0" in csv_text
    assert "model-b,model/b,dev,cypher,strict,0.3,100,3" in csv_text
    assert "model-a,model/a,dev,sparql,strict,0.0" not in csv_text


def test_write_reports_skips_plots_for_empty_frame(tmp_path: Path):
    output_dir = tmp_path / "reports"

    write_reports(pd.DataFrame(columns=["run_id", "language"]), output_dir, split="dev")

    assert not (output_dir / "dev_execution_accuracy_by_model.pdf").exists()
    assert not (output_dir / "dev_execution_accuracy_by_model.png").exists()
    assert not (output_dir / "dev_empty_prediction_rate_by_model.pdf").exists()
    assert not (output_dir / "dev_empty_prediction_rate_by_model.png").exists()
