import json
from pathlib import Path

import pandas as pd

from spider4ssc_zeroshot.report import collect_scores, write_reports


def _write_score(path: Path, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score = {
        "run_id": "run-b",
        "model_id": "test-model",
        "model_revision": "rev",
        "split": "test",
        "language": "cypher",
        "schema": "compact",
        "n_examples": 2,
        "execution_accuracy": 0.25,
        "n_execution_errors": 0,
        "n_empty_predictions": 1,
        "evaluator": "fake",
    }
    score.update(overrides)
    path.write_text(json.dumps(score), encoding="utf-8")


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
        "language",
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
                "language": "sql",
                "execution_accuracy": 1.0,
                "n_examples": 2,
                "n_empty_predictions": 0,
            },
            {
                "run_id": "run-b",
                "model_id": "model-b",
                "language": "sparql",
                "execution_accuracy": 0.5,
                "n_examples": 2,
                "n_empty_predictions": 1,
            },
        ]
    )
    output_dir = tmp_path / "reports"

    write_reports(frame, output_dir)

    csv_text = (output_dir / "test_main_matrix.csv").read_text(encoding="utf-8")
    md_text = (output_dir / "test_main_matrix.md").read_text(encoding="utf-8")
    tex_text = (output_dir / "test_main_matrix.tex").read_text(encoding="utf-8")
    failures = pd.read_csv(output_dir / "test_main_matrix_failures.csv")

    assert "run-a,model-a,sql,1.0,2,0" in csv_text
    assert "| run_id" in md_text
    assert "run-b" in md_text
    assert "\\begin{tabular}" in tex_text
    assert list(failures["run_id"]) == ["run-b"]
