import json
from pathlib import Path

import pytest

from spider4ssc_zeroshot import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )


def test_load_prediction_rows_skips_blank_lines(tmp_path: Path):
    predictions_file = tmp_path / "predictions.jsonl"
    predictions_file.write_text('{"example_id": 1}\n\n{"example_id": 2}\n', encoding="utf-8")

    assert evaluate.load_prediction_rows(predictions_file) == [
        {"example_id": 1},
        {"example_id": 2},
    ]


def test_evaluate_predictions_writes_scores_and_uses_metric(monkeypatch, tmp_path: Path):
    predictions_file = tmp_path / "predictions" / "sql.jsonl"
    dataset_root = tmp_path / "dataset"
    output_file = tmp_path / "runs" / "run-1" / "sql" / "scores.json"
    _write_jsonl(
        predictions_file,
        [
            {
                "example_id": 1,
                "language": "sql",
                "db_id": "concert_singer",
                "gold_sql": "SELECT count(*) FROM singer",
                "prediction": "SELECT count(*) FROM singer",
                "model_id": "model-a",
                "model_revision": "rev-a",
            },
            {
                "example_id": 2,
                "language": "sql",
                "db_id": "concert_singer",
                "gold_sql": "SELECT name FROM singer",
                "prediction": "  ",
                "model_id": "model-a",
                "model_revision": "rev-a",
            },
        ],
    )
    calls = []

    def fake_metric(predictions, references, db_dir=None, lang=None):
        calls.append(
            {
                "predictions": predictions,
                "references": references,
                "db_dir": db_dir,
                "lang": lang,
            }
        )
        return {"exec": 0.5}

    monkeypatch.setattr(evaluate, "compute_sql_test_suite_metric", fake_metric)

    scores = evaluate.evaluate_predictions(
        predictions_file=predictions_file,
        dataset_root=dataset_root,
        run_id="run-1",
        language="sql",
        output_file=output_file,
    )

    assert calls == [
        {
            "predictions": ["SELECT count(*) FROM singer", "  "],
            "references": [
                {
                    "db_id": "concert_singer",
                    "db_path": str(dataset_root / "database_test"),
                    "query": "SELECT count(*) FROM singer",
                    "sql": "SELECT count(*) FROM singer",
                },
                {
                    "db_id": "concert_singer",
                    "db_path": str(dataset_root / "database_test"),
                    "query": "SELECT name FROM singer",
                    "sql": "SELECT name FROM singer",
                },
            ],
            "db_dir": str(dataset_root / "database_test"),
            "lang": "sql",
        }
    ]
    assert scores == {
        "run_id": "run-1",
        "model_id": "model-a",
        "model_revision": "rev-a",
        "split": "test",
        "language": "sql",
        "schema": "compact",
        "n_examples": 2,
        "execution_accuracy": 0.5,
        "n_execution_errors": 0,
        "n_empty_predictions": 1,
        "evaluator": evaluate.EVALUATOR_DESCRIPTION,
    }
    assert json.loads(output_file.read_text(encoding="utf-8")) == scores


def test_evaluate_predictions_uses_empty_query_for_cross_language(
    monkeypatch,
    tmp_path: Path,
):
    predictions_file = tmp_path / "predictions" / "sparql.jsonl"
    dataset_root = tmp_path / "dataset"
    _write_jsonl(
        predictions_file,
        [
            {
                "db_id": "pets",
                "gold_sql": "SELECT name FROM pets",
                "prediction": "SELECT ?name WHERE {}",
                "model_id": "model-a",
                "model_revision": "rev-a",
            }
        ],
    )
    references_seen = []

    def fake_metric(predictions, references, db_dir=None, lang=None):
        references_seen.extend(references)
        return {"exec": 1}

    monkeypatch.setattr(evaluate, "compute_sparql_test_suite_metric", fake_metric)

    evaluate.evaluate_predictions(
        predictions_file=predictions_file,
        dataset_root=dataset_root,
        run_id="run-1",
        language="sparql",
        output_file=tmp_path / "scores.json",
    )

    assert references_seen[0]["query"] == ""
    assert references_seen[0]["sql"] == "SELECT name FROM pets"


def test_evaluate_predictions_rejects_unsupported_language(tmp_path: Path):
    predictions_file = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_file, [])

    with pytest.raises(ValueError, match="Unsupported language"):
        evaluate.evaluate_predictions(
            predictions_file=predictions_file,
            dataset_root=tmp_path,
            run_id="run-1",
            language="graphql",
            output_file=tmp_path / "scores.json",
        )
