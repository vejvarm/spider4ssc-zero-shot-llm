from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.metrics.spider.spider_test_suite import (
    compute_test_suite_metric as compute_sql_test_suite_metric,
)
from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.metrics.spidercypher.spider_test_suite import (
    compute_test_suite_metric as compute_cypher_test_suite_metric,
)
from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.metrics.spidersparql.spider_test_suite import (
    compute_test_suite_metric as compute_sparql_test_suite_metric,
)
from spider4ssc_zeroshot.vendor.ut5_ssc.third_party.spider.preprocess.get_tables import (
    dump_db_json_schema,
)

EVALUATOR_DESCRIPTION = (
    "Spider4SSC execution denotation; SPARQL/Cypher test use SQL-gold "
    "cross-language denotation"
)

MetricFunction = Callable[..., dict[str, Any]]


def load_prediction_rows(predictions_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with predictions_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metric_for_language(language: str) -> MetricFunction:
    if language == "sql":
        return compute_sql_test_suite_metric
    if language == "sparql":
        return compute_sparql_test_suite_metric
    if language == "cypher":
        return compute_cypher_test_suite_metric
    raise ValueError(f"Unsupported language: {language}")


def _sql_reference_schema(dataset_root: Path, db_id: str) -> dict[str, Any]:
    sqlite_path = dataset_root / "database_test" / db_id / f"{db_id}.sqlite"
    schema = dump_db_json_schema(str(sqlite_path), db_id)
    return {
        "db_table_names": schema["table_names_original"],
        "db_column_names": {
            "table_id": [table_id for table_id, _ in schema["column_names_original"]],
            "column_name": [
                column_name for _, column_name in schema["column_names_original"]
            ],
        },
        "db_foreign_keys": {
            "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
            "other_column_id": [
                other_column_id for _, other_column_id in schema["foreign_keys"]
            ],
        },
    }


def _build_references(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    language: str,
) -> list[dict[str, Any]]:
    schema_cache: dict[str, dict[str, Any]] = {}
    references = []
    for row in rows:
        db_id = row["db_id"]
        reference = {
            "db_id": db_id,
            "db_path": str(dataset_root / "database_test"),
            "query": row["gold_sql"] if language == "sql" else "",
            "sql": row["gold_sql"],
        }
        if language == "sql":
            if db_id not in schema_cache:
                schema_cache[db_id] = _sql_reference_schema(dataset_root, db_id)
            reference.update(schema_cache[db_id])
        references.append(reference)
    return references


def _compute_metric_without_side_effect_files(
    metric_fn: MetricFunction,
    predictions: list[str],
    references: list[dict[str, Any]],
    *,
    db_dir: Path,
    language: str,
) -> dict[str, Any]:
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as metric_cwd:
        try:
            os.chdir(metric_cwd)
            return metric_fn(
                predictions,
                references,
                db_dir=str(db_dir),
                lang=language,
            )
        finally:
            os.chdir(previous_cwd)


def evaluate_predictions(
    predictions_file: Path,
    dataset_root: Path,
    run_id: str,
    language: str,
    output_file: Path,
) -> dict[str, Any]:
    metric_fn = _metric_for_language(language)
    dataset_root = dataset_root.resolve()
    predictions_file = predictions_file.resolve()
    output_file = output_file.resolve()
    rows = load_prediction_rows(predictions_file)
    if not rows:
        raise ValueError(f"No prediction rows found in {predictions_file}")

    predictions = [row.get("prediction", "") for row in rows]
    references = _build_references(rows, dataset_root, language)

    metric = _compute_metric_without_side_effect_files(
        metric_fn,
        predictions,
        references,
        db_dir=dataset_root / "database_test",
        language=language,
    )

    scores = {
        "run_id": run_id,
        "model_id": rows[0].get("model_id", "unknown"),
        "model_revision": rows[0].get("model_revision", "unknown"),
        "split": "test",
        "language": language,
        "schema": "compact",
        "n_examples": len(rows),
        "execution_accuracy": float(metric["exec"]),
        "n_execution_errors": 0,
        "n_empty_predictions": sum(1 for prediction in predictions if not prediction.strip()),
        "evaluator": EVALUATOR_DESCRIPTION,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scores
