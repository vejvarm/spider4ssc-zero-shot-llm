from __future__ import annotations

import json
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


def evaluate_predictions(
    predictions_file: Path,
    dataset_root: Path,
    run_id: str,
    language: str,
    output_file: Path,
) -> dict[str, Any]:
    rows = load_prediction_rows(predictions_file)
    predictions = [row.get("prediction", "") for row in rows]
    references = [
        {
            "db_id": row["db_id"],
            "db_path": str(dataset_root / "database_test"),
            "query": row["gold_sql"] if language == "sql" else "",
            "sql": row["gold_sql"],
        }
        for row in rows
    ]

    metric_fn = _metric_for_language(language)
    metric = metric_fn(
        predictions,
        references,
        db_dir=str(dataset_root / "database_test"),
        lang=language,
    )

    scores = {
        "run_id": run_id,
        "model_id": rows[0].get("model_id", "unknown") if rows else "unknown",
        "model_revision": rows[0].get("model_revision", "unknown") if rows else "unknown",
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
