from __future__ import annotations

from pathlib import Path
from typing import Any

from spider4ssc_zeroshot.data import (
    load_split,
    split_db_root,
    split_file_path,
    validate_split,
)

EXPECTED_SPLIT_COUNTS = {
    "test": {"examples": 2147, "databases": 40},
    "dev": {"examples": 608, "databases": 20},
}
SCHEMA_MODES = {"strict", "fallback"}


def _validate_schema_mode(schema_mode: str) -> str:
    if schema_mode not in SCHEMA_MODES:
        supported = ", ".join(sorted(SCHEMA_MODES))
        raise ValueError(f"Unsupported schema_mode: {schema_mode}. Supported: {supported}")
    return schema_mode


def _missing_artifacts(db_root: Path, db_id: str, suffixes: list[str]) -> list[Path]:
    db_dir = db_root / db_id
    missing = []
    for suffix in suffixes:
        path = db_dir / f"{db_id}{suffix}"
        if not path.is_file():
            missing.append(path)
    return missing


def _count_native_golds(examples: list[Any]) -> dict[str, int]:
    return {
        "sql": sum(1 for example in examples if example.gold_sql),
        "sparql": sum(1 for example in examples if example.gold_sparql),
        "cypher": sum(1 for example in examples if example.gold_cypher),
    }


def validate_pipeline(
    dataset_root: Path,
    *,
    split: str,
    schema_mode: str,
    test_file: str = "test.json",
    test_db_dir: str = "database_test",
    enforce_expected_counts: bool = True,
) -> dict[str, Any]:
    split = validate_split(split)
    schema_mode = _validate_schema_mode(schema_mode)
    dataset_root = dataset_root.resolve()
    split_path = split_file_path(dataset_root, split, test_file=test_file)
    db_root = split_db_root(dataset_root, split, test_db_dir=test_db_dir)
    errors: list[str] = []

    if not split_path.is_file():
        errors.append(f"missing split file: {split_path}")
    if not db_root.is_dir():
        errors.append(f"missing database directory: {db_root}")
    if errors:
        raise ValueError("Pipeline validation failed:\n- " + "\n- ".join(errors))

    examples = load_split(dataset_root, split, test_file=test_file)
    db_ids = sorted({example.db_id for example in examples})
    native_gold_queries = _count_native_golds(examples)

    if enforce_expected_counts:
        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(examples) != expected["examples"]:
            errors.append(
                f"unexpected {split} row count: {len(examples)} "
                f"(expected {expected['examples']})"
            )
        if len(db_ids) != expected["databases"]:
            errors.append(
                f"unexpected {split} database count: {len(db_ids)} "
                f"(expected {expected['databases']})"
            )

    if split == "dev":
        for language in ["sparql", "cypher"]:
            if native_gold_queries[language] != len(examples):
                errors.append(
                    f"dev split requires native {language} gold queries for every row: "
                    f"{native_gold_queries[language]}/{len(examples)} present"
                )

    missing_required: list[str] = []
    missing_neo4j_schemas: list[str] = []
    for db_id in db_ids:
        base_missing = _missing_artifacts(
            db_root,
            db_id,
            [".sqlite", ".ttl", ".rdf-schema.json"],
        )
        missing_required.extend(str(path) for path in base_missing)
        neo4j_schema = db_root / db_id / f"{db_id}.neo4j-schema.json"
        if not neo4j_schema.is_file():
            missing_neo4j_schemas.append(db_id)
            if schema_mode == "strict":
                missing_required.append(
                    f"{neo4j_schema} (missing strict Neo4j schema)"
                )

    if missing_required:
        errors.extend(f"missing artifact: {path}" for path in missing_required)
    if errors:
        raise ValueError("Pipeline validation failed:\n- " + "\n- ".join(errors))

    return {
        "dataset_root": str(dataset_root),
        "split": split,
        "split_file": str(split_path),
        "database_root": str(db_root),
        "schema_mode": schema_mode,
        "n_examples": len(examples),
        "n_databases": len(db_ids),
        "native_gold_queries": native_gold_queries,
        "missing_neo4j_schemas": missing_neo4j_schemas,
        "schema_artifacts": {
            "sqlite": len(db_ids),
            "ttl": len(db_ids),
            "rdf_schema_json": len(db_ids),
            "neo4j_schema_json": len(db_ids) - len(missing_neo4j_schemas),
        },
    }
