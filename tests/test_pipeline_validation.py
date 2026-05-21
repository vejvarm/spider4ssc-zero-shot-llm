import json
from pathlib import Path

import pytest

from spider4ssc_zeroshot.pipeline_validation import validate_pipeline


def _write_split(root: Path, split: str, *, db_id: str = "tiny_school") -> None:
    rows = [
        {
            "db_id": db_id,
            "question": "How many students are there?",
            "sql": "SELECT count(*) FROM student",
            "sparql": "SELECT ?count WHERE {}" if split == "dev" else "",
            "cypher": "MATCH (n) RETURN count(n)" if split == "dev" else "",
        }
    ]
    (root / ("test.json" if split == "test" else "dev.json")).write_text(
        json.dumps(rows),
        encoding="utf-8",
    )


def _write_db_artifacts(
    root: Path,
    split: str,
    *,
    db_id: str = "tiny_school",
    neo4j_schema: bool = True,
) -> None:
    db_root = root / ("database_test" if split == "test" else "database")
    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    (db_dir / f"{db_id}.sqlite").write_text("sqlite bytes", encoding="utf-8")
    (db_dir / f"{db_id}.ttl").write_text("@prefix : <urn:test/> .\n", encoding="utf-8")
    (db_dir / f"{db_id}.rdf-schema.json").write_text(
        json.dumps({"Classes": [], "Properties": {}}),
        encoding="utf-8",
    )
    if neo4j_schema:
        (db_dir / f"{db_id}.neo4j-schema.json").write_text(
            json.dumps({"NodeLabels": [], "NodeProperties": {}, "Relationships": []}),
            encoding="utf-8",
        )


def test_validate_pipeline_accepts_strict_dev_with_native_golds(tmp_path: Path):
    _write_split(tmp_path, "dev")
    _write_db_artifacts(tmp_path, "dev")

    result = validate_pipeline(
        tmp_path,
        split="dev",
        schema_mode="strict",
        enforce_expected_counts=False,
    )

    assert result["split"] == "dev"
    assert result["schema_mode"] == "strict"
    assert result["n_examples"] == 1
    assert result["n_databases"] == 1
    assert result["native_gold_queries"]["sparql"] == 1
    assert result["native_gold_queries"]["cypher"] == 1


def test_validate_pipeline_strict_rejects_missing_neo4j_schema(tmp_path: Path):
    _write_split(tmp_path, "test")
    _write_db_artifacts(tmp_path, "test", neo4j_schema=False)

    with pytest.raises(ValueError, match="missing strict Neo4j schema"):
        validate_pipeline(
            tmp_path,
            split="test",
            schema_mode="strict",
            enforce_expected_counts=False,
        )


def test_validate_pipeline_fallback_allows_missing_neo4j_schema(tmp_path: Path):
    _write_split(tmp_path, "test")
    _write_db_artifacts(tmp_path, "test", neo4j_schema=False)

    result = validate_pipeline(
        tmp_path,
        split="test",
        schema_mode="fallback",
        enforce_expected_counts=False,
    )

    assert result["schema_mode"] == "fallback"
    assert result["missing_neo4j_schemas"] == ["tiny_school"]
