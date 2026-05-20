from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.utils.dataset import serialize_schema
from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.utils.neo4j_schema_extractor import (
    Neo4jSchemaExtractor,
    serialize_cypher_schema,
)
from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.utils.rdf_schema_extractor import (
    dump_kg_json_schema,
    serialize_sparql_schema,
)
from spider4ssc_zeroshot.vendor.ut5_ssc.third_party.spider.preprocess.get_tables import (
    dump_db_json_schema,
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_sparql_properties(properties: dict[str, Any]) -> dict[str, list[Any]]:
    if {"property", "domain", "range"}.issubset(properties):
        return properties

    normalized: dict[str, list[Any]] = {"property": [], "domain": [], "range": []}
    for property_name, metadata in properties.items():
        if not isinstance(metadata, dict):
            continue
        normalized["property"].append(property_name)
        normalized["domain"].append(_as_list(metadata.get("domain")))
        normalized["range"].append(_as_list(metadata.get("range")))
    return normalized


def _normalize_cypher_schema(schema: dict[str, Any]) -> dict[str, Any]:
    node_properties = schema.get("NodeProperties")
    if not isinstance(node_properties, dict):
        return schema

    normalized = dict(schema)
    normalized_properties: list[dict[str, Any]] = []
    for node_name, properties in node_properties.items():
        for property_entry in properties:
            normalized_entry = dict(property_entry)
            normalized_entry["nodeName"] = node_name
            normalized_properties.append(normalized_entry)
    normalized["NodeProperties"] = normalized_properties
    return normalized


def _db_root(dataset_root: Path, split: str) -> Path:
    return dataset_root / ("database_test" if split == "test" else "database")


def _column_names(schema: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "table_id": [table_id for table_id, _ in schema["column_names_original"]],
        "column_name": [column_name for _, column_name in schema["column_names_original"]],
    }


def _foreign_keys(schema: dict[str, Any]) -> dict[str, list[int]]:
    return {
        "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
        "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]],
    }


def _load_cypher_schema(db_root: Path, db_id: str) -> dict[str, Any]:
    schema_path = db_root / db_id / f"{db_id}.neo4j-schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema.setdefault("db_id", db_id)
        return schema
    return Neo4jSchemaExtractor(db_root=db_root).dump_neo4j_schema(
        db=db_root / db_id / f"{db_id}.ttl",
        f=db_id,
    )


def serialize_example_schema(dataset_root: Path, example: dict[str, Any], language: str) -> str:
    split = example["split"]
    db_id = example["db_id"]
    db_root = _db_root(dataset_root, split)

    if language == "sql":
        sqlite_path = db_root / db_id / f"{db_id}.sqlite"
        schema = dump_db_json_schema(str(sqlite_path), db_id)
        return serialize_schema(
            question=example["question"],
            db_path=str(db_root),
            db_id=db_id,
            db_column_names=_column_names(schema),
            db_table_names=schema["table_names_original"],
            db_column_types=schema["column_types"],
            db_primary_keys={"column_id": schema["primary_keys"]},
            db_foreign_keys=_foreign_keys(schema),
            schema_serialization_type="compact",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=False,
            normalize_query=True,
        )

    if language == "sparql":
        schema = dump_kg_json_schema(db_root / db_id / f"{db_id}.ttl", db_id)
        return serialize_sparql_schema(
            question=example["question"],
            db_path=str(db_root),
            db_id=db_id,
            classes=schema.get("Classes", []),
            properties=_normalize_sparql_properties(schema.get("Properties", {})),
            schema_serialization_type="compact",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=False,
        )

    if language == "cypher":
        return serialize_cypher_schema(
            question=example["question"],
            db_path=str(db_root),
            db_id=db_id,
            schema=_normalize_cypher_schema(_load_cypher_schema(db_root, db_id)),
            schema_serialization_type="compact",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=False,
        )

    raise ValueError(f"Unsupported language: {language}")
