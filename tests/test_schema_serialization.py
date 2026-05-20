import json
import sqlite3

import pytest

from spider4ssc_zeroshot.schema_serialization import serialize_example_schema


def _example() -> dict[str, str]:
    return {
        "split": "test",
        "db_id": "tiny_school",
        "question": "How many students are there?",
    }


def test_serialize_sql_schema_from_sqlite(tmp_path):
    db_dir = tmp_path / "database_test" / "tiny_school"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "tiny_school.sqlite")
    connection.execute("CREATE TABLE student (id INTEGER PRIMARY KEY, name TEXT)")
    connection.commit()
    connection.close()

    serialized = serialize_example_schema(tmp_path, _example(), "sql")

    assert serialized == " | tiny_school | student : id (number) , name (text)"


def test_serialize_sparql_schema_from_cached_json(tmp_path):
    db_dir = tmp_path / "database_test" / "tiny_school"
    db_dir.mkdir(parents=True)
    (db_dir / "tiny_school.rdf-schema.json").write_text(
        json.dumps(
            {
                "Classes": ["Student"],
                "Properties": {
                    "name": {
                        "domain": ["Student"],
                        "range": ["string"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    serialized = serialize_example_schema(tmp_path, _example(), "sparql")

    assert serialized == " | tiny_school | Student: name (string)"


def test_serialize_cypher_schema_from_cached_json(tmp_path):
    db_dir = tmp_path / "database_test" / "tiny_school"
    db_dir.mkdir(parents=True)
    (db_dir / "tiny_school.neo4j-schema.json").write_text(
        json.dumps(
            {
                "NodeLabels": ["Student"],
                "NodeProperties": {
                    "Student": [{"propertyName": "name", "propertyTypes": ["String"]}]
                },
                "Relationships": [],
            }
        ),
        encoding="utf-8",
    )

    serialized = serialize_example_schema(tmp_path, _example(), "cypher")

    assert serialized == " | tiny_school | Student: name (String)"


def test_serialize_example_schema_rejects_unsupported_language(tmp_path):
    with pytest.raises(ValueError, match="Unsupported language"):
        serialize_example_schema(tmp_path, _example(), "gremlin")
