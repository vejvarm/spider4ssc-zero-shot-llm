import json
from pathlib import Path

from spider4ssc_zeroshot.data import (
    compute_manifest,
    load_split,
    normalize_examples_for_language,
    required_dataset_paths,
)

FIXTURE = Path("tests/fixtures/tiny_spider4ssc")


def test_required_dataset_paths_for_test_split():
    paths = required_dataset_paths(FIXTURE, "test")

    assert FIXTURE / "test.json" in paths
    assert FIXTURE / "database_test" in paths


def test_load_split_assigns_example_ids():
    examples = load_split(FIXTURE, "test")

    assert [example.example_id for example in examples] == [0, 1]
    assert examples[0].db_id == "tiny_school"
    assert examples[0].gold_sql == "SELECT count(*) FROM student"


def test_normalize_examples_sets_target_language():
    examples = load_split(FIXTURE, "test")
    normalized = normalize_examples_for_language(examples, "sparql")

    assert normalized[0]["language"] == "sparql"
    assert normalized[0]["query"] == ""
    assert normalized[0]["sql"] == "SELECT count(*) FROM student"


def test_manifest_includes_sha256(tmp_path):
    file_path = tmp_path / "sample.json"
    file_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    manifest = compute_manifest(tmp_path)

    assert manifest["root"] == str(tmp_path)
    assert manifest["files"][0]["path"] == "sample.json"
    assert len(manifest["files"][0]["sha256"]) == 64
