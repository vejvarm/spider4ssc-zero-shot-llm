import json
import tarfile
from pathlib import Path

import pytest

from spider4ssc_zeroshot.data import (
    _safe_extract_tar,
    compute_manifest,
    ensure_dataset,
    load_split,
    normalize_examples_for_language,
    required_dataset_paths,
)

FIXTURE = Path("tests/fixtures/tiny_spider4ssc")


def test_required_dataset_paths_for_test_split():
    paths = required_dataset_paths(FIXTURE, "test")

    assert FIXTURE / "test.json" in paths
    assert FIXTURE / "database_test" in paths


def test_required_dataset_paths_honors_test_config():
    paths = required_dataset_paths(
        FIXTURE,
        "test",
        test_file="custom_test.json",
        test_db_dir="custom_database",
    )

    assert paths == [FIXTURE / "custom_test.json", FIXTURE / "custom_database"]


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


def test_load_split_rejects_non_list_json(tmp_path):
    split_file = tmp_path / "test.json"
    split_file.write_text(json.dumps({"db_id": "tiny_school"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a list of examples"):
        load_split(tmp_path, "test")


def test_load_split_rejects_missing_required_field(tmp_path):
    split_file = tmp_path / "test.json"
    split_file.write_text(
        json.dumps([{"question": "How many?", "sql": "SELECT 1"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Row 0 missing required field db_id"):
        load_split(tmp_path, "test")


def test_load_split_rejects_non_string_sql(tmp_path):
    split_file = tmp_path / "test.json"
    split_file.write_text(
        json.dumps(
            [
                {
                    "db_id": "tiny_school",
                    "question": "How many?",
                    "sql": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Row 0 field sql must be a string"):
        load_split(tmp_path, "test")


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.tgz"
    destination = tmp_path / "destination"
    destination.mkdir()

    with tarfile.open(archive_path, "w:gz") as archive:
        payload = tmp_path / "payload.txt"
        payload.write_text("evil", encoding="utf-8")
        archive.add(payload, arcname="../evil.txt")

    with pytest.raises(ValueError, match="Unsafe tar member"):
        _safe_extract_tar(archive_path, destination)

    assert not (tmp_path / "evil.txt").exists()


def test_ensure_dataset_rejects_malformed_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "Spider4SSC"

    with pytest.raises(FileNotFoundError, match="missing required path"):
        ensure_dataset(output, source=source, url=None)
