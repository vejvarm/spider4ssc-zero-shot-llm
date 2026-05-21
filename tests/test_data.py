import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from spider4ssc_zeroshot.data import (
    _safe_extract_tar,
    _verify_sha256,
    compute_manifest,
    ensure_dataset,
    load_split,
    normalize_examples_for_language,
    required_dataset_paths,
)

FIXTURE = Path("tests/fixtures/tiny_spider4ssc")


def _make_dataset_archive(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "archive_src" / "Spider4SSC"
    (dataset_root / "database_test").mkdir(parents=True)
    (dataset_root / "test.json").write_text("[]", encoding="utf-8")
    (dataset_root / "database_test" / ".gitkeep").write_text("", encoding="utf-8")
    archive_path = tmp_path / "Spider4SSC.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(dataset_root, arcname="Spider4SSC")
    return archive_path


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


def test_load_split_honors_custom_split_file(tmp_path):
    custom_file = tmp_path / "custom_test.json"
    custom_file.write_text(
        json.dumps(
            [
                {
                    "db_id": "tiny_school",
                    "question": "How many?",
                    "sql": "SELECT 1",
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = load_split(tmp_path, "test", split_file="custom_test.json")

    assert len(examples) == 1
    assert examples[0].gold_sql == "SELECT 1"


def test_normalize_examples_sets_target_language():
    examples = load_split(FIXTURE, "test")
    normalized = normalize_examples_for_language(examples, "sparql")

    assert normalized[0]["language"] == "sparql"
    assert normalized[0]["gold_query"] == ""
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


@pytest.mark.parametrize(
    ("member_type", "linkname"),
    [
        (tarfile.SYMTYPE, "/tmp/target"),
        (tarfile.LNKTYPE, "/tmp/target"),
        (tarfile.FIFOTYPE, ""),
    ],
)
def test_safe_extract_rejects_links_and_special_members(tmp_path, member_type, linkname):
    archive_path = tmp_path / "malicious.tgz"
    destination = tmp_path / "destination"
    destination.mkdir()

    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("Spider4SSC/unsafe")
        info.type = member_type
        info.linkname = linkname
        archive.addfile(info)

    with pytest.raises(ValueError, match="Unsafe tar member"):
        _safe_extract_tar(archive_path, destination)


def test_ensure_dataset_rejects_malformed_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "Spider4SSC"

    with pytest.raises(FileNotFoundError, match="invalid required path"):
        ensure_dataset(output, source=source, url=None)


def test_ensure_dataset_rejects_wrong_required_path_types(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "test.json").mkdir()
    (source / "database_test").write_text("not a directory", encoding="utf-8")
    output = tmp_path / "Spider4SSC"

    with pytest.raises(FileNotFoundError, match="expected file"):
        ensure_dataset(output, source=source, url=None)

    assert not output.exists()


def test_ensure_dataset_honors_custom_source_layout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "custom_test.json").write_text("[]", encoding="utf-8")
    (source / "custom_database").mkdir()
    output = tmp_path / "Spider4SSC"

    ensure_dataset(
        output,
        source=source,
        url=None,
        test_file="custom_test.json",
        test_db_dir="custom_database",
    )

    assert (output / "custom_test.json").exists()
    assert (output / "custom_database").is_dir()


def test_ensure_dataset_honors_non_test_split_layout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "dev.json").write_text("[]", encoding="utf-8")
    (source / "database").mkdir()
    output = tmp_path / "Spider4SSC"

    ensure_dataset(output, source=source, url=None, split="dev")

    assert (output / "dev.json").exists()
    assert (output / "database").is_dir()


def test_verify_sha256_requires_expected_value(tmp_path):
    archive_path = tmp_path / "Spider4SSC.tgz"
    archive_path.write_text("archive", encoding="utf-8")

    with pytest.raises(ValueError, match="archive_sha256 is required"):
        _verify_sha256(archive_path, None)


def test_verify_sha256_rejects_mismatch(tmp_path):
    archive_path = tmp_path / "Spider4SSC.tgz"
    archive_path.write_text("archive", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _verify_sha256(archive_path, "0" * 64)


def test_ensure_dataset_download_requires_checksum_before_network(tmp_path, monkeypatch):
    def fail_download(url: str, output_path: Path) -> None:
        raise AssertionError("download should not start without archive_sha256")

    monkeypatch.setattr("spider4ssc_zeroshot.data._download_file", fail_download)

    with pytest.raises(ValueError, match="archive_sha256 is required"):
        ensure_dataset(
            tmp_path / "Spider4SSC",
            source=None,
            url="https://example.org/Spider4SSC.tgz",
            archive_sha256=None,
        )


def test_ensure_dataset_download_rejects_mismatched_checksum(tmp_path, monkeypatch):
    archive_path = _make_dataset_archive(tmp_path)

    def fake_download(url: str, output_path: Path) -> None:
        shutil.copyfile(archive_path, output_path)

    monkeypatch.setattr("spider4ssc_zeroshot.data._download_file", fake_download)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ensure_dataset(
            tmp_path / "Spider4SSC",
            source=None,
            url="https://example.org/Spider4SSC.tgz",
            archive_sha256="0" * 64,
        )


def test_ensure_dataset_download_accepts_matching_checksum(tmp_path, monkeypatch):
    archive_path = _make_dataset_archive(tmp_path)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    def fake_download(url: str, output_path: Path) -> None:
        shutil.copyfile(archive_path, output_path)

    monkeypatch.setattr("spider4ssc_zeroshot.data._download_file", fake_download)

    output = tmp_path / "Spider4SSC"
    ensure_dataset(
        output,
        source=None,
        url="https://example.org/Spider4SSC.tgz",
        archive_sha256=archive_sha256,
    )

    assert (output / "test.json").read_text(encoding="utf-8") == "[]"
    assert (output / "database_test" / ".gitkeep").exists()
