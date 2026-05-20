from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

SUPPORTED_LANGUAGES = {"sql", "sparql", "cypher"}


@dataclass(frozen=True)
class SpiderExample:
    example_id: int
    split: str
    db_id: str
    question: str
    gold_sql: str
    gold_sparql: str
    gold_cypher: str


def required_dataset_paths(
    root: Path,
    split: str,
    *,
    test_file: str = "test.json",
    test_db_dir: str = "database_test",
) -> list[Path]:
    if split != "test":
        return [root / f"{split}.json", root / "database"]
    return [root / test_file, root / test_db_dir]


def _missing_required_paths(
    root: Path,
    split: str,
    *,
    test_file: str = "test.json",
    test_db_dir: str = "database_test",
) -> list[Path]:
    return [
        path
        for path in required_dataset_paths(
            root,
            split,
            test_file=test_file,
            test_db_dir=test_db_dir,
        )
        if not path.exists()
    ]


def _validate_required_paths(
    root: Path,
    split: str,
    *,
    test_file: str = "test.json",
    test_db_dir: str = "database_test",
) -> None:
    missing = _missing_required_paths(
        root,
        split,
        test_file=test_file,
        test_db_dir=test_db_dir,
    )
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Spider4SSC dataset is missing required path(s): {missing_text}")


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            parts = member.name.split("/")
            target_path = destination / member.name
            if (
                member_path.is_absolute()
                or ".." in parts
                or member.issym()
                or member.islnk()
                or member.ischr()
                or member.isblk()
                or member.isfifo()
                or not _is_relative_to(target_path, destination)
            ):
                raise ValueError(f"Unsafe tar member: {member.name}")
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=output_path.parent,
        ) as handle:
            temp_name = handle.name
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        Path(temp_name).replace(output_path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _verify_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual_sha256 = _sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def _promote_dataset(prepared_root: Path, root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if root.exists():
        backup_path = Path(
            tempfile.mkdtemp(prefix=f".{root.name}.backup-", dir=root.parent)
        )
        backup_path.rmdir()
        shutil.move(str(root), str(backup_path))
    try:
        shutil.move(str(prepared_root), str(root))
    except Exception:
        if backup_path is not None and backup_path.exists() and not root.exists():
            shutil.move(str(backup_path), str(root))
        raise
    if backup_path is not None and backup_path.exists():
        shutil.rmtree(backup_path)


def ensure_dataset(
    root: Path,
    *,
    source: Path | None,
    url: str | None,
    split: str = "test",
    test_file: str = "test.json",
    test_db_dir: str = "database_test",
    archive_sha256: str | None = None,
) -> None:
    if all(
        path.exists()
        for path in required_dataset_paths(
            root,
            split,
            test_file=test_file,
            test_db_dir=test_db_dir,
        )
    ):
        return

    root.parent.mkdir(parents=True, exist_ok=True)

    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source}")
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(source, root)
        _validate_required_paths(
            root,
            split,
            test_file=test_file,
            test_db_dir=test_db_dir,
        )
        return

    if url is None:
        raise ValueError("Either source or url must be provided")
    with tempfile.TemporaryDirectory(dir=root.parent) as staging_dir:
        staging = Path(staging_dir)
        archive_path = staging / "Spider4SSC.tgz"
        extracted = staging / "extracted"
        _download_file(url, archive_path)
        _verify_sha256(archive_path, archive_sha256)
        _safe_extract_tar(archive_path, extracted)
        prepared_root = extracted / "Spider4SSC"
        if not prepared_root.exists():
            raise FileNotFoundError(
                f"Archive did not create expected dataset root: {prepared_root}"
            )
        _validate_required_paths(
            prepared_root,
            split,
            test_file=test_file,
            test_db_dir=test_db_dir,
        )
        _promote_dataset(prepared_root, root)


def _required_string(row: dict, row_index: int, field: str) -> str:
    if field not in row:
        raise ValueError(f"Row {row_index} missing required field {field}")
    value = row[field]
    if not isinstance(value, str):
        raise ValueError(f"Row {row_index} field {field} must be a string")
    return value


def _optional_string(row: dict, row_index: int, field: str) -> str:
    value = row.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Row {row_index} field {field} must be a string or null")
    return value


def load_split(
    root: Path,
    split: str,
    *,
    split_file: str | Path | None = None,
) -> list[SpiderExample]:
    split_path = root / (split_file or f"{split}.json")
    with split_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of examples in {split_path}")
    examples: list[SpiderExample] = []
    for example_id, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {example_id} must be an object")
        examples.append(
            SpiderExample(
                example_id=example_id,
                split=split,
                db_id=_required_string(row, example_id, "db_id"),
                question=_required_string(row, example_id, "question"),
                gold_sql=_required_string(row, example_id, "sql"),
                gold_sparql=_optional_string(row, example_id, "sparql"),
                gold_cypher=_optional_string(row, example_id, "cypher"),
            )
        )
    return examples


def normalize_examples_for_language(examples: list[SpiderExample], language: str) -> list[dict]:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    normalized: list[dict] = []
    for example in examples:
        language_gold = {
            "sql": example.gold_sql,
            "sparql": example.gold_sparql,
            "cypher": example.gold_cypher,
        }[language]
        normalized.append(
            {
                "example_id": example.example_id,
                "split": example.split,
                "language": language,
                "lang": language,
                "db_id": example.db_id,
                "question": example.question,
                "sql": example.gold_sql,
                "query": language_gold,
            }
        )
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_manifest(root: Path) -> dict:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"root": str(root), "files": files}


def write_manifest(root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(compute_manifest(root), handle, indent=2, sort_keys=True)
        handle.write("\n")
