from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
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


def required_dataset_paths(root: Path, split: str) -> list[Path]:
    if split != "test":
        return [root / f"{split}.json", root / "database"]
    return [root / "test.json", root / "database_test"]


def ensure_dataset(root: Path, *, source: Path | None, url: str | None) -> None:
    if all(path.exists() for path in required_dataset_paths(root, "test")):
        return

    root.parent.mkdir(parents=True, exist_ok=True)

    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source}")
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(source, root)
        return

    if url is None:
        raise ValueError("Either source or url must be provided")

    archive_path = root.parent / "Spider4SSC.tgz"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(root.parent)

    if not root.exists():
        raise FileNotFoundError(f"Archive did not create expected dataset root: {root}")


def load_split(root: Path, split: str) -> list[SpiderExample]:
    split_file = root / f"{split}.json"
    with split_file.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    examples: list[SpiderExample] = []
    for example_id, row in enumerate(rows):
        examples.append(
            SpiderExample(
                example_id=example_id,
                split=split,
                db_id=row["db_id"],
                question=row["question"],
                gold_sql=row.get("sql") or "",
                gold_sparql=row.get("sparql") or "",
                gold_cypher=row.get("cypher") or "",
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
