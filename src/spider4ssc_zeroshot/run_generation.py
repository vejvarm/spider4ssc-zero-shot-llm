from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from spider4ssc_zeroshot.postprocess import postprocess_completion


@dataclass(frozen=True)
class GenerationRequest:
    example_id: int
    split: str
    language: str
    db_id: str
    question: str
    gold_sql: str
    prompt: str


class CompletionClient(Protocol):
    def complete(self, prompt: str, model_id: str, decoding: Any) -> dict[str, Any]:
        ...


def _load_existing_rows(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []

    rows: list[dict[str, Any]] = []
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _validate_existing_rows(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    language: str | None,
) -> None:
    for row in rows:
        row_model_id = row.get("model_id")
        if row_model_id != model_id:
            raise ValueError(
                f"Existing prediction row uses model_id {row_model_id}, expected {model_id}"
            )
        if language is not None and row.get("language") != language:
            raise ValueError(
                f"Existing prediction row uses language {row.get('language')}, expected {language}"
            )


def _created_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _decoding_metadata(decoding: Any) -> dict[str, Any]:
    keys = ["temperature", "top_p", "max_completion_tokens", "stop"]
    if isinstance(decoding, dict):
        return {key: decoding[key] for key in keys if key in decoding}
    return {key: getattr(decoding, key) for key in keys if hasattr(decoding, key)}


def _write_metadata(
    metadata_file: Path,
    *,
    output_file: Path,
    rows: list[dict[str, Any]],
    model_id: str,
    decoding: Any,
    n_requested: int,
    n_skipped_existing: int,
    n_generated: int,
) -> None:
    revisions = sorted(
        {
            row.get("model_revision", "unknown")
            for row in rows
            if row.get("model_revision")
        }
    )
    metadata = {
        "decoding": _decoding_metadata(decoding),
        "model_id": model_id,
        "model_revision": revisions[0] if len(revisions) == 1 else "mixed",
        "n_completed": len(rows),
        "n_generated": n_generated,
        "n_requested": n_requested,
        "n_skipped_existing": n_skipped_existing,
        "prediction_file": output_file.name,
        "updated_at_utc": _created_at_utc(),
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_generation(
    requests: list[GenerationRequest],
    client: CompletionClient,
    model_id: str,
    decoding: Any,
    output_file: Path,
) -> None:
    existing_rows = _load_existing_rows(output_file)
    expected_language = requests[0].language if requests else None
    _validate_existing_rows(existing_rows, model_id=model_id, language=expected_language)
    completed_ids = {row["example_id"] for row in existing_rows}
    generated_rows: list[dict[str, Any]] = []
    n_skipped_existing = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file = output_file.with_name("metadata.json")

    with output_file.open("a", encoding="utf-8") as handle:
        for request in requests:
            if request.example_id in completed_ids:
                n_skipped_existing += 1
                continue

            completion = client.complete(request.prompt, model_id, decoding)
            raw_completion = completion["raw_completion"]
            row = {
                **asdict(request),
                "raw_completion": raw_completion,
                "prediction": postprocess_completion(raw_completion, request.language),
                "finish_reason": completion.get("finish_reason"),
                "usage": completion.get("usage", {}),
                "model_id": model_id,
                "model_revision": completion.get("model_revision", "unknown"),
                "created_at_utc": _created_at_utc(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            generated_rows.append(row)
            completed_ids.add(request.example_id)

    _write_metadata(
        metadata_file,
        output_file=output_file,
        rows=existing_rows + generated_rows,
        model_id=model_id,
        decoding=decoding,
        n_requested=len(requests),
        n_skipped_existing=n_skipped_existing,
        n_generated=len(generated_rows),
    )
