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


def _load_completed_ids(output_file: Path) -> set[int]:
    if not output_file.exists():
        return set()

    completed_ids: set[int] = set()
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed_ids.add(row["example_id"])
    return completed_ids


def _created_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_generation(
    requests: list[GenerationRequest],
    client: CompletionClient,
    model_id: str,
    decoding: Any,
    output_file: Path,
) -> None:
    completed_ids = _load_completed_ids(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as handle:
        for request in requests:
            if request.example_id in completed_ids:
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
            completed_ids.add(request.example_id)
