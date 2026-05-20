import json
from pathlib import Path
from types import SimpleNamespace

from spider4ssc_zeroshot.run_generation import GenerationRequest, run_generation


class FakeCompletionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def complete(self, prompt: str, model_id: str, decoding: object) -> dict:
        self.calls.append((prompt, model_id, decoding))
        return {
            "raw_completion": "Here is the query:\n```sql\nSELECT COUNT(*) FROM student;\n```",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            "model_revision": "abc123",
        }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_generation_skips_existing_rows_and_appends_missing(tmp_path: Path):
    output_file = tmp_path / "predictions" / "sql.jsonl"
    output_file.parent.mkdir()
    existing_row = {
        "example_id": 1,
        "split": "test",
        "language": "sql",
        "db_id": "tiny_school",
        "question": "Existing question?",
        "gold_sql": "SELECT 1",
        "prompt": "existing prompt",
        "raw_completion": "SELECT 1",
        "prediction": "SELECT 1",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "model_id": "existing-model",
        "model_revision": "old",
        "created_at_utc": "2026-01-01T00:00:00Z",
    }
    output_file.write_text(json.dumps(existing_row) + "\n", encoding="utf-8")
    requests = [
        GenerationRequest(
            example_id=1,
            split="test",
            language="sql",
            db_id="tiny_school",
            question="Existing question?",
            gold_sql="SELECT 1",
            prompt="existing prompt",
        ),
        GenerationRequest(
            example_id=2,
            split="test",
            language="sql",
            db_id="tiny_school",
            question="How many students are there?",
            gold_sql="SELECT count(*) FROM student",
            prompt="missing prompt",
        ),
    ]
    client = FakeCompletionClient()
    decoding = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        max_completion_tokens=2048,
        stop=["```"],
    )

    run_generation(
        requests=requests,
        client=client,
        model_id="test-model",
        decoding=decoding,
        output_file=output_file,
    )

    rows = _read_jsonl(output_file)
    assert rows[0] == existing_row
    assert len(rows) == 2
    assert client.calls == [("missing prompt", "test-model", decoding)]
    assert rows[1] == {
        "example_id": 2,
        "split": "test",
        "language": "sql",
        "db_id": "tiny_school",
        "question": "How many students are there?",
        "gold_sql": "SELECT count(*) FROM student",
        "prompt": "missing prompt",
        "raw_completion": "Here is the query:\n```sql\nSELECT COUNT(*) FROM student;\n```",
        "prediction": "SELECT COUNT(*) FROM student;",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "model_id": "test-model",
        "model_revision": "abc123",
        "created_at_utc": rows[1]["created_at_utc"],
    }
    assert rows[1]["created_at_utc"].endswith("Z")
