import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spider4ssc_zeroshot.run_generation import GenerationRequest, run_generation
from spider4ssc_zeroshot.vllm_client import VllmClient, VllmClientConfig


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


class _Model:
    def __init__(self, model_id: str) -> None:
        self.id = model_id


class _ModelList:
    def __init__(self, model_ids: list[str]) -> None:
        self.data = [_Model(model_id) for model_id in model_ids]


class _ModelsEndpoint:
    def __init__(self, model_ids: list[str]) -> None:
        self.model_ids = model_ids

    def list(self) -> _ModelList:
        return _ModelList(self.model_ids)


class _Choice:
    finish_reason = "stop"
    message = SimpleNamespace(content="SELECT 1")


class _Usage:
    prompt_tokens = 2
    completion_tokens = 3
    total_tokens = 5


class _CompletionsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[_Choice()], usage=_Usage())


class _ChatEndpoint:
    def __init__(self) -> None:
        self.completions = _CompletionsEndpoint()


class _FakeOpenAI:
    def __init__(self, model_ids: list[str]) -> None:
        self.models = _ModelsEndpoint(model_ids)
        self.chat = _ChatEndpoint()


def _client_with_fake_openai(model_ids: list[str], **overrides) -> VllmClient:
    config = VllmClientConfig(
        base_url="http://example.invalid/v1",
        api_key="token",
        readiness_timeout_seconds=0.001,
        request_timeout_seconds=1,
        max_retries=0,
        retry_sleep_seconds=0,
    )
    config = SimpleNamespace(**{**config.__dict__, **overrides})
    client = VllmClient(config)  # type: ignore[arg-type]
    client._client = _FakeOpenAI(model_ids)
    return client


def test_wait_until_ready_rejects_wrong_served_model():
    client = _client_with_fake_openai(["other-model"])

    with pytest.raises(TimeoutError, match="expected expected-model"):
        client.wait_until_ready("expected-model")


def test_complete_supports_dict_decoding_and_caches_model_revision(monkeypatch):
    revisions: list[str] = []

    def fake_revision(model_id: str) -> str:
        revisions.append(model_id)
        return "revision-1"

    monkeypatch.setattr("spider4ssc_zeroshot.vllm_client.resolve_model_revision", fake_revision)
    client = _client_with_fake_openai(["test-model"])
    decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_completion_tokens": 10,
        "stop": ["```"],
    }

    first = client.complete("prompt", "test-model", decoding)
    second = client.complete("prompt", "test-model", decoding)

    assert first["model_revision"] == "revision-1"
    assert second["model_revision"] == "revision-1"
    assert revisions == ["test-model"]
    assert client._client.chat.completions.calls[0] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "prompt"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 10,
        "stop": ["```"],
    }


def test_complete_attempts_once_when_max_retries_is_zero(monkeypatch):
    monkeypatch.setattr(
        "spider4ssc_zeroshot.vllm_client.resolve_model_revision",
        lambda model_id: "revision-1",
    )
    client = _client_with_fake_openai(["test-model"])

    result = client.complete(
        "prompt",
        "test-model",
        {"temperature": 0.0, "top_p": 1.0, "max_completion_tokens": 10, "stop": []},
    )

    assert result["raw_completion"] == "SELECT 1"
    assert len(client._client.chat.completions.calls) == 1
