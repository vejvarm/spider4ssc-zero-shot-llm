import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_wait_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "wait_for_vllm.py"
    spec = importlib.util.spec_from_file_location("wait_for_vllm", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeClient:
    def __init__(self, model_ids: list[str]) -> None:
        self.models = SimpleNamespace(list=self._list_models)
        self._model_ids = model_ids

    def _list_models(self):
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self._model_ids]
        )


def test_wait_until_ready_returns_served_models_when_expected_matches():
    wait_for_vllm = _load_wait_module()

    model_ids = wait_for_vllm.wait_until_ready(
        _FakeClient(["Qwen/Qwen3-4B-Instruct-2507"]),
        expected_model="Qwen/Qwen3-4B-Instruct-2507",
        timeout_seconds=1,
        sleep_seconds=0,
    )

    assert model_ids == ["Qwen/Qwen3-4B-Instruct-2507"]


def test_wait_until_ready_rejects_wrong_served_model(monkeypatch):
    wait_for_vllm = _load_wait_module()
    monotonic_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(wait_for_vllm.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(wait_for_vllm.time, "sleep", lambda seconds: None)

    with pytest.raises(TimeoutError, match="expected expected-model") as exc_info:
        wait_for_vllm.wait_until_ready(
            _FakeClient(["wrong-model"]),
            expected_model="expected-model",
            timeout_seconds=1,
            sleep_seconds=0,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
