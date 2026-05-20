from pathlib import Path

import pytest
from pydantic import ValidationError

from spider4ssc_zeroshot.config import (
    DecodingConfig,
    EndpointConfig,
    ExperimentConfig,
    ModelConfig,
    load_experiment_config,
    load_model_groups,
)


def _valid_experiment_payload() -> dict:
    return {
        "dataset": {
            "name": "Spider4SSC",
            "url": "https://example.org/dataset.tgz",
            "local_path": "data/Spider4SSC",
            "split": "test",
            "test_file": "test.json",
            "test_db_dir": "database_test",
        },
        "experiment": {
            "schema_serialization": "compact",
            "languages": ["sql"],
            "prompt_files": {"sql": "prompts/sql_zero_shot.txt"},
            "output_root": "runs/test",
            "report_dir": "reports",
        },
        "decoding": {},
        "endpoint": {},
        "reproducibility": {
            "forbid_prompt_change_after_full_run": True,
            "record_full_prompt": True,
            "record_raw_completion": True,
            "record_model_revision": True,
        },
    }


def test_load_model_groups_reads_main_matrix():
    groups = load_model_groups(Path("configs/models.yaml"))

    assert "main" in groups
    assert groups["main"]["qwen3_4b_instruct_2507"].model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert groups["main"]["gemma3_27b_it"].tensor_parallel_size == 2


def test_load_experiment_config_reads_fixed_languages():
    config = load_experiment_config(Path("configs/experiment.yaml"))

    assert config.dataset.split == "test"
    assert config.experiment.languages == ["sql", "sparql", "cypher"]
    assert config.experiment.schema_serialization == "compact"
    assert config.decoding.temperature == 0.0
    assert config.decoding.max_completion_tokens == 2048
    assert config.endpoint.api_key_env == "VLLM_API_KEY"
    assert not hasattr(config.endpoint, "api_key")


def test_invalid_language_is_rejected():
    with pytest.raises(ValueError, match="Input should be 'sql', 'sparql' or 'cypher'"):
        ExperimentConfig(
            dataset={
                "name": "Spider4SSC",
                "url": "https://example.org/dataset.tgz",
                "local_path": "data/Spider4SSC",
                "split": "test",
                "test_file": "test.json",
                "test_db_dir": "database_test",
            },
            experiment={
                "schema_serialization": "compact",
                "languages": ["sql", "gremlin"],
                "prompt_files": {"sql": "prompts/sql_zero_shot.txt"},
                "output_root": "runs/test",
                "report_dir": "reports",
            },
            decoding=DecodingConfig(),
            endpoint=EndpointConfig(),
            reproducibility={
                "forbid_prompt_change_after_full_run": True,
                "record_full_prompt": True,
                "record_raw_completion": True,
                "record_model_revision": True,
            },
        )


def test_unknown_experiment_key_is_rejected():
    payload = _valid_experiment_payload()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ExperimentConfig(**payload)


def test_prompt_files_must_match_selected_languages():
    payload = _valid_experiment_payload()
    payload["experiment"]["prompt_files"] = {"sparql": "prompts/sparql_zero_shot.txt"}

    with pytest.raises(ValueError, match="Prompt files must match selected languages"):
        ExperimentConfig(**payload)


def test_invalid_decode_and_endpoint_bounds_are_rejected():
    with pytest.raises(ValueError):
        DecodingConfig(top_p=0)
    with pytest.raises(ValueError):
        DecodingConfig(max_completion_tokens=0)
    with pytest.raises(ValueError):
        EndpointConfig(request_timeout_seconds=0)


def test_invalid_model_dtype_is_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id="example/model",
            family="example",
            size_label="1B",
            tensor_parallel_size=1,
            dtype="fp8",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=False,
            requires_hf_terms=False,
        )
