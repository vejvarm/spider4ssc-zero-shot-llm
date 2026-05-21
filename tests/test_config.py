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
            "schema_mode": "strict",
            "output_root": "runs",
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
    assert groups["main"]["qwen3_4b_instruct_2507"].provider == "vllm"
    assert groups["main"]["gemma3_27b_it"].tensor_parallel_size == 2


def test_load_model_groups_reads_openai_group():
    groups = load_model_groups(Path("configs/models.yaml"))

    model = groups["openai"]["gpt54_mini_20260317"]
    assert model.provider == "openai"
    assert model.model_id == "gpt-5.4-mini-2026-03-17"
    assert model.family == "gpt-5.4"
    assert model.size_label == "mini"
    assert model.tensor_parallel_size is None
    assert model.dtype is None


def test_load_experiment_config_reads_fixed_languages():
    config = load_experiment_config(Path("configs/experiment.yaml"))

    assert config.dataset.split == "test"
    assert config.experiment.languages == ["sql", "sparql", "cypher"]
    assert config.experiment.schema_serialization == "compact"
    assert config.experiment.schema_mode == "strict"
    assert config.experiment.output_root == Path("runs")
    assert config.decoding.temperature == 0.0
    assert config.decoding.max_completion_tokens == 2048
    assert config.decoding.reasoning_effort is None
    assert config.endpoint.api_key_env == "VLLM_API_KEY"
    assert not hasattr(config.endpoint, "api_key")


def test_load_sm3_adapted_experiment_config_reads_isolated_prompt_variant():
    config = load_experiment_config(Path("configs/experiment_sm3_adapted.yaml"))

    assert config.dataset.split == "test"
    assert config.experiment.languages == ["sql", "sparql", "cypher"]
    assert config.experiment.prompt_files == {
        "sql": Path("prompts/sm3_adapted_sql_zero_shot.txt"),
        "sparql": Path("prompts/sm3_adapted_sparql_zero_shot.txt"),
        "cypher": Path("prompts/sm3_adapted_cypher_zero_shot.txt"),
    }
    assert config.experiment.output_root == Path("runs/sm3_adapted")
    assert config.experiment.report_dir == Path("reports/sm3_adapted")


def test_load_sm3_openai_experiment_config_uses_openai_endpoint_and_reasoning():
    config = load_experiment_config(Path("configs/experiment_sm3_openai.yaml"))

    assert config.experiment.prompt_files == {
        "sql": Path("prompts/sm3_adapted_sql_zero_shot.txt"),
        "sparql": Path("prompts/sm3_adapted_sparql_zero_shot.txt"),
        "cypher": Path("prompts/sm3_adapted_cypher_zero_shot.txt"),
    }
    assert config.experiment.output_root == Path("runs/sm3_adapted")
    assert config.experiment.report_dir == Path("reports/sm3_adapted")
    assert config.endpoint.base_url == "https://api.openai.com/v1"
    assert config.endpoint.api_key_env == "OPENAI_API_KEY"
    assert config.decoding.reasoning_effort == "none"


def test_invalid_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported language"):
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
                "output_root": "runs",
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


def test_duplicate_languages_are_rejected():
    payload = _valid_experiment_payload()
    payload["experiment"]["languages"] = ["sql", "sql"]

    with pytest.raises(ValueError, match="Duplicate languages"):
        ExperimentConfig(**payload)


def test_malformed_language_values_are_rejected_without_type_error():
    payload = _valid_experiment_payload()
    payload["experiment"]["languages"] = [["sql"]]

    with pytest.raises(ValueError, match="Unsupported language"):
        ExperimentConfig(**payload)


def test_invalid_decode_and_endpoint_bounds_are_rejected():
    with pytest.raises(ValueError):
        DecodingConfig(top_p=0)
    with pytest.raises(ValueError):
        DecodingConfig(max_completion_tokens=0)
    with pytest.raises(ValueError):
        EndpointConfig(request_timeout_seconds=0)


def test_decoding_reasoning_effort_accepts_none_and_rejects_unknown_values():
    assert DecodingConfig(reasoning_effort="none").reasoning_effort == "none"
    assert DecodingConfig(reasoning_effort=None).reasoning_effort is None
    with pytest.raises(ValueError):
        DecodingConfig(reasoning_effort="minimal")


def test_float_fields_reject_integer_values():
    with pytest.raises(ValueError, match="top_p must be a float"):
        DecodingConfig(top_p=1)
    with pytest.raises(ValueError, match="temperature must be a float"):
        DecodingConfig(temperature=0)
    with pytest.raises(ValueError, match="gpu_memory_utilization must be a float"):
        ModelConfig(
            model_id="fake/model",
            family="fake",
            size_label="fake",
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=1,
            max_model_len=8192,
            trust_remote_code=False,
            requires_hf_terms=False,
        )


def test_strict_scalars_reject_quoted_values():
    with pytest.raises(ValueError):
        DecodingConfig(top_p="0.5")
    with pytest.raises(ValueError):
        EndpointConfig(request_timeout_seconds="30")
    with pytest.raises(ValueError):
        ModelConfig(
            model_id="fake/model",
            family="fake",
            size_label="fake",
            tensor_parallel_size="2",
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=False,
            requires_hf_terms=False,
        )
    with pytest.raises(ValueError):
        ModelConfig(
            model_id="fake/model",
            family="fake",
            size_label="fake",
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code="false",
            requires_hf_terms=False,
        )


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


def test_openai_model_config_allows_missing_vllm_serving_fields():
    model = ModelConfig(
        provider="openai",
        model_id="gpt-5.4-mini-2026-03-17",
        family="gpt-5.4",
        size_label="mini",
    )

    assert model.provider == "openai"
    assert model.tensor_parallel_size is None
    assert model.gpu_memory_utilization is None


def test_vllm_model_config_requires_serving_fields():
    with pytest.raises(ValueError, match="vllm provider requires"):
        ModelConfig(
            model_id="example/model",
            family="example",
            size_label="1B",
        )
