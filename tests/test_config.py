from pathlib import Path

import pytest

from spider4ssc_zeroshot.config import (
    DecodingConfig,
    EndpointConfig,
    ExperimentConfig,
    load_experiment_config,
    load_model_groups,
)


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
