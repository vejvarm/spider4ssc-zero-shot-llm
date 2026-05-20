from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Language = Literal["sql", "sparql", "cypher"]


class ModelConfig(BaseModel):
    model_id: str
    family: str
    size_label: str
    tensor_parallel_size: int = Field(ge=1)
    dtype: str
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    max_model_len: int = Field(ge=1024)
    trust_remote_code: bool
    requires_hf_terms: bool


class DatasetConfig(BaseModel):
    name: str
    url: str
    local_path: Path
    split: str
    test_file: str
    test_db_dir: str


class ExperimentSettings(BaseModel):
    schema_serialization: Literal["compact"]
    languages: list[str]
    prompt_files: dict[str, Path]
    output_root: Path
    report_dir: Path

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, value: list[str]) -> list[str]:
        supported = {"sql", "sparql", "cypher"}
        invalid = [language for language in value if language not in supported]
        if invalid:
            raise ValueError(f"Unsupported language(s): {', '.join(invalid)}")
        return value


class DecodingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 2048
    stop: list[str] = Field(default_factory=lambda: ["```"])


class EndpointConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "VLLM_API_KEY"
    readiness_timeout_seconds: int = 1800
    request_timeout_seconds: int = 180
    max_retries: int = 5
    retry_sleep_seconds: int = 5


class ReproducibilityConfig(BaseModel):
    forbid_prompt_change_after_full_run: bool
    record_full_prompt: bool
    record_raw_completion: bool
    record_model_revision: bool


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    experiment: ExperimentSettings
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    reproducibility: ReproducibilityConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_model_groups(path: Path) -> dict[str, dict[str, ModelConfig]]:
    raw = _load_yaml(path)
    groups: dict[str, dict[str, ModelConfig]] = {}
    for group_name, model_map in raw.items():
        if not isinstance(model_map, dict):
            raise ValueError(f"Expected mapping for model group {group_name}")
        groups[group_name] = {
            run_id: ModelConfig.model_validate(model_config)
            for run_id, model_config in model_map.items()
        }
    return groups


def load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_yaml(path))
