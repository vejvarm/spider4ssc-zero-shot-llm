from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

DType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
Language = Literal["sql", "sparql", "cypher"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictBaseModel):
    model_id: str
    family: str
    size_label: str
    tensor_parallel_size: StrictInt = Field(ge=1)
    dtype: DType
    gpu_memory_utilization: StrictFloat = Field(gt=0.0, le=1.0)
    max_model_len: StrictInt = Field(ge=1024)
    trust_remote_code: StrictBool
    requires_hf_terms: StrictBool

    @field_validator("gpu_memory_utilization", mode="before")
    @classmethod
    def validate_gpu_memory_utilization_float(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("gpu_memory_utilization must be a float")
        return value


class DatasetConfig(StrictBaseModel):
    name: str
    url: str
    local_path: Path
    split: str
    test_file: str
    test_db_dir: str


class ExperimentSettings(StrictBaseModel):
    schema_serialization: Literal["compact"]
    languages: list[Language]
    prompt_files: dict[Language, Path]
    output_root: Path
    report_dir: Path

    @field_validator("languages", mode="before")
    @classmethod
    def validate_languages(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        supported = {"sql", "sparql", "cypher"}
        invalid = [language for language in value if language not in supported]
        if invalid:
            unsupported = ", ".join(str(language) for language in invalid)
            raise ValueError(f"Unsupported language(s): {unsupported}")
        return value

    @model_validator(mode="after")
    def validate_prompt_files(self) -> ExperimentSettings:
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("Duplicate languages are not allowed")
        if set(self.prompt_files) != set(self.languages):
            raise ValueError("Prompt files must match selected languages")
        return self


class DecodingConfig(StrictBaseModel):
    temperature: StrictFloat = Field(default=0.0, ge=0.0)
    top_p: StrictFloat = Field(default=1.0, gt=0.0, le=1.0)
    max_completion_tokens: StrictInt = Field(default=2048, ge=1)
    stop: list[str] = Field(default_factory=lambda: ["```"])

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def validate_float_value(cls, value: object, info: ValidationInfo) -> object:
        if type(value) is not float:
            raise ValueError(f"{info.field_name} must be a float")
        return value


class EndpointConfig(StrictBaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "VLLM_API_KEY"
    readiness_timeout_seconds: StrictInt = Field(default=1800, ge=1)
    request_timeout_seconds: StrictInt = Field(default=180, ge=1)
    max_retries: StrictInt = Field(default=5, ge=0)
    retry_sleep_seconds: StrictInt = Field(default=5, ge=0)


class ReproducibilityConfig(StrictBaseModel):
    forbid_prompt_change_after_full_run: StrictBool
    record_full_prompt: StrictBool
    record_raw_completion: StrictBool
    record_model_revision: StrictBool


class ExperimentConfig(StrictBaseModel):
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
