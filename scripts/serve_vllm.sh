#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: scripts/serve_vllm.sh <run_id> [model_group]}"
MODEL_GROUP="${2:-main}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
PYTHON_BIN="${PYTHON:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-models}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if ! command -v "${VLLM_BIN}" >/dev/null 2>&1 && [[ -x ".venv/bin/vllm" ]]; then
  VLLM_BIN=".venv/bin/vllm"
fi

sanitize_ld_preload_for_torch() {
  local preload="${LD_PRELOAD:-}"
  [[ -n "${preload}" ]] || return 0

  local entries=()
  local entry
  for entry in ${preload//:/ }; do
    [[ -n "${entry}" ]] || continue
    case "$(basename "${entry}")" in
      libnccl.so|libnccl.so.*)
        continue
        ;;
    esac
    entries+=("${entry}")
  done

  if ((${#entries[@]} == 0)); then
    unset LD_PRELOAD
  else
    local IFS=:
    export LD_PRELOAD="${entries[*]}"
  fi
}

sanitize_ld_preload_for_torch

mkdir -p "${DOWNLOAD_DIR}"
DOWNLOAD_DIR="$(cd "${DOWNLOAD_DIR}" && pwd -P)"

mapfile -t MODEL_FIELDS < <(
  "${PYTHON_BIN}" - "${MODEL_GROUP}" "${RUN_ID}" <<'PY'
from pathlib import Path
import sys

from spider4ssc_zeroshot.config import load_model_groups

model_group = sys.argv[1]
run_id = sys.argv[2]
model = load_model_groups(Path("configs/models.yaml"))[model_group][run_id]
if model.provider != "vllm":
    raise SystemExit(f"{run_id} uses provider {model.provider}; serve_vllm.sh only serves vLLM models")
print(model.model_id)
print(model.tensor_parallel_size)
print(model.dtype)
print(model.max_model_len)
print(model.gpu_memory_utilization)
print(str(model.trust_remote_code).lower())
PY
)

MODEL_ID="${MODEL_FIELDS[0]}"
TP_SIZE="${MODEL_FIELDS[1]}"
DTYPE="${MODEL_FIELDS[2]}"
MAX_MODEL_LEN="${MODEL_FIELDS[3]}"
GPU_UTIL="${MODEL_FIELDS[4]}"
TRUST_REMOTE_CODE="${MODEL_FIELDS[5]}"

COMMAND=(
  "${VLLM_BIN}" serve "${MODEL_ID}"
  --host 0.0.0.0
  --port 8000
  --api-key "${VLLM_API_KEY}"
  --download-dir "${DOWNLOAD_DIR}"
  --tensor-parallel-size "${TP_SIZE}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
)

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  COMMAND+=(--trust-remote-code)
fi

exec "${COMMAND[@]}"
