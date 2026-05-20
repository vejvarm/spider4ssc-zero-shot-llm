#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: scripts/serve_vllm.sh <run_id> [model_group]}"
MODEL_GROUP="${2:-main}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"

mapfile -t MODEL_FIELDS < <(
  python - "${MODEL_GROUP}" "${RUN_ID}" <<'PY'
from pathlib import Path
import sys

from spider4ssc_zeroshot.config import load_model_groups

model_group = sys.argv[1]
run_id = sys.argv[2]
model = load_model_groups(Path("configs/models.yaml"))[model_group][run_id]
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
  vllm serve "${MODEL_ID}"
  --host 0.0.0.0
  --port 8000
  --api-key "${VLLM_API_KEY}"
  --download-dir "${VLLM_DOWNLOAD_DIR:-models}"
  --tensor-parallel-size "${TP_SIZE}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
)

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  COMMAND+=(--trust-remote-code)
fi

exec "${COMMAND[@]}"
