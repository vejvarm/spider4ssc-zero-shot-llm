#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-test}"
LIMIT="${2:-}"
MODEL_GROUP="openai"
CONFIG="configs/experiment_sm3_openai.yaml"
RUNS_ROOT="runs/sm3_adapted"
REPORT_DIR="reports/sm3_adapted"

if [[ -z "${OPENAI_API_KEY:-}" && -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ "${SPLIT}" != "test" && "${SPLIT}" != "dev" ]]; then
  echo "Usage: scripts/run_sm3_openai.sh SPLIT [LIMIT]" >&2
  echo "SPLIT must be 'test' or 'dev'." >&2
  exit 2
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for OpenAI SM3 runs." >&2
  exit 2
fi

mapfile -t RUN_IDS < <(
  python - "${MODEL_GROUP}" <<'PY'
from pathlib import Path
import sys

from spider4ssc_zeroshot.config import load_model_groups

model_group = sys.argv[1]
for run_id in load_model_groups(Path("configs/models.yaml"))[model_group]:
    print(run_id)
PY
)

for RUN_ID in "${RUN_IDS[@]}"; do
  for LANGUAGE in sparql sql cypher; do
    if [[ -n "${LIMIT}" ]]; then
      spider4ssc-zeroshot generate "${RUN_ID}" "${LANGUAGE}" \
        --config "${CONFIG}" \
        --model-group "${MODEL_GROUP}" \
        --split "${SPLIT}" \
        --schema-mode strict \
        --limit "${LIMIT}"
    else
      spider4ssc-zeroshot generate "${RUN_ID}" "${LANGUAGE}" \
        --config "${CONFIG}" \
        --model-group "${MODEL_GROUP}" \
        --split "${SPLIT}" \
        --schema-mode strict
    fi
    spider4ssc-zeroshot evaluate "${RUN_ID}" "${LANGUAGE}" \
      --config "${CONFIG}" \
      --split "${SPLIT}" \
      --schema-mode strict
  done
done

spider4ssc-zeroshot report \
  --split "${SPLIT}" \
  --runs-dir "${RUNS_ROOT}/${SPLIT}" \
  --output-dir "${REPORT_DIR}"
