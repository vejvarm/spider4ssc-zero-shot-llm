#!/usr/bin/env bash
set -euo pipefail

MODEL_GROUP="${1:-main}"
SPLIT="${2:-test}"
LIMIT="${3:-}"
CONFIG="configs/experiment_sm3_adapted.yaml"
RUNS_ROOT="runs/sm3_adapted"
REPORT_DIR="reports/sm3_adapted"

if [[ "${SPLIT}" != "test" && "${SPLIT}" != "dev" ]]; then
  echo "Usage: scripts/run_sm3_matrix.sh MODEL_GROUP SPLIT [LIMIT]" >&2
  echo "SPLIT must be 'test' or 'dev'." >&2
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
  MODEL_ID="$(
    python - "${MODEL_GROUP}" "${RUN_ID}" <<'PY'
from pathlib import Path
import sys

from spider4ssc_zeroshot.config import load_model_groups

model_group = sys.argv[1]
run_id = sys.argv[2]
print(load_model_groups(Path("configs/models.yaml"))[model_group][run_id].model_id)
PY
  )"
  echo "Start vLLM manually in another terminal:"
  echo "  . .venv/bin/activate && VLLM_API_KEY=\${VLLM_API_KEY:-token-abc123} scripts/serve_vllm.sh ${RUN_ID} ${MODEL_GROUP}"
  echo "Then press Enter here after the server is ready."
  read -r _

  python scripts/wait_for_vllm.py --expected-model "${MODEL_ID}"
  for LANGUAGE in sql sparql cypher; do
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
