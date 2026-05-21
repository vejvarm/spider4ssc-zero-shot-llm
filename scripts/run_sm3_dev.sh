#!/usr/bin/env bash
set -euo pipefail

MODEL_GROUP="${1:-main}"
LIMIT="${2:-}"

scripts/run_sm3_matrix.sh "${MODEL_GROUP}" dev "${LIMIT}"
