#!/usr/bin/env bash

# Source this file to configure the current shell:
#   source scripts/setup_env_vars.sh
#
# Or execute it as a wrapper for one command:
#   scripts/setup_env_vars.sh scripts/serve_vllm.sh qwen3_4b_instruct_2507 main

_spider4ssc_model_cache_root="${SPIDER4SSC_MODEL_CACHE_ROOT:-/backup/models}"

export HF_HOME="${_spider4ssc_model_cache_root}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_DOWNLOAD_DIR="${_spider4ssc_model_cache_root}/vllm"
export VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"

if ! mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${VLLM_DOWNLOAD_DIR}"; then
  echo "Failed to create model cache directories under ${_spider4ssc_model_cache_root}" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if (($# > 0)); then
    exec "$@"
  fi

  cat <<EOF
Model cache paths are configured for this process only.
To configure your current shell, run:

  source scripts/setup_env_vars.sh

Current values:
  HF_HOME=${HF_HOME}
  HF_HUB_CACHE=${HF_HUB_CACHE}
  VLLM_DOWNLOAD_DIR=${VLLM_DOWNLOAD_DIR}
  VLLM_API_KEY=${VLLM_API_KEY}

Override the root with:
  SPIDER4SSC_MODEL_CACHE_ROOT=/backup/models source scripts/setup_env_vars.sh
EOF
fi

unset _spider4ssc_model_cache_root
