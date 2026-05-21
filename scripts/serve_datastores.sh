#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:?usage: scripts/serve_datastores.sh <test|dev>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

DATASET_ROOT="${SPIDER4SSC_DATASET_ROOT:-data/Spider4SSC}"
COMPOSE_FILE="${SPIDER4SSC_DATASTORE_COMPOSE:-docker/compose.datastores.yml}"
NEO4J_ROOT="${SPIDER4SSC_NEO4J_ROOT:-docker/neo4j-root}"
NEO4J_CONTAINER="${NEO4J_DOCKER_CONTAINER:-neo4j_server}"
RDF4J_READY_URL="${RDF4J_READY_URL:-http://localhost:8181/rdf4j-server/protocol}"
WAIT_SECONDS="${SPIDER4SSC_DATASTORE_WAIT_SECONDS:-120}"
PYTHON_BIN="${PYTHON:-python}"
CLI_BIN="${SPIDER4SSC_CLI:-spider4ssc-zeroshot}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if ! command -v "${CLI_BIN}" >/dev/null 2>&1 && [[ -x ".venv/bin/spider4ssc-zeroshot" ]]; then
  CLI_BIN=".venv/bin/spider4ssc-zeroshot"
fi

case "${SPLIT}" in
  test)
    DB_SUBFOLDER="database_test"
    IMPORT_SUBFOLDER="import/Spider4SSC/database_test"
    ;;
  dev)
    DB_SUBFOLDER="database"
    IMPORT_SUBFOLDER="import/Spider4SSC/database"
    ;;
  *)
    echo "usage: scripts/serve_datastores.sh <test|dev>" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATASET_ROOT}/${DB_SUBFOLDER}" ]]; then
  echo "Missing ${SPLIT} database folder: ${DATASET_ROOT}/${DB_SUBFOLDER}" >&2
  echo "Run spider4ssc-zeroshot prepare-data first." >&2
  exit 1
fi

run() {
  printf "+ " >&2
  printf "%q " "$@" >&2
  printf "\n" >&2
  "$@"
}

wait_for_rdf4j() {
  local deadline=$((SECONDS + WAIT_SECONDS))
  until curl --fail --silent --show-error --max-time 2 "${RDF4J_READY_URL}" >/dev/null; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for RDF4J at ${RDF4J_READY_URL}" >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_neo4j() {
  local deadline=$((SECONDS + WAIT_SECONDS))
  until docker exec "${NEO4J_CONTAINER}" cypher-shell -u neo4j -p secretserver "RETURN 1;" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for Neo4j container ${NEO4J_CONTAINER}" >&2
      return 1
    fi
    sleep 2
  done
}

echo "Resetting Spider4SSC datastores for split: ${SPLIT}"
run docker compose -f "${COMPOSE_FILE}" down -v
run docker compose -f "${COMPOSE_FILE}" up -d

echo "Waiting for RDF4J and Neo4j..."
wait_for_rdf4j
wait_for_neo4j

run chmod -R a+rX "${DATASET_ROOT}/${DB_SUBFOLDER}"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "Loading RDF4J repositories from ${DATASET_ROOT}/${DB_SUBFOLDER}"
run "${PYTHON_BIN}" -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_rdf4j_graphs \
  "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --db-subfolder "${DB_SUBFOLDER}"

echo "Loading Neo4j databases from ${DATASET_ROOT}/${DB_SUBFOLDER}"
export NEO4J_HOST_DB_ROOT="${REPO_ROOT}/${DATASET_ROOT}"
export NEO4J_DB_SUBFOLDER="${DB_SUBFOLDER}"
export NEO4J_IMPORT_SUBFOLDER="${IMPORT_SUBFOLDER}"
export NEO4J_DOCKER_CONTAINER="${NEO4J_CONTAINER}"
run "${PYTHON_BIN}" -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_neo4j_graphs \
  "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --neo4j-root "${NEO4J_ROOT}" \
  --db-subfolder "${DB_SUBFOLDER}" \
  --import-subfolder "${IMPORT_SUBFOLDER}"

echo "Extracting strict Neo4j schemas for ${SPLIT}"
run "${CLI_BIN}" extract-neo4j-schemas \
  --split "${SPLIT}" \
  --neo4j-root "${NEO4J_ROOT}" \
  --import-subfolder "${IMPORT_SUBFOLDER}" \
  --no-wipe

echo "Validating strict ${SPLIT} pipeline"
run "${CLI_BIN}" validate-pipeline --split "${SPLIT}" --schema-mode strict

echo "Datastores are ready for split: ${SPLIT}"
