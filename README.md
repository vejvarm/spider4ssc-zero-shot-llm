# Spider4SSC Zero-Shot LLM Baselines

This repository reproduces zero-shot instruction-LLM baselines for Spider4SSC in
SQL, SPARQL, and Cypher.

The experiment is an external modern-model sanity check for the uT5 Spider4SSC
journal paper. It is not a controlled pretraining-bias experiment because the
tested instruction models have unknown pretraining and instruction-tuning
exposure.

## Main Matrix

| Run id | Model |
| --- | --- |
| `qwen3_4b_instruct_2507` | `Qwen/Qwen3-4B-Instruct-2507` |
| `qwen3_30b_a3b_instruct_2507` | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `gemma3_4b_it` | `google/gemma-3-4b-it` |
| `gemma3_27b_it` | `google/gemma-3-27b-it` |

Gemma models require accepting Google's Gemma terms on Hugging Face before
downloading.

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt -r requirements-gpu.txt
```

## Prepare Data

From an existing local Spider4SSC checkout:

```bash
spider4ssc-zeroshot prepare-data \
  --source /home/vejvar-martin-nj/git/uT5-ssc/data/Spider4SSC \
  --output data/Spider4SSC
```

The command writes `data/Spider4SSC.manifest.json` with file sizes and SHA-256
hashes. Remote archive downloads verify `dataset.archive_sha256` when it is
provided; release runs should set it after recording the exact downloaded
archive.

## Start Datastores

```bash
docker compose -f docker/compose.datastores.yml down -v
docker compose -f docker/compose.datastores.yml up -d
```

Load SPARQL and Cypher graph data:

```bash
PYTHONPATH=src python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_rdf4j_graphs \
  data/Spider4SSC \
  --split test

NEO4J_DB_ROOT="$(pwd)/data/Spider4SSC" \
NEO4J_DB_SUBFOLDER=database_test \
NEO4J_DOCKER_CONTAINER=neo4j_server \
PYTHONPATH=src python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_neo4j_graphs \
  data/Spider4SSC \
  --split test \
  --neo4j-root "$(pwd)/data/Spider4SSC"
```

The SQL evaluator reads SQLite files directly from `data/Spider4SSC/database_test`.

## Serve One Model

The OpenAI-compatible vLLM API key is read from `VLLM_API_KEY`. Local vLLM
accepts a dummy value when the server was started with the same key.

```bash
. .venv/bin/activate
export VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
scripts/serve_vllm.sh qwen3_4b_instruct_2507 main
```

In another terminal:

```bash
. .venv/bin/activate
python scripts/wait_for_vllm.py
```

## Smoke Run

Use a limit only for pipeline verification. Do not edit prompts after seeing
test predictions.

```bash
spider4ssc-zeroshot generate qwen3_4b_instruct_2507 sql --limit 5
spider4ssc-zeroshot evaluate qwen3_4b_instruct_2507 sql
spider4ssc-zeroshot report --runs-dir runs/test --output-dir reports
```

## Full Matrix

Run one model server at a time:

```bash
scripts/run_matrix.sh main
```

Outputs:

```text
runs/test/<run_id>/<language>/predictions.jsonl
runs/test/<run_id>/<language>/scores.json
reports/test_main_matrix.csv
reports/test_main_matrix.md
reports/test_main_matrix.tex
```

## Reproducibility Rules

- Use the prompts in `prompts/` unchanged for reported runs.
- Use `temperature: 0.0`, `top_p: 1.0`, and `max_completion_tokens: 2048`.
- Record raw completions and postprocessed predictions.
- Record the Hugging Face model revision in every prediction row.
- Report SQL, SPARQL, and Cypher test execution accuracy for every model.
- Treat SPARQL and Cypher test-set evaluation as cross-language denotation
  comparison against gold SQL, using paired Spider4SSC stores.
