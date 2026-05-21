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
hashes. Remote archive downloads require `dataset.archive_sha256` in
`configs/experiment.yaml` before any network request is made, so a mutable URL
cannot be extracted without an integrity check.

## Start Datastores

Use the split-specific datastore scripts. Each script resets RDF4J, starts both
datastore containers, loads the selected split into RDF4J and Neo4j, extracts
any missing strict Neo4j schema JSON files, and validates the selected split.
Running one script after the other is supported; the later run leaves both graph
stores ready for that split.

```bash
scripts/serve_test.sh
scripts/serve_dev.sh
```

The test split uses `database_test`; the dev split uses `database`. The SQL
evaluator reads SQLite files directly from the configured split database folder.
The graph scripts are needed for SPARQL/Cypher execution and strict Cypher
schema extraction.

Manual equivalent for test:

```bash
docker compose -f docker/compose.datastores.yml down -v
docker compose -f docker/compose.datastores.yml up -d

chmod -R a+rX data/Spider4SSC/database_test data/Spider4SSC/database

PYTHONPATH=src python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_rdf4j_graphs \
  data/Spider4SSC \
  --split test \
  --db-subfolder database_test

NEO4J_HOST_DB_ROOT="$(pwd)/data/Spider4SSC" \
NEO4J_DB_SUBFOLDER=database_test \
NEO4J_IMPORT_SUBFOLDER=import/Spider4SSC/database_test \
NEO4J_DOCKER_CONTAINER=neo4j_server \
PYTHONPATH=src python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_neo4j_graphs \
  data/Spider4SSC \
  --split test \
  --neo4j-root "$(pwd)/docker/neo4j-root" \
  --db-subfolder database_test \
  --import-subfolder import/Spider4SSC/database_test

spider4ssc-zeroshot extract-neo4j-schemas \
  --split test \
  --neo4j-root docker/neo4j-root \
  --import-subfolder import/Spider4SSC/database_test \
  --no-wipe

spider4ssc-zeroshot validate-pipeline --split test --schema-mode strict
```

For manual dev loading, replace `--split test`, `database_test`, and
`import/Spider4SSC/database_test` with `--split dev`, `database`, and
`import/Spider4SSC/database`.

## Validate Pipeline

Reported paper-facing runs use strict schema mode. Strict Cypher serialization
requires cached `<db_id>.neo4j-schema.json` files from the Neo4j datastore, not
RDF-derived fallback schemas.

```bash
spider4ssc-zeroshot validate-pipeline --split test --schema-mode strict
spider4ssc-zeroshot validate-pipeline --split dev --schema-mode strict
```

If strict validation reports missing Neo4j schema JSON after loading a split,
extract schemas without wiping the loaded Neo4j databases:

```bash
spider4ssc-zeroshot extract-neo4j-schemas \
  --split test \
  --neo4j-root docker/neo4j-root \
  --import-subfolder import/Spider4SSC/database_test \
  --no-wipe
```

## Serve One Model

The OpenAI-compatible vLLM API key is read from `VLLM_API_KEY`. Local vLLM
accepts a dummy value when the server was started with the same key.

To keep Hugging Face and vLLM model files off the main disk, configure the
model cache paths before starting vLLM:

```bash
source scripts/setup_env_vars.sh
```

By default this uses `/backup/models`. To use another root:

```bash
SPIDER4SSC_MODEL_CACHE_ROOT=/backup/models source scripts/setup_env_vars.sh
```

You can also apply the environment to one command without changing the current
shell:

```bash
scripts/setup_env_vars.sh scripts/serve_vllm.sh qwen3_4b_instruct_2507 main
```

```bash
. .venv/bin/activate
export VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
scripts/serve_vllm.sh qwen3_4b_instruct_2507 main
```

In another terminal:

```bash
. .venv/bin/activate
python scripts/wait_for_vllm.py --expected-model Qwen/Qwen3-4B-Instruct-2507
```

## Strict Test Workflow

Use strict mode for reviewer-facing test results. Do not edit prompts after
seeing test predictions.

```bash
spider4ssc-zeroshot generate qwen3_4b_instruct_2507 sql --split test --schema-mode strict
spider4ssc-zeroshot evaluate qwen3_4b_instruct_2507 sql --split test --schema-mode strict
spider4ssc-zeroshot report --split test --runs-dir runs/test --output-dir reports
```

## Dev Workflow

The dev split has native SQL, SPARQL, and Cypher gold queries, so it is useful
for debugging model behavior before spending time on the test matrix.

```bash
spider4ssc-zeroshot generate qwen3_4b_instruct_2507 sparql --split dev --schema-mode strict
spider4ssc-zeroshot evaluate qwen3_4b_instruct_2507 sparql --split dev --schema-mode strict
spider4ssc-zeroshot report --split dev --runs-dir runs/dev --output-dir reports
```

## SM3-Adapted Prompt Variant

The default prompts remain the baseline. To connect runs to the SM3-Text-to-Query
schema 0-shot prompt family, use the SM3-adapted config. It keeps outputs
separate from baseline runs.

Run the full SM3-adapted matrix for all configured models and all three
languages:

```bash
scripts/run_sm3_dev.sh main
scripts/run_sm3_test.sh main
```

For a limited smoke run:

```bash
scripts/run_sm3_dev.sh main 20
```

For a single-language SPARQL smoke run:

```bash
spider4ssc-zeroshot generate qwen3_4b_instruct_2507 sparql \
  --config configs/experiment_sm3_adapted.yaml \
  --split dev \
  --schema-mode strict \
  --limit 20
spider4ssc-zeroshot evaluate qwen3_4b_instruct_2507 sparql \
  --config configs/experiment_sm3_adapted.yaml \
  --split dev \
  --schema-mode strict
spider4ssc-zeroshot report \
  --split dev \
  --runs-dir runs/sm3_adapted/dev \
  --output-dir reports/sm3_adapted
```

## Fallback Smoke Run

Fallback mode allows Cypher schema serialization from cached RDF schemas when
Neo4j schema JSON is not ready. It is only for smoke/debug runs and must not be
mixed with strict results.

```bash
spider4ssc-zeroshot generate qwen3_4b_instruct_2507 cypher \
  --split test \
  --schema-mode fallback \
  --limit 5
spider4ssc-zeroshot evaluate qwen3_4b_instruct_2507 cypher \
  --split test \
  --schema-mode fallback
```

## Full Matrix

Run one model server at a time:

```bash
scripts/run_matrix.sh main test
scripts/run_matrix.sh main dev
```

Outputs:

```text
runs/test/<run_id>/<language>/predictions.jsonl
runs/test/<run_id>/<language>/metadata.json
runs/test/<run_id>/<language>/scores.json
reports/test_main_matrix.csv
reports/test_main_matrix.md
reports/test_main_matrix.tex
```

## Reproducibility Rules

- Use the prompts in `prompts/` unchanged for reported runs.
- Use `configs/experiment_sm3_adapted.yaml` for the SM3-adapted prompt variant
  and report it separately from the baseline prompt results.
- Use `temperature: 0.0`, `top_p: 1.0`, and `max_completion_tokens: 2048`.
- Record raw completions and postprocessed predictions.
- Record the Hugging Face model revision in every prediction row.
- Record `split`, `schema_mode`, `gold_query`, and schema provenance in every
  prediction row.
- Report SQL, SPARQL, and Cypher test execution accuracy for every model using
  `schema_mode=strict`.
- Treat SPARQL and Cypher test-set evaluation as cross-language denotation
  comparison against gold SQL, using paired Spider4SSC stores.
- Treat dev-set SPARQL and Cypher evaluation as native-language execution
  against dev gold queries.
