# Spider4SSC Zero-Shot LLM Baselines

Initial scaffold for the reproducibility package. The full protocol is added after the CLI, vLLM scripts, and datastore commands are implemented.

## Data Source

`configs/experiment.yaml` contains the public Spider4SSC archive URL used by the companion training repository's dataset loader. The query string is part of the public Dropbox download link, not a private credential. For release artifacts, record the downloaded archive and extracted files in `data/Spider4SSC.manifest.json`.

## Local API Keys

The OpenAI-compatible vLLM API key is read from `VLLM_API_KEY`, as configured by `endpoint.api_key_env` in `configs/experiment.yaml`. Local vLLM accepts a dummy value when the server was started with the same key, but credentials are not committed to this repository.

Spider4SSC can be copied from a local source tree with `spider4ssc-zeroshot prepare-data --source ...`. Remote archive downloads require `dataset.archive_sha256` in the experiment config so a mutable URL cannot be extracted without an integrity check.

## Reproducible Resolver Workflow

Direct dependencies are pinned in `pyproject.toml`. After implementation, release artifacts should include the environment generated with `python -m pip freeze > requirements-lock.txt` from the validated Python 3.11 environment.

## Source Checkout Assumption

The default commands are intended to run from the repository root so that `configs/` and `prompts/` are available as source-tree files. Release packages should either include these files as package resources or require explicit `--config` and prompt paths.
