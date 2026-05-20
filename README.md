# Spider4SSC Zero-Shot LLM Baselines

Initial scaffold for the reproducibility package. The full protocol is added after the CLI, vLLM scripts, and datastore commands are implemented.

## Reproducible Resolver Workflow

Direct dependencies are pinned in `pyproject.toml`. After implementation, release artifacts should include the environment generated with `python -m pip freeze > requirements-lock.txt` from the validated Python 3.11 environment.
