.PHONY: venv install install-gpu test lint prepare-data report

PYTHON ?= python3.11
DATA_SOURCE ?= /home/vejvar-martin-nj/git/uT5-ssc/data/Spider4SSC

venv:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel

install:
	. .venv/bin/activate && python -m pip install -r requirements-dev.txt

install-gpu:
	. .venv/bin/activate && python -m pip install -r requirements-dev.txt -r requirements-gpu.txt

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && ruff check src tests scripts

prepare-data:
	. .venv/bin/activate && spider4ssc-zeroshot prepare-data --source $(DATA_SOURCE) --output data/Spider4SSC

report:
	. .venv/bin/activate && spider4ssc-zeroshot report --runs-dir runs/test --output-dir reports
