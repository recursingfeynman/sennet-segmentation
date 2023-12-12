.PHONY: all

venv:
	python3 -m venv .venv

install-dev: venv
	.venv/bin/pip install -e '.[dev]'
	.venv/bin/pre-commit install

lint: venv
	.venv/bin/ruff check ./angionet/
	.venv/bin/mypy ./angionet/
	.venv/bin/codespell ./angionet/

test: venv
	.venv/bin/pytest -p no:cacheprovider ./tests/

submission: venv
	.venv/bin/python3 bin/download-model.py
	.venv/bin/python3 bin/initialize.py -t dataset
	.venv/bin/python3 bin/upload-snapshot.py

submission-notebook: submission
	.venv/bin/python3 bin/initialize.py -t kernel
	.venv/bin/kaggle kernels push -p ./submission-notebook/

all: lint test

