.PHONY all

venv:
	python3 -m venv .venv

install-dev: venv
	.venv/bin/pip install -e '.[dev]'
	.venv/bin/pre-commit install

lint: venv
	.venv/bin/ruff check ./angionet/
	.venv/bin/mypy ./angionet/

test: venv
	.venv/bin/pytest -p no:cacheprovider ./tests/

all: lint test
