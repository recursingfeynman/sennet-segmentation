from setuptools import find_packages, setup

NAME = "angionet"
VERSION = "0.0.1"
URL = "https://github.com/recursingfeynman/sennet-segmentation"
REQUIRED_PYTHON = ">=3.10"

with open("requirements.txt", encoding="utf-8") as f:
    REQUIRED = f.read().split("\n")

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = "\n" + f.read()

EXTRAS = {"dev": ["pytest", "ruff", "mypy", "codespell", "pre-commit"]}

setup(
    name=NAME,
    version=VERSION,
    url=URL,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=("tests")),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    python_requires=REQUIRED_PYTHON,
    license="Apache License 2.0",
)
