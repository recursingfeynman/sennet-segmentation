import os
import subprocess

HASH = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
HASH = HASH.decode("ascii").strip()
KAGGLE = ".venv/bin/kaggle"
MESSAGE = f"'Submission hash {HASH}'"


if __name__ == "__main__":
    os.system(f"{KAGGLE} datasets version -p ./submission/ -r zip -m {MESSAGE}")

