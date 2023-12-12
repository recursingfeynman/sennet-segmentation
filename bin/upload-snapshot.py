import json
import os
import subprocess

HASH = subprocess.check_output(["git", "rev-parse", "HEAD"])
HASH = HASH.decode("ascii").strip()
KAGGLE = ".venv/bin/kaggle"
MESSAGE = f"'Submission hash {HASH[:-7]}'"
PIP = ".venv/bin/pip"
METADATA = "./submission/snapshot-metadata.json"


if __name__ == "__main__":
    same = False
    try:
        with open(METADATA, "r") as f:
            current = json.load(f)
        same = current["commit-hash"] == HASH
    except FileNotFoundError:
        pass

    # Build project
    if same:
        print("Build already up to date.")
    else:
        if os.path.exists(METADATA):
            os.remove(METADATA)

        os.system(f"{PIP} wheel -w ./submission/snapshot --no-deps .")

        metadata = {"commit-hash": HASH}
        with open(METADATA, "w") as f:
            json.dump(metadata, f)

        print("Build success.")

    # Upload dataset
    os.system(f"{KAGGLE} datasets version -p ./submission/ -r zip -m {MESSAGE}")
