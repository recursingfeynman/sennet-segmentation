import os
import subprocess

URL = "https://github.com/recursingfeynman/sennet-segmentation.git"
HASH = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
HASH = HASH.decode("ascii").strip()
PIP = ".venv/bin/pip"


if __name__ == "__main__":
    if os.path.exists(f"./submission/snapshot-{HASH}"):
        print("Build already up to date.")
    else:
        os.system("rm -rf ./submission/snapshot-*")
        os.system(f"{PIP} wheel -w ./submission/snapshot-{HASH} --no-deps git+{URL}")
        print("Build success.")

