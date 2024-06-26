import glob
import os
import sys

import neptune
import pandas as pd
from environs import Env

env = Env()
env.read_env()

PROJECT = env("NEPTUNE-PROJECT")
TOKEN = env("NEPTUNE-TOKEN")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def display_runs(runs, nrows):
    title = "\t" + "       |       ".join(runs.columns)
    entry = "{} {:>18} {: >15} {: >23} {: >17}"
    print(title)
    print("-" * 87)
    for index, row in runs.iloc[-min(nrows, len(runs)) :].iterrows():
        timestamp = row["Timestamp"]
        id_ = row["ID"]
        owner = row["Owner"]
        score = str(round(row["Highest score"], 4)).replace("nan", "-\t\t\t")
        print(entry.format(index, timestamp, id_, owner, score))

        if index == nrows - 1:
            break


if __name__ == "__main__":
    # Fetch all runs
    with HiddenPrints():
        project = neptune.init_project(
            project=PROJECT, api_token=TOKEN, mode="read-only"
        )
        runs = project.fetch_runs_table(
            columns=["sys/creation_time", "sys/id", "sys/owner", "test/highest-score"]
        ).to_pandas()

        project.stop()

    runs.columns = ["Timestamp", "ID", "Owner", "Highest score"]
    runs["Timestamp"] = runs["Timestamp"].apply(
        pd.to_datetime, format="%Y-%m-%dT%H:%M:%S.%fZ"
    )
    runs["Timestamp"] = runs["Timestamp"].dt.strftime("%d-%m-%Y %H:%M")
    runs = runs.sort_values("Timestamp")

    display_runs(runs, 15)

    run_id = input("Enter run ID: ")

    if run_id not in runs["ID"].tolist():
        raise FileNotFoundError(f"Incorrect run ID: {run_id}")

    # Download latest checkpoint
    dest = "./submission"
    os.makedirs(dest, exist_ok=True)
    with HiddenPrints():
        run = neptune.init_run(
            with_id=run_id,
            project=PROJECT,
            api_token=TOKEN,
            capture_hardware_metrics=False,
            capture_stderr=False,
            capture_stdout=False,
            capture_traceback=False,
        )
        if "models" not in run.get_structure().keys():
            raise FileNotFoundError("Selected run does not contain any checkpoints")

        for model in glob.glob(f"{dest}/*.pt"):
            os.remove(model)

    models = list(run.get_structure()["models"].keys())

    print("    Checkpoint name")
    print("-" * 23)
    for idx, model in enumerate(models):
        print("{}    {}".format(idx, model))

    ckpt = int(input("Select checkpoint index: "))
    model = dest + f"/{run_id.lower()}-c{ckpt}-model.pt"
    run[f"models/{models[ckpt]}"].download(model)

    with HiddenPrints():
        run.stop()

    print("Model downloaded successfully.")
