import json
import os

import click

dataset = {
    "title": "Angionet Snapshot",
    "id": "recursingfeynman/angionet-snapshot",
    "licenses": [{"name": "CC0-1.0"}],
}

kernel = {
    "id": "recursingfeynman/sennet-inference",
    "title": "SenNet | Inference",
    "code_file": "../notebooks/sennet-inference.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "dataset_sources": ["recursingfeynman/angionet-snapshot"],
    "competition_sources": ["blood-vessel-segmentation"],
    "kernel_sources": ["recursingfeynman/sennet-wheels"],
    "model_sources": [],
}


@click.command()
@click.option("-t", "--target", type=str)
def initialize(target: str):
    if target == "dataset":
        with open("./submission/dataset-metadata.json", "w") as f:
            json.dump(dataset, f, indent=4)
    elif target == "kernel":
        os.makedirs("./submission-notebook", exist_ok=True)
        with open("./submission-notebook/kernel-metadata.json", "w") as f:
            json.dump(kernel, f, indent=4)
    else:
        raise ValueError("Incorrect target. Supported: dataset, kernel.")


if __name__ == "__main__":
    initialize()
