import json

metadata = {
    "title": "Angionet Snapshot",
    "id": "recursingfeynman/angionet-snapshot",
    "licenses": [{"name": "CC0-1.0"}],
}

if __name__ == "__main__":
    with open("./submission/dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

