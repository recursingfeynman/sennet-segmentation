# SenNet Segmentation

## Submission History

| Run/Checkpoint |   CV  |  LB | Notes                                                                                        |
|:--------------:|:-----:|:---:|----------------------------------------------------------------------------------------------|
|    ANG-25/1    | 0.916 | OOM | per-image standardization, D800-S600, cc3d filtering (t16,c26), vessels * kidney  (threshold 0.2)|

## Installation

```shell
$ pip install git+https://github.com/recursingfeynman/sennet-segmentation.git
```

## Development

To install project with development dependencies use following code:

```shell
$ git clone https://github.com/recursingfeynman/sennet-segmentation.git

$ cd sennet-segmentation

$ make install-dev
```

### Testing

```shell
$ make all
```

## Kaggle
To generate personal access tokens follow this link: https://github.com/settings/tokens

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

GITHUB_TOKEN = secrets.get_secret("github-token")
USERNAME = secrets.get_secret("github-username")
URL = f"https://{USERNAME}:{GITHUB_TOKEN}@github.com/{USERNAME}/sennet-segmentation.git"
    
os.system(f"pip install -q git+{URL}")
```

## Submission

### Authentication

#### Kaggle

In order to use the Kaggle’s public API, you must first authenticate using an API token. Go to the **'Account'** tab of your user profile and select **'Create New Token'**. This will trigger the download of *kaggle.json*, a file containing your API credentials.

If you are using the Kaggle CLI tool, the tool will look for this token at `~/.kaggle/kaggle.json` on Linux, OSX, and other UNIX-based operating systems, and at `C:\Users\<Windows-username>\.kaggle\kaggle.json` on Windows. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.

#### Neptune

In the bottom-left corner of the Neptune app, expand the user menu and select **Get your API token**. You need to set up some environment variables before running the script:

```dosini
# Inside .env file

NEPTUNE-PROJECT="segteam/sennet"
NEPTUNE-TOKEN="<neptune-api-token>"
```

### Model submission

```shell
$ make submission
```

This code facilitates the integration between Neptune and Kaggle. It fetches runs from the Neptune tracking server, downloads the latest model checkpoint from a selected run, generates a code snapshot, and uploads the downloaded model and code snapshot to Kaggle as dataset.

### Notebook submission
To submit entire inference notebook to competition use code below: 

```shell
$ make submission-notebook
```

It follows the same steps as described in the *Model submission* section. Once the snapshot is uploaded, it submits the notebook to Kaggle. Then just click “Submit to Competition”.
