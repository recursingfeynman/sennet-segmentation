# SenNet Segmentation

## Installation

```console
$ pip install git+https://github.com/recursingfeynman/sennet-segmentation.git
```

## Development

To install project with development dependencies use following code:

```console
$ git clone https://github.com/recursingfeynman/sennet-segmentation.git

$ cd sennet-segmentation

$ make install-dev
```

### Testing

```console
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

In the bottom-left corner of the Neptune app, expand the user menu and select `Get your API token`.

### Model submission

Code below facilitates the integration between Neptune and Kaggle. It fetches runs from the Neptune tracking server, downloads the latest model checkpoint from a selected run, generates a code snapshot, and uploads the downloaded model and code snapshot to Kaggle as dataset.

```console
$ make submission
```

