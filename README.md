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