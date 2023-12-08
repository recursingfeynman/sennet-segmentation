# SenNet Segmentation

## Installation
```
$ pip install git+https://github.com/recursingfeynman/sennet-segmentation.git
```

## Development

To install project with development dependencies use following code:
```
$ git clone https://github.com/recursingfeynman/sennet-segmentation.git

$ cd sennet-segmentation

$ make install-dev
```

### Testing
```
$ make all
```

## Kaggle
To generate personal access tokens follow this link: https://github.com/settings/tokens
```
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

GITHUB_TOKEN = secrects.get_secret("<token-label>")
USERNAME = secrect.get_secret("<username-label>")
URL = f"https://{USERNAME}:{GITHUB_TOKEN}@github.com/{USERNAME}/sennet-segmentation.git"
    
os.system(f"pip install -q git+{URL}")
```