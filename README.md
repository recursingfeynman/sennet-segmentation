# SenNet Segmentation

## Submission History

| Run/Checkpoint |   CV  |   LB  | Notes |
|:--------------:|:-----:|:-----:|-------|
|    ANG-25/1    | 0.916 |  OOM  | patches, z-scaling, D800-S600, cc3d filtering (t16,c26), vessels * kidney (2cls, threshold 0.2) |
|    ANG-27/4    | 0.913 |  OOM  | patches, z-scaling, ElasticTransform, D800-S600, cc3d filtering (t16,c26), vessels * kidney  (2cls, threshold 0.4) |
|    ANG-27/4    | 0.913 | 0.634 | patches, z-scaling, ElasticTransform, D800-S600, cc3d filtering (t16,c26), vessels (threshold 0.9) |
|    ANG-27/4    | 0.913 | 0.671 | patches, z-scaling, ElasticTransform, D800-S600, cc3d filtering (t16,c26), vessels (threshold 0.5) |
|    ANG-27/4    | 0.913 | 0.686 | patches, z-scaling, ElasticTransform, D800-S600, cc3d filtering (t16,c26), vessels (threshold 0.2) |
|    ANG-34/4    | 0.860 | 0.679 | patches, z-scaling, D800-S600, cc3d filtering (t4,c26), vessels (threshold 0.5) |
|    ANG-34/4    | 0.860 | 0.694 | patches, z-scaling, TTA (rotate90, hflip, vflip), D800-S600, cc3d filtering (t4,c26), vessels (threshold 0.5) |
|    ANG-34/4    | 0.860 | 0.699 | patches, z-scaling, TTA (rotate90, hflip, vflip), D800-S600, cc3d filtering (t4,c26), vessels (threshold 0.2) |
|    ANG-34/4    | 0.860 | 0.724 | patches, z-scaling, TTA (2x rotate90, hflip, vflip), D800-S600, w/o postprocessing, vessels * kidney (threshold 0.5) |
|    ANG-34/4    | 0.860 | 0.730 | patches, z-scaling, TTA (2x rotate90, hflip, vflip), D800-S600, w/o postprocessing, vessels * kidney (threshold 0.1) |
|    ANG-47/3    | 0.830 | 0.690 | full, z-scaling, TTA (2x rotate90, hflip, vflip), w/o postprocessing, vessels (1cls threshold 0.2) |
|    ANG-65/5    | 0.804 | 0.663 | patches, WNet, z-scaling, TTA (2x rotate90, hflip, vflip), w/o postprocessing, vessels * kidney (threshold 0.2) |
|    ANG-71/8    | 0.904 | 0.704 | patches, rescale, +kidney-3-sparse, D800-S600, TTA (2x rotate90, hflip, vflip), fill holes, w/o lomc, vessels * kidney (hysteresis 0.2, 0.6)|
|    ANG-71/8    | 0.904 | 0.708 | patches, rescale, +kidney-3-sparse, D512-S412, TTA (2x rotate90, hflip, vflip), fill holes, w/o lomc, vessels * kidney (hysteresis 0.1, 0.4)|
|    ANG-71/8    | 0.904 | 0.703 | patches, rescale, +kidney-3-sparse, D512-S412, TTA (2x rotate90, hflip, vflip), fill holes, w/o lomc, vessels * kidney (threshold 0.2) |
|    ANG-79/3    | 0.815 | 0.726 | full resolution, rescale, TTA, fill holes, vessels * kidney (hysteresis 0.2, 0.5) |
|    ANG-84/5    | 0.838 | 0.739 | full resolution, z-scaling, TTA, fill holes, vessels * kidney (hysteresis 0.2, 0.5) |

## Todos 💡

- [ ] Investigate another distance-based loss functions (Hausdorff, Edge, etc).
- [ ] Subvolume based normalization (test).
- [ ] Train on kidney_1_dense + kidney_3_dense/sparse.
- [ ] ConvNext, transformer-based architectures, UNet++, BCU-Net.
- [ ] Fix generalized surface loss.
- [ ] 2.5D approach.

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
