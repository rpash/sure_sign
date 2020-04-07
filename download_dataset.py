#!/usr/bin/env python3

"""
Downloads the ASL Alphabet dataset from Kaggle. This requires that your have a
Kaggle account and a Kaggle API token.

To get a token:
    1. Visit `https://www.kaggle.com/<username>/account`
    2. Click "Create New API Token"
    3. Move the "kaggle.json" file into "~/.kaggle/"
    4. Run `chmod 600 ~/.kaggle/kaggle.json`

Alternatively, you can set the `KAGGLE_USERNAME` and `KAGGLE_KEY` based on the
info in the kaggle.json file if you do not wish to pullute your home directory
with garbage.
"""

import yaml
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
with open(r"config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    api.dataset_download_files(config['kaggle_dataset'],
                               path="dataset", unzip=True)
