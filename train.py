#!/usr/bin/env python3

import os
import sys
import yaml
import src.dataset as dataset

from src.features.featurizer import Featurizer

def full_path(path):
    """
    Append the given path to this file's full path
    """
    this_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_path, path)

def train():
    """
    Train! TBD TODO
    """
    with open(r"config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        train_path = full_path(config["dataset"]["train"])
        test_path = full_path(config["dataset"]["test"])

        sift_pickle = full_path(config["featurizers"]["sift"]["pickle"])
        sift_feature_len = config["featurizers"]["sift"]["feature_size"]

        X_train, y_train, X_test, y_test = dataset.load_asl_alphabet(
            train_path, test_path, train_len=1000)

        featurizer = Featurizer()
        features = featurizer.sift(X_train)
        test_features = featurizer.sift(X_test)

        print(features.shape)
        print(test_features.shape)

if __name__ == '__main__':
    train()
