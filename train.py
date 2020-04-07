#!/usr/bin/env python3

import os
import yaml
import src.dataset as dataset


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

        print(train_path)
        print(test_path)

        X_train, y_train, X_test, y_test = dataset.load_asl_alphabet(
            train_path, test_path)

if __name__ == '__main__':
    train()
