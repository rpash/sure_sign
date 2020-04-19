#!/usr/bin/env python3

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import src.dataset as dataset
from src.features.featurizer import Featurizer
from src.models.svm.classifier import ASLClassifier


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

        ft_name = config["featurizers"]["featurizer"]
        ft_config = config["featurizers"][ft_name]
        clf_config = config["classification"]
        train_len = config["dataset"]["train_examples"]
        kfold = config["classification"]["k-fold"]

        X_train, y_train, X_test, y_test = dataset.load_asl_alphabet(
            train_path, test_path, train_len=train_len)

        print(X_train.shape)
        print(X_test.shape)

        featurizer = Featurizer()
        if ft_name == "fft":
            features = featurizer.fft(X_train, ft_config)
            test_features = featurizer.fft(X_test, ft_config)
            print(features.shape)
        elif ft_name == "dwt":
            features = featurizer.dwt(X_train, ft_config)
            test_features = featurizer.dwt(X_test, ft_config)
            print(features.shape)
        elif ft_name == "sift":
            features = featurizer.sift(X_train, ft_config)
            test_features = featurizer.sift(X_test, ft_config)
        elif ft_name == "surf":
            features = featurizer.surf(X_train, ft_config)
            test_features = featurizer.surf(X_test, ft_config)
        elif ft_name == "orb":
            features = featurizer.orb(X_train, ft_config)
            test_features = featurizer.orb(X_test, ft_config)

        clf = ASLClassifier(clf_config)
        xval_res = clf.cross_val_score(features, y_train, kfold, 11)
        print(xval_res)
        print(np.mean(xval_res))

        clf.fit(features, y_train)
        pred = clf.predict(test_features)

        for y_true, y_pred, img in zip(y_test, pred, X_test):
            plt.imshow(img, cmap="gray")
            plt.title("True: {}     Predicted: {}".format(
                dataset.number_to_label(y_true),
                dataset.number_to_label(y_pred)
            ))
            plt.show()

        print(pred)
        print(y_test)
        print(np.sum(pred == y_test) / len(pred))


if __name__ == '__main__':
    train()
