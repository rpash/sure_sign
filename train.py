#!/usr/bin/env python3

import os
import time
import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt

import src.dataset as dataset
from src.features.featurizer import Featurizer
from src.models.ensemble.classifier import ASLClassifier
import src.utils as utils

def full_path(path):
    """
    Append the given path to this file's full path
    """
    this_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_path, path)

def train():
    """
    Train AdaBoost on given classifier.
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
        utils.ask_for_load(config["always_load"])
        utils.init_logger(config)

        # Read images from dataset
        X_train, y_train, X_test, y_test = dataset.load_asl_alphabet(
            train_path, test_path, train_len=train_len)

        N, H, W = X_train.shape
        logging.info("Loaded {} training images of size ({}, {})".format(N, H, W))
        N, H, W = X_test.shape
        logging.info("Loaded {} test images of size ({}, {})".format(N, H, W))

        # Extract features based on config
        featurizer = Featurizer()
        if ft_name == "fft":
            features = featurizer.fft(X_train, ft_config)
            test_features = featurizer.fft(X_test, ft_config)
        elif ft_name == "rgb":
            features = featurizer.fft(X_train, ft_config)
            test_features = featurizer.fft(X_test, ft_config)
        elif ft_name == "dwt":
            features = featurizer.dwt(X_train, ft_config)
            test_features = featurizer.dwt(X_test, ft_config)
        elif ft_name == "sift":
            features = featurizer.sift(X_train, ft_config)
            test_features = featurizer.sift(X_test, ft_config)
        elif ft_name == "surf":
            features = featurizer.surf(X_train, ft_config)
            test_features = featurizer.surf(X_test, ft_config)
        elif ft_name == "orb":
            features = featurizer.orb(X_train, ft_config)
            test_features = featurizer.orb(X_test, ft_config)

        N, M = features.shape
        logging.info("Extracted {} training features of length {}".format(N, M))
        N, M = test_features.shape
        logging.info("Extracted {} test features of length {}".format(N, M))

        # Cross validations
        clf = ASLClassifier(clf_config)
        xval_res = clf.cross_val_score(features, y_train, kfold, 11)
        logging.critical(xval_res)
        logging.critical(np.mean(xval_res))

        # Predictions on test images
        start = time.time()
        clf.fit(features, y_train)
        end = time.time()
        logging.info("Fit model in {} seconds".format(end - start))
        start = time.time()
        pred = clf.predict(test_features)
        end = time.time()
        logging.info("Predicted test dataset in {} seconds".format(end - start))

        # Plot each prediction and ground truth for test images
        for y_true, y_pred, img in zip(y_test, pred, X_test):
            plt.imshow(img, cmap="gray")
            plt.title("True: {}     Predicted: {}".format(
                utils.number_to_label(y_true),
                utils.number_to_label(y_pred)
            ))
            plt.show()

        logging.critical(pred)
        logging.critical(y_test)
        logging.critical(np.sum(pred == y_test) / len(pred))


if __name__ == '__main__':
    train()
