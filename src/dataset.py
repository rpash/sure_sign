"""
Load dataset into memory
"""

import os
import logging
import cv2
import numpy as np
import src.utils as utils


def load_asl_alphabet(train_path, test_path, train_len=1e4):
    """
    Load the ASL Alphabet dataset into memory. The number of test and train
    examples can be specified, but no more examples than exists can be returned.
    Input:
        train_path: File path to root of training data directory
        test_path: File path to root of testing data directory
        train_len: Number of train examples to read in per class
        test_len: Number of test examples to read in per class

    Output:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    """

    X_train, y_train = [], []
    X_test, y_test = [], []

    logging.info("Loading ASL Alphabet train")
    counter = 0
    for path, _, filenames in os.walk(train_path):
        label = utils.label_to_number(os.path.basename(path).split('.')[-1])
        for f in filenames:
            X_train.append(cv2.imread(
                os.path.join(path, f), cv2.IMREAD_GRAYSCALE))
            y_train.append(label)
            counter += 1
            if counter >= train_len:
                counter = 0
                break

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    logging.info("Loading ASL Alphabet test")
    for path, _, filenames in os.walk(test_path):
        for f in filenames:
            X_test.append(cv2.imread(os.path.join(
                path, f), cv2.IMREAD_GRAYSCALE))
            y_test.append(utils.label_to_number(f.split('_')[0]))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test
