"""
Load dataset into memory
"""

import os
import cv2
import numpy as np


def load_asl_alphabet(train_path, test_path, train_len=10000, test_len=10000):
    """
    Load the ASL Alphabet dataset into memory. The number of test and train
    examples can be specified, but no more examples than exists can be returned.
    Input:
        train_path: File path to root of training data directory
        test_path: File path to root of testing data directory
        train_len: Number of train examples to read in
        test_len: Number of test examples to read in

    Output:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    """

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("Loading ASL Alphabet train...", end="", flush=True)
    counter = 0
    for path, _, filenames in os.walk(train_path):
        label = os.path.basename(path).split('.')[-1]
        for f in filenames:
            X_train.append(cv2.imread(os.path.join(path, f)))
            y_train.append(label)
            counter += 1
            if counter >= train_len:
                break
        else:
            continue
        break
    # Transform from list (N,H,W,C) to np.array (N,C,H,W)
    X_train = np.array(X_train).transpose(0, 3, 1, 2)
    y_train = np.array(y_train)
    print("Done")

    print("Loading ASL Alphabet test...", end="", flush=True)
    counter = 0
    for path, _, filenames in os.walk(test_path):
        for f in filenames:
            X_test.append(cv2.imread(os.path.join(path, f)))
            y_test.append(f.split('_')[0])
            counter += 1
            if counter >= test_len:
                break
        else:
            continue
        break
    # Transform from list (N,H,W,C) to np.array (N,C,H,W)
    X_test = np.array(X_test).transpose(0, 3, 1, 2)
    y_test = np.array(y_test)
    print("Done")


    return X_train, y_train, X_test, y_test
