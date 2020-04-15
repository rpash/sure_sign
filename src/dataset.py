"""
Load dataset into memory
"""

import os
import cv2
import numpy as np

def label_to_number(label):
    """
    Convert a string label to a number label
    Input:
        label: a string label
    Output:
        number: a number label
    """
    label = label.lower()
    if len(label) == 1:
        return ord(label) - 97

    if label == "del":
        return 26
    if label == "space":
        return 27

    # Nothing, label cannot be determined
    return 28


def number_to_label(number):
    """
    Convert a number label to a string label
    Input:
        number: a number label
    Output:
        label: a string label
    """
    if number < 26:
        return chr(number + 65)  # 65 to convert to uppercase label
    if number == 26:
        return "del"
    if number == 27:
        return "space"
    return "nothing"


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

    print("Loading ASL Alphabet train...", end="", flush=True)
    counter = 0
    for path, _, filenames in os.walk(train_path):
        label = label_to_number(os.path.basename(path).split('.')[-1])
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

    print("Done")

    print("Loading ASL Alphabet test...", end="", flush=True)
    for path, _, filenames in os.walk(test_path):
        for f in filenames:
            X_test.append(cv2.imread(os.path.join(
                path, f), cv2.IMREAD_GRAYSCALE))
            y_test.append(label_to_number(f.split('_')[0]))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print("Done")

    return X_train, y_train, X_test, y_test
