"""
A collection of utility functions
"""

import os
import sys
import logging
import joblib


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


def yes_no(question):
    """
    Asks a yes/no question.
    Input:
        The question to ask
    Output:
        True: User answered YES
        False: User answered NO
    """
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_no(question)


def __log_level(level):
    if level == "debug":
        return logging.DEBUG
    elif level == "warning":
        return logging.WARNING
    elif level == "quiet":
        return logging.CRITICAL
    return logging.NOTSET


def init_logger(config):
    """
    Initialize the loggers
    """
    level = __log_level(config["log_level"])
    logging.basicConfig(level=level, format="%(msg)s")

    formatter = logging.Formatter("%(name)s: %(msg)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)

    cfl_logger = logging.getLogger("Classifier")
    cfl_logger.setLevel(__log_level(config["featurizers"]["log_level"]))
    cfl_logger.addHandler(sh)
    cfl_logger.propagate = False

    ft_logger = logging.getLogger("Featurizer")
    ft_logger.setLevel(__log_level(config["classification"]["log_level"]))
    ft_logger.addHandler(sh)
    ft_logger.propagate = False



def save_model(model, path):
    """
    Save a model for later use
    Input:
        model: the model to save (must be pickle-able)
        path: where to save the model
    """
    pickle_dir = os.path.dirname(path)
    if not os.path.isdir(pickle_dir):
        os.makedirs(pickle_dir)
    joblib.dump(model, path, compress=9)


__ALWAYS_LOAD = False


def load_model(path):
    """
    Load a model from a pickle file
    Input:
        path: location of pickle file
    Output:
        model: The loaded model
    """
    if not os.path.exists(path):
        return None

    global __ALWAYS_LOAD
    if __ALWAYS_LOAD:
        return joblib.load(path)
    elif not yes_no("Do you want to load pickle file at {}?".format(path)):
        return None

    return


def ask_for_load(always_load):
    """
    Set whether to ask before loading a pickled model
    """
    global __ALWAYS_LOAD
    __ALWAYS_LOAD = always_load
