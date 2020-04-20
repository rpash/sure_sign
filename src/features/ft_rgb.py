"""
Featurization method for RGB
"""

import logging
import numpy as np
import cv2


def __transform(img, size):
    return cv2.resize(img, size).flatten()


def featurize(data, size):
    """
    Computes the features of each image by simply downsampling and flattening.
    Input:
        data: (N,H,W) matrix of N images.
        size: (H, W) tuple, resize images to this size
    Output:
        features: (N, M) matrix of N features of length M
    """
    logger = logging.getLogger("Featurizer")
    logger.info("Featurizing by downsampling to {}".format(size))
    features = np.array([__transform(img, size) for img in data])
    return features
