"""
Featurization method for FFT
"""

import logging
import numpy as np
import scipy.fftpack as fp


def __transform(img, shape):
    return np.abs(fp.fft2(img, shape=shape)).flatten()


def featurize(data, shape):
    """
    Computes the FFT features of a set of images
    Input:
        data: (N,H,W) matrix of N images.
    Output:
        features: (N, M) matrix of N features of length M
    """
    logger = logging.getLogger("Featurizer")
    logger.info("Featurizing using {}-point FFT2".format(shape))
    return np.array([__transform(img, shape) for img in data])
