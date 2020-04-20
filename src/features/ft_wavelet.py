"""
Featurization method for Wavelet
"""

import logging
import numpy as np
import pywt


def __transform(img, wavelet, level):
    return pywt.wavedec2(img, wavelet, level=level)[0].flatten()


def featurize(data, wavelet, level):
    """
    Compute the features of the given set of images using the discrete
    wavelet transform. We use the reconstructed image from a givel level of
    the wavelet decomposition. For 200x200 images, a level 3 yields 25x25
    images. Each level cuts each dimension in half.
    Input:
        data: (N,H,W) matrix of N grayscale images.
        wavelet: Wavelet to use, anything supported by PyWavelets works
        level: Level of wavelet decomposition
    Output:
        features: (N, M) matrix of N features of length M
    """
    logger = logging.getLogger("Featurizer")
    logger.info("Featurizing using level {} {} wavelet decomposition".format(
                level, wavelet))
    return np.array([__transform(img, wavelet, level) for img in data])
