"""
Featurization method for FFT
"""

import numpy as np
import cv2

def featurize(data):
    """
    Computes the FFT features of a set of images
    Input:
        data: (N,H,W[,C]) matrix of N images. If C is provided, the images will
              be converted to greyscale first
    Output:
        features: (N, M) matrix of N features of length M
    """
    if data.ndim > 3:
        grey_data = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data]
    else:
        grey_data = data

    features = np.array([np.fft.fft2(img).flatten() for img in grey_data])
    return features
