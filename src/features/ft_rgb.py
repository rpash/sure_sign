"""
Featurization method for RGB
"""

import numpy as np
import cv2

def featurize(data):
    """
    Computes the features of each image by simply flattening it. If the image is
    in color, it will be converted to greyscale
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

    features = np.array([img.flatten() for img in grey_data])
    return features