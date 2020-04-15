"""
Featurization method for Wavelet
"""

import numpy as np
import pywt
import cv2

from sklearn import decomposition
from sklearn import datasets

def featurize(data, wavelet):
    """
    Compute the features of the given set of images using the discrete
    wavelet transform as described by Sarlashkar et all in "Feature extraction
    using wavelet transform for neural network based image classification"
    (1998). This is an old paper and very, very simplistic. We should use
    a different method.
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

    # TODO Find a better feature extraction method, maybe look into energy,
    # variance, entropy, etc or a combo? Consider also using multi level wavelet
    # transform with pywt.wavedec2
    # Possible inspirations:
    # https://github.com/RamiKhushaba/getmswtfeat
    # https://www.researchgate.net/publication/4319558_Feature_Extraction_Technique_using_Discrete_Wavelet_Transform_for_Image_Classification
    # https://www.sciencedirect.com/science/article/abs/pii/S0925231215017531?via%3Dihub

    features = [pywt.wavedec2(img, wavelet)[0] for img in grey_data]
    pca = decomposition.PCA(n_components=3)
    pca.fit(features)
    features = pca.transform(features)
    '''
    I attempted to implement wavedec2 and then perform PCA on it. Feel free to change it as appropriate.
    '''
    return np.array(features)

