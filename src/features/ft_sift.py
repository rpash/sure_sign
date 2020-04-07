"""
Featurization method for SIFT
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2


class SiftFeaturizer:
    def __init__(self, vocab_size):
        self.__vocab_size = vocab_size
        self.__kmeans = None
        self.__sift = cv2.xfeatures2d.SIFT_create()

    def train_features(self, data):
        n_images = data.shape[0]
        # SIFT uses 128 element feature vector
        descriptors = np.zeros((0, 128))
        features = np.zeros((n_images, self.__vocab_size))
        for img in data:
            _, desc = self.__sift.detectAndCompute(img, None)
            if desc is not None:
                np.append(descriptors, desc, axis=0)

        # K means to determine a feature vector for each image
        # Use MiniBatchKmeans as it is faster
        # TODO save this kmeans on disk maybe?
        self.__kmeans = MiniBatchKMeans(
            n_clusters=self.__vocab_size).fit(descriptors)
        for i in np.arange(n_images):
            features[i, self.__kmeans.labels_[i]] += 1

        return features

    def test_features(self, data):
        if self.__kmeans is None:
            return None

        n_images = data.shape[0]
        # SIFT uses 128 element feature vector
        descriptors = np.zeros((0, 128))
        features = np.zeros((n_images, self.__vocab_size))
        for i, img in enumerate(data):
            features[i] = self.featurize(img)

        return features

    def featurize(self, img):
        if self.__kmeans is None:
            return None

        features = np.zeros(self.__vocab_size)
        _, desc = self.__sift.detectAndCompute(img, None)
        pred = self.__kmeans.predict(desc)
        features[pred] += 1
        return features



def featurize(data):
    """
    Extract SIFT features from images
    TODO Using BGR now, check if greyscale gives same or better results
    Input:
        data: (N,H,W,C) matrix of images
    Output:
        features: (N, M) matrix of M SIFT features for N images
    """
    # TODO moved to class, update dataset.py