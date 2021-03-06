"""
Featurize images
"""

import src.features.ft_rgb as ft_rgb
import src.features.ft_fft as ft_fft
import src.features.ft_wavelet as ft_wavelet
from src.features.ft_kmeans import KMeansFeaturizer


class Featurizer:
    """
    Featurize data and cache features
    """

    def __init__(self):
        """
        Store data in this object. The data stored here will be used to extract
        features using the featurization methods.
        """
        # featurizer cache
        self.__featurizer = {}

    def rgb(self, data, config):
        """
        Transform the data into a feature vector without any transformation
        Input:
            data: The data to featurize. This is a (NxHxWxC) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
        Return:
            The original set of images transformed into a feature vector
        """
        xdim = config["xdim"]
        ydim = config["ydim"]

        return ft_rgb.featurize(data, (xdim, ydim))

    def fft(self, data, config):
        """
        Construct a feature vector by applying FFT to the data
        Input:
            data: The data to featurize. This is a (NxHxWxC) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
        Return:
            The image feature vectors extracted using F FT
        """
        xdim = config["xdim"]
        ydim = config["ydim"]
        return ft_fft.featurize(data, (xdim, ydim))

    def dwt(self, data, config):
        """
        Construct a feature vector by applying DWT to the data
        Input:
            data: The data to featurize. This is a (NxHxWxC) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
            config: YAML node for DWT configuration
        Return:
            The image feature vectors extracted using DWT
        """
        wavelet = config["wavelet"]
        level = config["level"]
        return ft_wavelet.featurize(data, wavelet, level)

    def sift(self, data, config):
        """
        Construct a feature vector using SIFT
        Input:
            data: The data to featurize. This is a (NxHxWxC) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
            feature_size: size of each feature vector.
            pickle_path: path to model pickle file
            retrain: whether to retrain the model using new data
        Return:
            The image feature vectors extracted using SIFT
        """
        feature_size = config["feature_size"]
        pickle_path = config["pickle"]
        if "sift" not in self.__featurizer:
            self.__featurizer["sift"] = KMeansFeaturizer(
                feature_size, "sift")
            return self.__featurizer["sift"].train(data, pickle_path)

        return self.__featurizer["sift"].test(data)

    def surf(self, data, config):
        """
        Construct a feature vector using SURF
        Input:
            data: The data to featurize. This is a (NxHxWxC) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
            feature_size: size of each feature vector
            pickle_path: path to model pickle file
            retrain: whether to retrain the model using new data
        Return:
            The image feature vectors extracted using SURF
        """
        feature_size = config["feature_size"]
        pickle_path = config["pickle"]
        if "surf" not in self.__featurizer:
            self.__featurizer["surf"] = KMeansFeaturizer(
                feature_size, "surf")
            return self.__featurizer["surf"].train(data, pickle_path)

        return self.__featurizer["surf"].test(data)

    def orb(self, data, config):
        """
        Construct a feature vector using ORB
        Input:
            data: The data to featurize. This is a (NxHxWxC) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
            feature_size: size of each feature vector
            pickle_path: path to model pickle file
            retrain: whether to retrain the model using new data
        Return:
            The image feature vectors extracted using ORB
        """
        feature_size = config["feature_size"]
        pickle_path = config["pickle"]
        if "orb" not in self.__featurizer:
            self.__featurizer["orb"] = KMeansFeaturizer(
                feature_size, "orb")
            return self.__featurizer["orb"].train(data, pickle_path)

        return self.__featurizer["orb"].test(data)
