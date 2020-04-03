"""
Featurize images
"""

import ft_rgb
import ft_fft
import ft_wavelet
import ft_sift
import ft_surf


class Featurizer:
    """
    Featurize data and cache features
    """

    def __init__(self, data):
        """
        Store data in this object. The data stored here will be used to extract
        features using the featurization methods.
        Input:
            data: The data to featurize. This is a (NxCxHxW) matrix where
                N = number of images
                C = Image channels
                H = Image height in pixels
                W = Image width in pixels
        """
        self.__data = data

        # cached features
        self.__features = {}

        # featurization functions
        self.__featurize = {"rgb": ft_rgb.featurize,
                            "fft": ft_fft.featurize,
                            "wavelet": ft_wavelet.featurize,
                            "sift": ft_sift.featurize,
                            "surf": ft_surf.featurize}

    def __extract_featues(self, key):
        """
        Dynamic feature extraction based on given key. This is meant as a helper
        function for the featurization methods.
        """
        if key not in self.__features:
            self.__features[key] = self.__featurize[key](self.__data)

        return self.__features[key]

    def rgb(self):
        """
        Transform the data into a feature vector without any transformation
        Return:
            The original set of images transformed into a feature vector
        """
        return self.__extract_featues("rgb")

    def fft(self):
        """
        Construct a feature vector by applying FFT to the data
        Return:
            The image feature vectors extracted using FFT
        """
        return self.__extract_featues("fft")

    def wavelet(self):
        """
        Construct a feature vector by applying DWT to the data
        Return:
            The image feature vectors extracted using DWT
        """
        return self.__extract_featues("wavelet")

    def sift(self):
        """
        Construct a feature vector using SIFT
        Return:
            The image feature vectors extracted using SIFT
        """
        return self.__extract_featues("sift")

    def surf(self):
        """
        Construct a feature vector using SURF
        Return:
            The image feature vectors extracted using SURF
        """
        return self.__extract_featues("surf")
