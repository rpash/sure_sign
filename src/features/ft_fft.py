"""
Featurization method for FFT
"""

import cv2
import numpy as np
from math import exp
from scipy.spatial import distance

#Gaussian High Pass Filter Code from: https://medium.com/@hicraigchen/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82
def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance.euclidean((y,x),center)**2)/(2*(D0**2))))
    return base


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
        gray_data = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data]
    else:
        gray_data = data
    fft_data = [np.fft.fft2(img) for img in gray_data]
    centered_data = [np.fft.fftshift(img) for img in fft_data]
    centered_filtered_data = [img * gaussianHP(50,img.shape) for img in centered_data]
    filtered_data = [np.fft.ifftshift(img) for img in centered_filtered_data] 
    features = np.array([img.flatten() for img in filtered_data])
    return features
