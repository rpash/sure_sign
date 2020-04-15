"""
Featurization method for FFT
"""

from math import exp
import cv2
import numpy as np
from scipy.spatial import distance
import scipy.fftpack as fp

#Gaussian High Pass Filter Code from: https://medium.com/@hicraigchen/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82
def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance.euclidean((y,x),center)**2)/(2*(D0**2))))
    return base

def __process_img(img, n=200, filter_width = 25):
    fft_img = fp.fftshift(fp.fft2((img).astype(float)))
    h, _ = fft_img.shape

    # assume square images
    pmin = int(h / 2 - filter_width)
    pmax = int(h / 2 + filter_width + 1)
    fft_img[pxmin:pmax, pxmin:pmax] = 0
    return np.argpartition(fft_img.flatten(), -n)[-n:]

def featurize(data):
    """
    Computes the FFT features of a set of images
    Input:
        data: (N,H,W[,C]) matrix of N images. If C is provided, the images will
              be converted to greyscale first
    Output:
        features: (N, M) matrix of N features of length M
    """
    features = [np.argpartition(np.fft.fft2(img).flatten(), -200)[-200:] for img in data]
    #centered_data = [np.fft.fftshift(img) for img in fft_data]
    #centered_filtered_data = [img * gaussianHP(50,img.shape) for img in centered_data]
    #filtered_data = [np.fft.ifftshift(img) for img in centered_filtered_data]
    print("PCA")
    return np.array(features)
