"""
Featurization method for FFT
"""

import numpy as np
import cv2

def featurize(data):
    # TODO implement this
    ret = []
    grayData = []
    fftData = []
    
    grayData = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data]
    fftData = [np.fft.fft(img) for img in grayData]
    ret = [img.flatten() for img in fftData]
    
    return ret
