"""
Featurization method for FFT
"""

import numpy as np
import cv2

def featurize(data):
    # TODO implement this
    ret = []
    grayData = []
    for i in data:
        grayData.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
    ret = [np.fft.fft(i) for i in grayData]
    return ret
