"""
Featurization method for Wavelet
"""

import numpy as np
import pywt
import cv2

def featurize(data,txName):
    ret = []
    grayData = []
    fftData = []
    
    grayData = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data]
    wlData = [pywt.dwt(img,txName) for img in grayData]
    #TODO: wlData returns two arrays cA and cD. Don't know which one to return. Also cA and cD are not same shape as original grayscaled image
    
    
    return ret

