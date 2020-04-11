"""
Featurization method for Wavelet
"""

import numpy as np
import pywt

def featurize(data):
    # TODO implement this
    
    grayData = []
    coeffs = []
    ret = []
    A - []
    grayData = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data]
    coeffs = [pywt.dwt2(grayData, 'haar') for img in grayData]
    for item in coeffs:
        cA, (cV, cH, cD) = item
        A.append(cA)

    
    ret = [coef.flatten() for coef in A]
    
    return ret