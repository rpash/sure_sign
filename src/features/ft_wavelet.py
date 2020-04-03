"""
Featurization method for Wavelet
"""

import numpy as np
import pywt

def featurize(data):
    # TODO implement this
    return pywt.dwt2(data, 'wavelet.name')