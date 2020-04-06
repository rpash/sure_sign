"""
Featurization method for FFT
"""

import numpy as np

def featurize(data):
    # TODO implement this
    ret = []
    ret = [np.fft.fft(data[i]) for i in data]
    return ret
