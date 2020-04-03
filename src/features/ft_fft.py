"""
Featurization method for FFT
"""

import numpy as np

def featurize(data):
    # TODO implement this
    return np.fft.fft(data)