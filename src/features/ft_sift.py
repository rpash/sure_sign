"""
Featurization method for SIFT
"""

import numpy as np
import cv2

def featurize(data):
    # TODO implement this
    sift = cv2.xfeatures2d.SIFT_create()
    kpt, desc = sift.detectAndCompute(data, None)
    return kpt, desc