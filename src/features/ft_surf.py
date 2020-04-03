"""
Featurization method for SURF
"""

import numpy as np
import cv2

def featurize(data):
    # TODO implement this
    surf = cv2.xfeatures2d.SURF_create()
    kpt, desc = surf.detectAndCompute(data, None)
    return kpt, desc