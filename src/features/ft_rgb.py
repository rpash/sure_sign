"""
Featurization method for RGB
"""

import numpy as np

def featurize(data):
    c = []
    d = []
    
    for i in data:
        for j in i:
            for k in j:
                for l in k:
                    d.append(l)
        c.append(d)
        d = []  

    return c