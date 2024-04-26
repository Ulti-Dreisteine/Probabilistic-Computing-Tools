from PyIF import te_compute
import numpy as np


class PairwiseTE(object):
    """MIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
    
    def cal_te(self, k: int = 1, embedding: int = 1):
        res = te_compute.te_compute(
            self.x,
            self.y,
            k = k,
            embedding = embedding,
            safetyCheck = True,
            GPU = False
        )
        return res