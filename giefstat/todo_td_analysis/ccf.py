import numpy as np


class PairwiseCCF(object):
    """MIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
    
    def cal_ccf(self):
        ccf = np.cov(np.vstack((self.x, self.y.flatten())))[0, 1] / (np.std(self.x) * np.std(self.y))
        return np.abs(ccf)