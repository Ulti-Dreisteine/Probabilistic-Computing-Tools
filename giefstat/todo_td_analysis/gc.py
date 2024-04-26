from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np


class PairwiseGC(object):
    """成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
    
    def cal_gc(self, max_lag: int = 5, alpha: float = 0.05):
        res = grangercausalitytests(
            np.vstack((self.y, self.x)).T,
            maxlag = max_lag,
            verbose = False,
        )
        
        f_value = res[max_lag][0]["ssr_ftest"][0]
        p_value = res[max_lag][0]["ssr_ftest"][1]
        res = f_value if p_value < alpha else 0.0
        return res