import pandas as pd
import numpy as np

from ...util import stdize_values, discretize_series
from ...pyitlib import discrete_random_variable as drv


class MutualInfoClassic(object):
    """
    基于经典离散化的互信息和条件互信息计算
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray=None):
        self.x_norm = stdize_values(x, "c")
        self.y_norm = stdize_values(y, "c")
        self.z_norm = stdize_values(z, "c") if z is not None else None
    
    def __call__(self, method="qcut") -> float:
        x_enc = discretize_series(self.x_norm, method=method).astype(int)
        y_enc = discretize_series(self.y_norm, method=method).astype(int)
        
        if self.z_norm is None:
            mi = drv.information_mutual(x_enc, y_enc)
            return float(mi)
        else:
            z_enc = discretize_series(self.z_norm, method=method).astype(int)
            cmi = drv.information_mutual_conditional(x_enc, y_enc, z_enc)
            return float(cmi)