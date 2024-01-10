# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2022/09/18 17:04:28

@File -> mi_cmi.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于模型估计MI和CMI
"""

__doc__ = """
    使用模型近似估计互信息, 原理如下
    I(x;Y) = Accu(y, model.predict(x))
    I(x;Y|Z) = I(Y; xZ) - I(Y; Z)
             = Accu(y, model.predict(xz)) - Accu(y, model.predict(z))
"""

from sklearn.metrics import f1_score as cal_f1, r2_score as cal_r2
import numpy as np

from ...setting import DTYPES
from ...util import stdize_values, exec_model_test, cal_metric


def _exec_model_test(x: np.ndarray, y: np.ndarray, ytype: str, model, test_ratio: float, 
                     z: np.ndarray=None) -> float:
    """
    进行建模测试
    
    Note:
    -----
    如果z为None, 则返回 Accu(x, y), 对应于 I(X;Y); 否则返回 Accu(xz, y) - Accu(z, y), 对应于 I(X;Y|Z)
    """
    
    metric = "f1" if ytype == "d" else "r2"
    
    if z is None:
        A1 = x.reshape(len(x), -1)
        accu_x, _ = exec_model_test(A1, y, model, metric, test_ratio, rounds=1)
        return accu_x
    else:
        A2 = np.c_[x, z]
        A3 = z.reshape(len(z), -1)
        accu_xz, _ = exec_model_test(A2, y, model, metric, test_ratio, rounds=1)
        accu_z, _ = exec_model_test(A3, y, model, metric, test_ratio, rounds=1)
        return accu_xz - accu_z


class MutualInfoModel(object):
    """
    基于模型的MI
    """
    
    def __init__(self, x: np.ndarray, xtype: str, y: np.ndarray, ytype: str):
        assert xtype in DTYPES
        assert ytype in DTYPES
        self.x_norm = stdize_values(x, xtype)
        self.y_norm = stdize_values(y, ytype)
        self.xtype = xtype
        self.ytype = ytype
        
    def __call__(self, model, test_ratio=0.3) -> float:
        return _exec_model_test(
            self.x_norm, self.y_norm.flatten(), self.ytype, model, test_ratio)
        
        
class CondMutualInfoModel(object):
    """
    基于模型的CMI
    """
    
    def __init__(self, x: np.ndarray, xtype: str, y: np.ndarray, ytype: str, z: np.ndarray, ztype: str):
        assert xtype in DTYPES
        assert ytype in DTYPES
        assert ztype in DTYPES
        self.x_norm = stdize_values(x, xtype)
        self.y_norm = stdize_values(y, ytype)
        self.z_norm = stdize_values(z, ztype)
        self.xtype = xtype
        self.ytype = ytype
        self.ztype = ztype
        
    def __call__(self, model, test_ratio=0.3) -> float:
        return _exec_model_test(
            self.x_norm, self.y_norm.flatten(), self.ytype, model, test_ratio, self.z_norm)