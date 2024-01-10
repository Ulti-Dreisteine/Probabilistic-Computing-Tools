from pyitlib import discrete_random_variable as drv
from typing import Union
import numpy as np

from ..setting import DTYPES

from .mi_gief import MutualInfoGIEF, CondMutualInfoGIEF, MargEntropy, CondEntropy
from .mi_kde import MutualInfoKDE
from .mic import MIC, CMIC
from .mi_model import MutualInfoModel, CondMutualInfoModel
from .mi_quant import MutualInfoClassic, MutualInfoDarbellay
from .corr_coeff.coeff import cal_dist_corr, cal_pearson_corr, cal_spearman_corr

ASSOC_METHODS = [
    "PearsonCorr", "SpearmanCorr", "DistCorr",
    "MI-GIEF", "MI-model", "MI-cut", "MI-qcut", "MI-Darbellay", "MI-KDE",
    "MIC", "RMIC"
]
COND_ASSOC_METHODS = [
    "CMI-GIEF", "CMI-model", "CMI-cut", "CMI-qcut",
    "CMIC", "CRMIC", "DRV"
]


def cal_marg_entropy(x: np.ndarray, xtype: str, **kwargs) -> float:
    """
    计算边际熵

    Params:
    -------
    x: x数据, 一维或多维
    xtype: x的数值类型, "d"或者"c"
    kwargs:
        k: int, 当xtype == "c"时选用的KNN近邻数, defaults to 3
        metric: str, 当xtype == "c"时选用的距离度量方式, defaults to "chebyshev"
    """
    
    assert xtype in DTYPES
    
    return MargEntropy(x, xtype)(**kwargs)


def cal_cond_entropy(x: np.ndarray, xtype: str, z: np.ndarray, ztype: str, **kwargs) -> float:
    """
    计算条件熵

    Params:
    -------
    x: x数据, 一维或多维
    xtype: x的数值类型, "d"或者"c"
    z: z数据, 一维或多维
    ztype: z的数值类型, "d"或者"c"
    kwargs:
        k: int, 当xtype == "c"时选用的KNN近邻数, defaults to 3
        metric: str, 当xtype == "c"时选用的距离度量方式, defaults to "chebyshev"
    """
    
    assert xtype in DTYPES
    assert ztype in DTYPES
    
    return CondEntropy(x, xtype, z, ztype)(**kwargs)


def cal_assoc(x: np.ndarray, y: np.ndarray, method: str, xtype: str = None, ytype: str = None, 
              **kwargs) -> float:
    """
    计算相关或关联系数
    """
    
    assert method in ASSOC_METHODS
    
    # 线性相关系数
    if method == "PearsonCorr":
        return cal_pearson_corr(x, y)
    elif method == "SpearmanCorr":
        return cal_spearman_corr(x, y)
    elif method == "DistCorr":
        return cal_dist_corr(x, y)
    
    # 互信息
    elif method == "MI-GIEF":
        # kwargs: k, metric
        return MutualInfoGIEF(x, xtype, y, ytype)(**kwargs)
    elif method == "MI-model":
        # kwargs: model, test_ratio
        return MutualInfoModel(x, xtype, y, ytype)(**kwargs)
    elif method == "MI-cut":
        return MutualInfoClassic(x, y)(method="cut")
    elif method == "MI-qcut":
        return MutualInfoClassic(x, y)(method="qcut")
    elif method == "MI-Darbellay":
        return MutualInfoDarbellay(x, y)()
    elif method == "MI-KDE":
        return MutualInfoKDE(x, y)()
    
    # 最大信息系数
    elif method == "MIC":
        return MIC(x, y)(method="mic")
    elif method == "RMIC":
        return MIC(x, y)(method="rmic", **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}.")
    
    
def cal_cond_assoc(x: np.ndarray, y: np.ndarray, z: np.ndarray, method: str, xtype: str = None, 
                   ytype: str = None, ztype: str = None, **kwargs) -> float:
    """
    计算条件关联系数
    """
    
    assert method in COND_ASSOC_METHODS
    
    # 条件互信息
    if method == "CMI-GIEF":
        # kwargs: k, metric, method for estimating CMI
        return CondMutualInfoGIEF(x, xtype, y, ytype, z, ztype)(**kwargs)  
    elif method == "CMI-model":
        # kwargs: model, test_ratio
        return CondMutualInfoModel(x, xtype, y, ytype, z, ztype)(**kwargs)
    elif method == "CMI-cut":
        return MutualInfoClassic(x, y, z)(method="cut")
    elif method == "CMI-qcut":
        return MutualInfoClassic(x, y, z)(method="qcut")
    
    # 条件最大信息系数
    elif method == "CMIC":
        return CMIC(x, y, z)(method="mic")
    elif method == "CRMIC":
        return CMIC(x, y, z)(method="rmic")
    elif method == "DRV":
        return drv.information_mutual_conditional(x, y, z)
    else:
        raise ValueError(f"Unsupported method: {method}.")
    
    
def cal_general_assoc(x: Union[np.ndarray, list], y: Union[np.ndarray, list], 
                      z: Union[np.ndarray, list, None], method: str, xtype: str, ytype: str, 
                      ztype: str, **kwargs) -> float:
    """
    关联和条件关联的通用计算
    """
    
    return cal_assoc(x, y, method, xtype, ytype, **kwargs) if z is None \
        else cal_cond_assoc(x, y, z, method, xtype, ytype, ztype, **kwargs)