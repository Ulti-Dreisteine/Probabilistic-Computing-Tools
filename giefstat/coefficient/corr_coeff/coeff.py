from scipy.stats import pearsonr, spearmanr
from typing import Union, List
import numpy as np
from pingouin import distance_corr as dcor

from ...util.univar_encoding import SuperCategorEncoding

# 数据编码

def _encode(x, y) -> np.ndarray:
    """
    如果x是类别型变量, 则对x进行编码
    注意: 这里选择有监督的编码,因此入参有y, 其他编码方式可以在univar_encoding里选择
    """
    
    super_enc = SuperCategorEncoding(x, y)
    return super_enc.mhg_encoding()


# 计算系数

def cal_dist_corr(x: Union[np.ndarray, List[float]], y: Union[np.ndarray, List[float]], 
                  x_type: str="c") -> float:
    """
    距离相关系数
    """
    
    x, y = np.array(x).flatten(), np.array(y).flatten()
    
    if x_type == "d":
        x = _encode(x, y)
        
    return dcor(x, y)[0]


def cal_pearson_corr(x: Union[np.ndarray, List[float]], y: Union[np.ndarray, List[float]],
                     x_type: str="c") -> float:
    """
    Pearson相关系数
    """
    
    if x_type == "d":
        x = _encode(x, y)
        
    return pearsonr(x, y)[0]


def cal_spearman_corr(x: Union[np.ndarray, List[float]], y: Union[np.ndarray, List[float]],
                      x_type: str="c") -> float:
    """
    Spearman相关系数
    """
    
    if x_type == "d":
        x = _encode(x, y)
        
    return spearmanr(x, y)[0]
    

