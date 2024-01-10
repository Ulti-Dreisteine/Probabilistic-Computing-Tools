from scipy.signal import find_peaks
from typing import List, Tuple
import numpy as np
import random


def _build_td_series(x: np.ndarray, y: np.ndarray, td_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: 这部分代码功能与util.build_td_series重复了
    x_td_, y_td_ = x.flatten(), y.flatten()
    lag_remain = np.abs(td_lag) % len(x_td_)    # 求余数

    if td_lag == 0:                             # 没有时滞, 那么x_td和y_td_1同时发生
        x_td = x_td_[1:].copy()
        y_td = y_td_[1:].copy()
    elif td_lag > 0:                            # 正时滞, x_td比y_td_1早lag_remain发生
        x_td = x_td_[:-lag_remain].copy()
        y_td = y_td_[lag_remain:].copy()
    else:                                       # 负时滞, x_td比y_td_1晚lag_remain发生
        x_td = x_td_[lag_remain + 1:].copy()
        y_td = y_td_[1: -lag_remain].copy()
        
    return x_td, y_td


def _shuffle(x: np.ndarray) -> np.ndarray:
    x_srg = np.random.choice(x, len(x), replace=True)  # 有放回采样
    return x_srg


class TransferEntropy(object):
    """传递熵时序因果检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, tau_x: int, tau_y: int, 
                 sub_sample_size: int = None, alpha: float = 0.01, rounds: int = 50):
        """
        Params:
        -------
        x: X的序列样本
        y: Y的序列样本
        tau_x: X的特征时间参数
        tau_y: Y的特征时间参数
        sub_sample_size: 子采样数
        alpha: 显著性水平
        rounds: 随机测试轮数
        
        Note:
        -----
        x和y序列值都必须为整数
        """
        assert isinstance(x.dtype, int)
        assert isinstance(y.dtype, int)
        
        self.x = x.flatten()
        self.y = y.flatten()
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.sub_sample_size = sub_sample_size
        self.alpha = alpha
        self.rounds = rounds
        self.N = len(self.x)
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1  # X和Y的窗口样本数以及Y的超前时间
        
    def _cal_te(self, x: np.ndarray, y: np.ndarray, sub_sample_size: int = None, 
                rounds: int = None) -> Tuple[float, float, List[float]]:
        """
        计算特定时延构造的时序样本的传递熵
        """
        
        ...
        
    def cal_td_te(self, td_lag, sub_sample_size: int = None, rounds: int = None):
        """
        计算X->Y@td_lag处的传递熵
        """
        
        x_td, y_td = _build_td_series(self.x, self.y, td_lag)
        return self._cal_te(x_td, y_td, sub_sample_size, rounds)
        
    