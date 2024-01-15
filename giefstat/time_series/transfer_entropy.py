from typing import List, Tuple
import numpy as np
import random

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from giefstat.time_series.util import build_td_series, shuffle
from giefstat.probability_estimation.discrete import cal_discrete_prob


class TransferEntropy(object):
    """
    传递熵成对时延因果检验
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, tau_x: int, tau_y: int, alpha: float = 0.01):
        """
        Params:
        -------
        x: X的序列样本
        y: Y的序列样本
        tau_x: X的特征时间参数
        tau_y: Y的特征时间参数
        alpha: 显著性水平
        # sub_sample_size: 子采样数
        # rounds: 随机测试轮数
        
        Note:
        -----
        x和y序列值都必须为整数
        """
        
        try:
            assert "int" in str(x.dtype)
            assert "int" in str(y.dtype)
        except Exception as _:
            raise RuntimeError("x和y序列值必须均为整数") from _
        
        self.x = x.flatten()
        self.y = y.flatten()
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.alpha = alpha
        self.N = len(self.x)
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1  # X和Y的窗口样本数以及Y的超前时间
    
    @staticmethod
    def _sum(_arr_yyx, _states_yyx, eps) -> float:
        _te = 0.0
        
        for _state in _states_yyx:
            prob_yyx = cal_discrete_prob(_arr_yyx, [0, 1, 2], _state)
            prob_y_yx = cal_discrete_prob(_arr_yyx, [0], _state[[0]], [1, 2], _state[[1, 2]])
            prob_y_y = cal_discrete_prob(_arr_yyx, [0], _state[[0]], [1], _state[[1]])
            
            prob = prob_yyx * np.log2(prob_y_yx / (prob_y_y + eps) + eps)
            
            if not np.isnan(prob):
                _te += prob
        
        return _te
    
    def _cal_te(self, x: np.ndarray, y: np.ndarray, sub_sample_size: int = None, rounds: int = None,
                eps: float = 1e-12) -> Tuple[float, float, List[float]]:
        """
        计算特定序列样本的传递熵
        
        Params:
        -------
        x: 待计算的X序列
        y: 待计算的Y序列
        sub_sample_size: 每次计算的子采样数
        rounds: 重复次数
        """
        
        # 构造联合分布样本
        idxs = np.arange(0, len(x) - self.h * self.tau_y - 1)
        xk = x[idxs].reshape(-1, 1)
        yk = y[idxs].reshape(-1, 1)
        yh = yk.copy()
        
        arr_yyx = np.concatenate([
            yh[self.h * self.tau_y:, :],
            yk[: -self.h * self.tau_y, :], 
            xk[self.h * self.tau_y:, :], 
            ], axis = 1)  # type: np.ndarray
        
        # 进行多轮随机抽样计算
        N = len(arr_yyx)
        sub_sample_size = N if sub_sample_size is None else sub_sample_size
        rounds = 10 if rounds is None else rounds
        
        te_lst = []
        
        for _ in range(rounds):
            # 随机子采样以控制样本规模
            _idxs = random.sample(range(N), sub_sample_size)
            _arr_yyx = arr_yyx.copy()[_idxs, :]
            _states_yyx = np.unique(_arr_yyx, axis = 0)
            
            _te = self._sum(_arr_yyx, _states_yyx, eps)
            
            te_lst.append(_te)
            
        te_mean = np.nanmean(te_lst)
        te_std = np.nanstd(te_lst)
        
        return te_mean, te_std, te_lst
        
    def cal_td_te(self, td_lag: int, **kwargs) -> Tuple[float, float, List[float]]:
        """
        计算X->Y@td_lag处的传递熵
        
        Param:
        ------
        td_lag: X到Y的时延
        """
        
        # pylint: disable-next = unbalanced-tuple-unpacking
        x_td, y_td =  build_td_series(self.x, self.y, td_lag)
        te_mean, te_std, te_lst = self._cal_te(x_td, y_td, **kwargs)
        
        return te_mean, te_std, te_lst
    
    def cal_bg_te(self, rounds: int = None, **kwargs) -> Tuple[float, float]:
        """
        获得背景分布均值和标准差
        
        Param:
        ------
        rounds: 重复测算次数
        """
        
        rounds = 50 if rounds is None else rounds
        te_lst = []
        
        for _ in range(rounds):
            x_shuff, y_shuff = shuffle(self.x), shuffle(self.y)
            te_bg, _, _ = self._cal_te(x_shuff, y_shuff, rounds=1, **kwargs)
            te_lst.append(te_bg)
        
        te_mean, te_std = np.nanmean(te_lst), np.nanstd(te_lst)
        
        return te_mean, te_std
    