from typing import List, Tuple, Optional
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
    偏传递熵时序因果检验
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray, tau_x: int, tau_y: int, alpha: float = 0.01):
        """
        Params:
        -------
        x: X的序列样本
        y: Y的序列样本
        Z: Z的样本
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
            assert "int" in str(Z.dtype)
        except Exception as _:
            raise RuntimeError("x,y和Z的序列值必须均为整数") from _
        
        self.x = x.flatten()
        self.y = y.flatten()
        self.Z = Z.copy().reshape(len(Z), -1)  # 转为二维np数组
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.alpha = alpha
        self.N = len(self.x)
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1  # X和Y的窗口样本数以及Y的超前时间
        
    @staticmethod
    def _sum(_arr_yyxz, _states_yyxz, eps) -> float:
        _te = 0.0
        _, D = _arr_yyxz.shape[1]
        
        for _state in _states_yyxz:
            dims_yyxz = list(range(D)) 
            prob_yyxz = cal_discrete_prob(_arr_yyxz, dims_yyxz, _states_yyxz)
            
            cdims_yxz = list(range(1, D))
            prob_y_yxz = cal_discrete_prob(_arr_yyxz, [0], _state[[0]], cdims_yxz, _state[cdims_yxz])
            
            cdims_yz = [1] + list(range(3, D))
            prob_y_yz = cal_discrete_prob(_arr_yyxz, [0], _state[[0]], cdims_yz, _state[cdims_yz])
            
            prob = prob_yyxz * np.log2(prob_y_yxz / (prob_y_yz + eps) + eps)
            
            if not np.isnan(prob):
                _te += prob
        
        return _te
        
    def _cal_te(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray, sub_sample_size: Optional[int] = None, 
                rounds: int = 10, eps: float = 1e-12) -> Tuple[float, float, List[float]]:   
        """
        计算特定时延构造的时序样本的传递熵
        
        Params:
        -------
        x: 待计算的X序列
        y: 待计算的Y序列
        Z: 条件集样本序列
        sub_sample_size: 每次计算的子采样数
        rounds: 重复次数
        """
        
        # 构造联合分布样本
        idxs = np.arange(0, len(x) - self.h * self.tau_y - 1)
        xk = x[idxs].reshape(-1, 1)
        yk = y[idxs].reshape(-1, 1)
        yh = yk.copy()
        Zk = Z[idxs, :]
        
        arr_yyxz = np.concatenate([
            yh[self.h * self.tau_y:, :],
            yk[: -self.h * self.tau_y, :], 
            xk[self.h * self.tau_y:, :],
            Zk[self.h * self.tau_y:],
            ], axis = 1)  # type: np.ndarray
        
        # 进行多轮随机抽样计算
        N = len(arr_yyxz)
        sub_sample_size = N if sub_sample_size is None else sub_sample_size
        
        te_lst = []
        
        for _ in range(rounds):
            # 随机子采样以控制样本规模
            _idxs = random.sample(range(N), sub_sample_size)
            _arr_yyxz = arr_yyxz.copy()[_idxs, :]
            _states_yyxz = np.unique(_arr_yyxz, axis = 0)
            
            _te = self._sum(_arr_yyxz, _states_yyxz, eps)
            
            te_lst.append(_te)
            
        te_mean = np.nanmean(te_lst)
        te_std = np.nanstd(te_lst)
        
        return te_mean, te_std, te_lst
    
    def cal_td_te(self, td_lag: int) -> Tuple[float, float, List[float]]:
        """
        计算X->Y@td_lag|Z处的传递熵
        
        Param:
        ------
        td_lag: X到Y的时延
        """
        
        # pylint: disable-next = unbalanced-tuple-unpacking
        x_td, y_td, Z_td = build_td_series(self.x, self.y, td_lag, self.Z) # type: ignore
        te_mean, te_std, te_lst = self._cal_te(x_td, y_td, Z_td)
        
        return te_mean, te_std, te_lst
    
    def cal_bg_te(self, rounds: int = 50) -> Tuple[float, float]:
        """
        获得背景分布均值和标准差
        
        Param:
        ------
        rounds: 重复测算次数
        """
        
        te_lst = []
        
        for _ in range(rounds):
            x_shuff, y_shuff, Z_shuff = shuffle(self.x), shuffle(self.y), shuffle(self.Z)
            te_bg, _, _ = self._cal_te(x_shuff, y_shuff, Z_shuff, rounds=1)
            te_lst.append(te_bg)
        
        te_mean, te_std = np.nanmean(te_lst), np.nanstd(te_lst)
        
        return te_mean, te_std