from scipy.signal import find_peaks
from typing import List, Tuple
import numpy as np
import random


def _build_td_series(x: np.ndarray, y: np.ndarray, Z: np.ndarray, td_lag: int) -> \
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_td_, y_td_, Z_td_ = x.flatten(), y.flatten(), Z.copy()
    lag_remain = np.abs(td_lag) % len(x_td_)  # 求余数

    if td_lag == 0:
        # 没有时滞, 那么x_td和y_td_1同时发生
        x_td = x_td_[1:].copy()
        y_td = y_td_[1:].copy()
        Z_td = Z_td_[1:, :].copy()
    elif td_lag > 0:
        # 正时滞, x_td比y_td_1早lag_remain发生
        x_td = x_td_[:-lag_remain].copy()
        y_td = y_td_[lag_remain:].copy()
        Z_td = Z_td_[lag_remain:, :].copy()
    else:
        # 负时滞, x_td比y_td_1晚lag_remain发生
        x_td = x_td_[lag_remain + 1:].copy()
        y_td = y_td_[1: -lag_remain].copy()
        Z_td = Z_td_[1: -lag_remain, :].copy()
        
    return x_td, y_td, Z_td


def _shuffle(x: np.ndarray) -> np.ndarray:
    idxs = np.arange(len(x))
    idxs_shuff = np.random.choice(idxs, len(x), replace=True)
    
    if len(x.shape) == 1:
        x_srg = x[idxs_shuff]
    elif len(x.shape) == 2:
        x_srg = x[idxs_shuff, :]
    else:
        raise RuntimeError
    
    return x_srg


class TransferEntropy(object):
    """传递熵时序因果检验"""
    
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
            raise RuntimeError("x,y和Z的序列值必须均为整数")
        
        self.x = x.flatten()
        self.y = y.flatten()
        self.Z = Z.copy().reshape(len(Z), -1)
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.alpha = alpha
        self.N = len(self.x)
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1  # X和Y的窗口样本数以及Y的超前时间
        
    def _cal_te(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray, sub_sample_size: int = None, rounds: int = None) -> \
        Tuple[float, float, List[float]]:
        """
        计算特定时延构造的时序样本的传递熵
        
        Params:
        -------
        x: 待计算的X序列
        y: 待计算的Y序列
        sub_sample_size: 每次计算的子采样数
        rounds: 重复次数
        """
        
        # 构造样本
        idxs = np.arange(0, len(x) - self.h * self.tau_y - 1)
        _xk = x[idxs].reshape(-1, 1)
        _Zk = Z[idxs, :]
        _yk = y[idxs].reshape(-1, 1)
        _yh = _yk.copy()
        
        concat_xzyy = np.concatenate([
            _xk[self.h * self.tau_y:], 
            _Zk[self.h * self.tau_y:], 
            _yk[: -self.h * self.tau_y],
            _yh[self.h * self.tau_y:], 
            ], axis=1)
        
        states_xzyy = np.unique(concat_xzyy, axis = 0)
        _N = concat_xzyy.shape[0]
        
        # 进行多轮随机抽样计算
        sub_sample_size = _N if sub_sample_size is None else sub_sample_size
        rounds = 10 if rounds is None else rounds
        
        eps = 1e-6
        te_lst = []
        
        for _ in range(rounds):
            _idxs = random.sample(range(_N), sub_sample_size)
            _concat_xzyy = concat_xzyy[_idxs, :]
            
            _te = 0.0
            for state in states_xzyy:
                # P(xk, zk, yk, yh)
                prob1 = (_concat_xzyy == state).all(axis = 1).sum() / sub_sample_size
                
                # P(zk, yk)
                prob2 = (_concat_xzyy[:, 1 : -1] == state[1 : -1]).all(axis = 1).sum() / sub_sample_size
                
                # P(xk, zk, yk)
                prob3 = (_concat_xzyy[:, : -1] == state[: -1]).all(axis = 1).sum() / sub_sample_size
                
                # p(zk, yk, yh)
                prob4 = (_concat_xzyy[:, 1 :] == state[1 :]).all(axis = 1).sum() / sub_sample_size

                prob = prob1 * np.log2((prob1 * prob2) / (prob3 * prob4 + eps) + eps)
                
                if np.isnan(prob) == False:
                    _te += prob
                    
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
        
        x_td, y_td, Z_td = _build_td_series(self.x, self.y, self.Z, td_lag)
        te_mean, te_std, te_lst = self._cal_te(x_td, y_td, Z_td)
        
        return te_mean, te_std, te_lst
    
    def cal_bg_te(self, rounds: int = None) -> Tuple[float, float]:
        """
        获得背景分布均值和标准差
        
        Param:
        ------
        rounds: 重复测算次数
        """
        
        rounds = 50 if rounds is None else rounds
        te_lst = []
        
        for _ in range(rounds):
            x_shuff, y_shuff, Z_shuff = _shuffle(self.x), _shuffle(self.y), _shuffle(self.Z)
            te_bg, _, _ = self._cal_te(x_shuff, y_shuff, Z_shuff, rounds=1)
            te_lst.append(te_bg)
        
        te_mean, te_std = np.nanmean(te_lst), np.nanstd(te_lst)
        
        return te_mean, te_std