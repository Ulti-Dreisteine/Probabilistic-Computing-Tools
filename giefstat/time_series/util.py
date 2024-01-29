from typing import Tuple, Optional, Union, List
from scipy.signal import find_peaks
from itertools import permutations
import numpy as np


def shuffle(x: np.ndarray) -> np.ndarray:
    idxs = np.arange(len(x))
    idxs_shuff = np.random.choice(idxs, len(x), replace=True)  # 有返回抽样
    
    if len(x.shape) == 1:
        x_srg = x[idxs_shuff]
    elif len(x.shape) == 2:
        x_srg = x[idxs_shuff, :]
    else:
        raise RuntimeError
    
    return x_srg


# ---- 时延序列构建 ---------------------------------------------------------------------------------

def build_td_series(x: np.ndarray, y: np.ndarray, td_lag: int, Z: Optional[np.ndarray] = None) -> \
    Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    构建延迟序列
    
    Params:
    -------
    x: X的一维序列
    y: Y的一维序列
    td_lag: X->Y的以样本单位计的时延
    Z: 多维条件数组
    """
    
    x_td_, y_td_ = x.flatten(), y.flatten()
    Z_td_ = Z.copy().reshape(len(Z), -1) if Z is not None else None

    lag_remain = np.abs(td_lag) % len(x_td_)  # 求余数

    if td_lag == 0:
        # 没有时滞, 那么x_td和y_td_1同时发生
        x_td = x_td_[1:].copy()
        y_td = y_td_[1:].copy()
        Z_td = Z_td_[1:, :].copy() if Z is not None else None
    elif td_lag > 0:
        # 正时滞, x_td比y_td_1早lag_remain发生
        x_td = x_td_[:-lag_remain].copy()
        y_td = y_td_[lag_remain:].copy()
        Z_td = Z_td_[lag_remain:, :].copy() if Z is not None else None
    else:
        # 负时滞, x_td比y_td_1晚lag_remain发生
        x_td = x_td_[lag_remain + 1:].copy()
        y_td = y_td_[1: -lag_remain].copy()
        Z_td = Z_td_[1: -lag_remain, :].copy() if Z is not None else None

    return (x_td, y_td) if Z is None else (x_td, y_td, Z_td) # type: ignore


def _build_td_series(x: np.ndarray, y: np.ndarray, td_lag: int) -> Tuple[np.ndarray, np.ndarray]:
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


# ---- 时延传递熵峰值解析 ----------------------------------------------------------------------------

def parse_peaks(tau_x: int, td_lags: np.ndarray, td_te_info: List[tuple], ci_bg_ub: float, 
                thres: Optional[float] = None, distance: Optional[int] = None, 
                prominence: Optional[float] = None):
    """
    从时延TE结果中寻找是否有高于阈值的一个或多个峰值, 如果没有则默认峰在0时延处
    
    Params:
    -------
    tau_x: 因变量的特征时间参数, 以样本为单位计
    td_lags: 检测用的作用时延序列
    td_te_info: 包含了各作用时延td_lag上的te均值和方差的列表, 形如 [(te_mean@td_lag_1, te_std@td_lag_1), ...]
    ci_bg_ub: 从te背景值结果中解析获得的CI上界
    distance: 相邻两个峰的最小间距, 见scipy.signal.find_peaks()
    prominence: 在wlen范围内至少超过最低值的程度, 见scipy.signal.find_peaks()
    """
    
    if thres is None:
        thres = ci_bg_ub
        
    if distance is None:
        distance = 2 * tau_x
        
    if prominence is None:
        prominence = 0.01
    
    td_te_means = [p[0] for p in td_te_info]
    td_te_stds = [p[1] for p in td_te_info]
    
    peak_idxs, _ = find_peaks(
        td_te_means, height=thres, distance=distance, prominence=prominence, 
        wlen=max([2, len(td_lags) // 2]))

    peak_signifs = []
    
    if len(peak_idxs) == 0:
        peak_taus = []
        peak_strengths = []
        peak_stds = []
    else:
        # 获得对应峰时延、强度和显著性信息
        peak_taus = [td_lags[p] for p in peak_idxs]
        peak_strengths = [td_te_means[p] for p in peak_idxs]
        peak_stds = [td_te_stds[p] for p in peak_idxs]

        for idx in peak_idxs:
            _n = len(td_lags) // 10
            _series = np.append(td_te_means[: _n], td_te_means[-_n :])
            _mean, _std = np.mean(_series), np.std(_series)
            signif = (td_te_means[idx] > ci_bg_ub) & (td_te_means[idx] > _mean + 3 * _std)  # 99% CI
            peak_signifs.append(signif)

    return peak_idxs, peak_taus, peak_strengths, peak_stds, peak_signifs


# ---- 序列符号化 -----------------------------------------------------------------------------------

def _build_embed_series(x: np.ndarray, idxs: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    构造一维序列x的m维嵌入序列
    
    Params:
    -------
    x: 一维序列
    idxs: 首尾截断(避免空值)后的连续索引序列
    m: 嵌入维度
    tau: 嵌入延迟(以样本为单位计), 应等于序列自身的特征时间参数
    """
    
    X_embed = x[idxs]
    
    for i in range(1, m):
        X_embed = np.c_[x[idxs - i * tau], X_embed]
        
    return X_embed.reshape(len(X_embed), -1)


def continuously_symbolize(x: np.ndarray, y: np.ndarray, m: int, tau_x: int, tau_y: int) -> \
    Tuple[np.ndarray, np.ndarray]:
    """
    生成具有连续索引的符号样本
    
    Params:
    -------
    x: X序列
    y: Y序列
    m: 嵌入维度
    tau_x: X序列的嵌入延迟(以样本为单位计), 应等于序列自身的特征时间参数
    tau_y: Y序列的嵌入延迟(以样本为单位计), 应等于序列自身的特征时间参数
    """
    
    # 提取所有可能的离散模式
    patterns = list(permutations(np.arange(m) + 1))
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    # 构造嵌入序列
    idxs = np.arange((m - 1) * max(tau_x, tau_y), len(x))       # 连续索引
    X_embed = _build_embed_series(x, idxs, m, tau_x)
    Y_embed = _build_embed_series(y, idxs, m, tau_y)
    
    # 获得对应的索引
    X = np.argsort(X_embed) + 1                                 # 滚动形成m维时延嵌入样本  一个时刻成为一个标签
    X = np.array([dict_pattern_index[tuple(p)] for p in X])     # 对应映射到符号上
    
    Y = np.argsort(Y_embed) + 1
    Y = np.array([dict_pattern_index[tuple(p)] for p in Y])
    
    return X.flatten(), Y.flatten()