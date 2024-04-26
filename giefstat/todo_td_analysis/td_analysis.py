from typing import Tuple
import numpy as np
import random

from ..util.univar_encoding import SuperCategorEncoding, UnsuperCategorEncoding
from . import PairwiseCCF, PairwiseMIC, PairwiseRMIC, PairwiseTE, PairwiseKNN, PairwiseGC
from ._numpy import random_sampling, add_noise, gen_time_delayed_series


def cal_association(xs: np.ndarray, ys: np.ndarray, method: str):
    x = xs.copy()
    y = ys.copy()
    
    if method == "ccf":
        pair_ccf = PairwiseCCF(x, y)
        return pair_ccf.cal_ccf()
    elif method == "mic":
        pair_mic = PairwiseMIC(x, y)
        return pair_mic.cal_mic()
    elif method == "rmic":
        pair_rmic = PairwiseRMIC(x, y)
        return pair_rmic.cal_rmic()
    elif method == "te":
        x = add_noise(x.reshape(-1, 1), ["numeric"], noise_coeff = 1e-3, for_te = True)
        y = add_noise(y.reshape(-1, 1), ["numeric"], noise_coeff = 1e-3, for_te = True)
        pair_knn = PairwiseTE(x, y)
        return pair_knn.cal_te()
    elif method == "knn":
        pair_knn = PairwiseKNN(x, y)
        return pair_knn.cal_knn()
    elif method == "gc":
        x = add_noise(x.reshape(-1, 1), ["numeric"], noise_coeff = 1e-3, for_te = True)
        y = add_noise(y.reshape(-1, 1), ["numeric"], noise_coeff = 1e-3, for_te = True)
        pair_gc = PairwiseGC(x, y)
        return pair_gc.cal_gc()
    else:
        raise ValueError("unknown method '{}'".format(method))


def cal_time_delay_association(x: np.ndarray, y: np.ndarray, x_type: str, method: str, keep_n: int, td_lags: list,
                               seed: int = None) -> Tuple[list, list]:
    """
    计算时滞关联
    :param x: x样本数据
    :param y: y样本数据
    :param x_type: x的数值类型
    :param method: 计算方法
    :param keep_n: 每个时间步计算使用的样本量
    :param td_lags: 时滞序列
    :param seed: 随机种子
    """
    x, y = x.copy(), y.copy()
    assert method in ["rmic", "mic", "te", "knn", "gc", "ccf"]
    
    # 数据编码.
    if x_type == "categoric":
        if method == "rmic":
            print("encoding x...")
            enc = SuperCategorEncoding(x, y)
            x = enc.encode(method = "mhg")  # 这里可以选择编码方式
        elif method in ["mic", "te", "knn", "gc", "ccf"]:
            print("encoding x...")
            enc = UnsuperCategorEncoding(x)
            # x = enc.encode(method = "random", seed = 0)  # 这里可以选择编码方式
            x = enc.encode(method = "ordinal")  # 这里可以选择编码方式
        else:
            ...
    
    # 时滞平移计算.
    td_assocs = []
    for i, td_lag in enumerate(td_lags):
        print("%{:.2f}\r".format(i / len(td_lags) * 100), end = "")
    
        # 生成时滞样本序列.
        if method in ["rmic", "mic", "knn", "ccf"]:
            pass
        elif method == "te":
            td_lag += 1
        elif method == "gc":
            td_lag -= 1
        else:
            pass
    
        # 注意x, y长度随td_lag变化.
        x_td, y_td = gen_time_delayed_series(np.vstack((x, y)).T, td_lag)
    
        # 由于x, y长度可变, 这里各td_lag对应的idx结果不是相同的.
        if seed is None:
            seed = random.randint(0, 9999)
            # seed = 0
    
        xs, ys = None, None
        if method in ["rmic", "mic", "knn", "ccf"]:
            xs = random_sampling(x_td, keep_n, seed)
            ys = random_sampling(y_td, keep_n, seed)
        elif method in ["te", "gc"]:
            random.seed(seed)
            idx = random.randint(0, len(x_td) - keep_n)
            xs = x_td[idx: idx + keep_n]
            ys = y_td[idx: idx + keep_n]
        else:
            pass
    
        assoc = cal_association(xs, ys, method)
        td_assocs.append(assoc)
    
    return td_lags, td_assocs