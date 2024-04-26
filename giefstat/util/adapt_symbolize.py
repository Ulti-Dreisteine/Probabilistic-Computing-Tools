# -*- coding: utf-8 -*-
"""
Created on 2024/04/26 14:56:10

@File -> adapt_symbolize.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 自适应符号化
"""

from typing import Tuple
from math import factorial
import pandas as pd
import numpy as np


def savitzky_golay(y, window_size, order, deriv = 0, rate = 1) -> np.ndarray:
    """
    savitzky_golay滤波
    
    Reference:
    ----------
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    """
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except Exception as _:
        raise ValueError("window_size and order have to be of type int") from _

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # 预计算系数.
    b = np.mat([[k ** i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv) # type: ignore

    # pad the signal at the extremes with values taken from the signal itself.
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode = "valid")


def symbolize(x: np.ndarray, q: int) -> np.ndarray:
    """
    一维（平稳或非平稳）序列符号化
    
    Params:
    -------
    x: 待处理序列
    q: 离散化个数
    """
    x = x.flatten().astype(np.float32)  # type: np.ndarray
    return pd.qcut(x, q, labels=False, duplicates="drop").astype(int)


def cal_snr(x, x_denoise) -> float:
    """
    计算信噪比
    """
    x, x_denoise = x.flatten(), x_denoise.flatten()
    return 10 * np.log10(np.sum(x_denoise ** 2) / np.sum((x - x_denoise) ** 2))


def adapt_symbolize(x: np.ndarray, q: int = None, sg_window_size: int = 3, sg_order: int = 1, 
                    q_max: int = 10, eps: float = 1e-12) -> Tuple[np.ndarray, int]:
    """
    一维序列自适应符号化
    
    Params:
    -------
    x: 待处理序列
    sg_window_size: Savitzky-Golay滤波窗口大小
    sg_order: Savitzky-Golay滤波阶数
    q_max: 符号化离散化个数上限
    eps: 防止除数为0
    """
    
    if q is not None:
        x_sym, q_opt = symbolize(x, q), q
    else:
        x += + np.random.normal(0, 0.01 * np.max(x), len(x))
        x_sg = savitzky_golay(x, window_size=sg_window_size, order=sg_order)

        # 根据信噪比之比选出最佳的参数q_opt
        metrics = []
        qs = np.arange(2, q_max)

        for q in qs:
            x_sym = symbolize(x, q)

            snr_in = cal_snr(x, x_sg)
            snr_out = cal_snr(x_sym, x_sg)

            metrics.append(snr_in / (snr_out + eps))

        # 符号化
        q_opt = qs[np.argmin(metrics)]
        x_sym = symbolize(x, q_opt)

    return x_sym, q_opt


# if __name__ == "__main__":
#     import pandas as pd
#     import sys
#     import os
    
#     BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
#     sys.path.insert(0, BASE_DIR)
    
#     from setting import plt
    
#     # ---- 载入数据 ---------------------------------------------------------------------------------
    
#     data = pd.read_csv(f"{BASE_DIR}/dataset/fcc/X.csv")
#     x = data.values[:, 3]
    
#     # ---- 符号化 -----------------------------------------------------------------------------------
    
#     x_sym = adapt_symbolize(x)
    
#     plt.subplot(2, 1, 1)
#     plt.plot(x)
#     plt.subplot(2, 1, 2)
#     plt.plot(x_sym)