# -*- coding: utf-8 -*-
"""
Created on 2024/03/12 15:17:08

@File -> multi_temporal_te.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 多时间尺度的传递熵分析
"""

from itertools import permutations
import seaborn as sns
import pandas as pd
import numpy as np
import json
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.time_series.transfer_entropy import TransferEntropy  # 已经针对STE进行了修改


def load_json_file(fp: str) -> dict:
    with open(fp, "r") as f:
        results = json.load(f)
    return results


def _gen_embed_series(x, idxs, m, tau):
    X_embed = x.copy()[idxs]
    for i in range(1, m):
        X_embed = np.c_[x[idxs - i * tau], X_embed]
    return X_embed


def symbolize(x: np.ndarray, tau, m, tau_max):
    """符号化"""
    # 确定所有模式集合
    patterns = list(permutations(np.arange(m) + 1))
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    idxs = np.arange((m - 1) * tau_max, len(x))  # 连续索引
    X_embed = _gen_embed_series(x, idxs, m, tau)
    
    X = np.argsort(X_embed) + 1  # NOTE: 滚动形成m维时延嵌入样本  一个时刻成为一个标签
    X = np.array([dict_pattern_index[tuple(p)] for p in X])  # 对应映射到符号上
    return X


if __name__ == "__main__":
    data = pd.read_csv(f"{BASE_DIR}/dataset/chemical_process/petro_data.csv")
    taus = load_json_file("taus.json")  # NOTE: 必须先求解时间参数
    
    cols = ["Y"] + [f"X{i}" for i in range(1, 19)]
    x_col, y_col = "X4", "Y"
    x = data[x_col].values
    y = data[y_col].values
    
    tau_x, tau_y = taus[str(cols.index(x_col))], taus[str(cols.index(y_col))]
    
    # ---- 符号序列化 --------------------------------------------------------------------------------
    
    m = 3
    
    x_lags = np.arange(-30 * tau_x, (30 + 1) * tau_x, tau_x)
    
    # NOTE: 待检测的y间隔有个较大跨度：自回归、因果检测、关联检测
    y_gaps = np.arange(1, int(3 * tau_y) + 1, 1)
    
    sub_sample_size, rounds = 600, 10  # 参数提醒
    
    te_mtx = np.zeros((len(x_lags), len(y_gaps)))
    for i, x_lag in enumerate(x_lags):
        for j, y_gap in enumerate(y_gaps):
            print("\r" + f"processing x_lag: {x_lag}, y_gap: {y_gap}", end="")
            
            x_sym = symbolize(x, tau_x, m, tau_max=20)
            y_sym = symbolize(y, y_gap, m, tau_max=20)  # NOTE: 自适应符号化
            
            self = TransferEntropy(x_sym, y_sym, tau_x, y_gap)
            kwargs = {"sub_sample_size": sub_sample_size, "rounds": rounds}
            te_mean, te_std, _ = self.cal_td_te(x_lag, **kwargs)
            te_mtx[i, j] = te_mean
            
    sns.heatmap(te_mtx, cmap="Reds")
    plt.xlabel("y_gap")
    plt.ylabel("x_lag")
    
    