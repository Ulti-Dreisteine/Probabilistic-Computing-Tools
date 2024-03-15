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
    x_col, y_col = "X17", "Y"
    x = data[x_col].values
    y = data[y_col].values
    
    tau_x, tau_y = taus[str(cols.index(x_col))], taus[str(cols.index(y_col))]
    
    # ---- 多尺度时延因果检测 -------------------------------------------------------------------------
    
    m = 3
    sub_sample_size, rounds = 500, 1  # 参数提醒
    
    len_lags = 60
    
    # tau_max = int(max(tau_x, tau_y) * 20)
    tau_max = 200
    taus = np.arange(1, tau_max + 1, 1)  # 时间尺度
    
    te_mtx = np.zeros((len_lags * 2 + 1, len(taus)))
    
    for i, tau in enumerate(taus):
        print("\r" + "processing tau %s" % (tau), end="")
            
        x_sym = symbolize(x, tau, m, tau_max=tau_max)
        y_sym = symbolize(y, tau, m, tau_max=tau_max)  # NOTE: 自适应符号化
        
        self = TransferEntropy(x_sym, y_sym, tau, tau)
        
        # <<<<<<<<
        # lags = np.arange(-len_lags * tau, (len_lags + 1) * tau, tau)
        # >>>>>>>>
        lags = np.arange(-len_lags * 1, (len_lags + 1) * 1, 1)
        
        for j, lag in enumerate(lags):
            kwargs = {"sub_sample_size": sub_sample_size, "rounds": rounds}
            te_mean, te_std, _ = self.cal_td_te(lag, **kwargs)
            te_mtx[j, i] = te_mean
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(te_mtx, cmap="rainbow")
    plt.xlabel(r"time scale $\tau$")
    plt.xticks([0 + 0.5, 99 + 0.5, 199 + 0.5], [1, 100, 200], rotation=0)
    plt.ylabel(r"time delay $X \rightarrow Y$")
    # plt.yticks([0 + 0.5, 60 + 0.5, 120 + 0.5], [r"-60$\times\tau$", 0, r"60$\times\tau$"], rotation=0)
    plt.yticks([0 + 0.5, 60 + 0.5, 120 + 0.5], [-60, 0, 60], rotation=0)
    plt.title(r"Transfer Entropy $X \rightarrow Y$ at different time scales and delays", fontsize=14)
    plt.hlines(60 + 0.5, 0, 200, colors="black", linestyles="dashed")
    plt.tight_layout()
    
    plt.savefig("te_mtx.png", dpi=450)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.ylabel(r"$%s$" % (x_col), font="Times New Roman")
    plt.grid(linewidth=0.8)
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")

    plt.subplot(2, 1, 2)
    plt.plot(y)
    plt.ylabel(r"$Y$", font="Times New Roman")
    plt.grid(linewidth=0.8)
    plt.xlabel("time", font="Times New Roman")
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")

    plt.tight_layout()
    plt.savefig("time_series.png", dpi=450)