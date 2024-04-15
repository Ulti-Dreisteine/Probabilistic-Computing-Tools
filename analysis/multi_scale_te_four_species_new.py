from itertools import permutations
import seaborn as sns
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.time_series.transfer_entropy import TransferEntropy  # 已经针对STE进行了修改


def gen_data(rounds: int):
    eps = 1e-3
    X = np.ones(4).reshape(1, -1) * 0.4
    for i in range(rounds + 5000):
        x = X[i, :].copy()
        x_new = [
            x[0] * (3.9 - 3.9 * x[0]) + np.random.uniform(-1, 1) * eps,
            x[1] * (3.6 - 0.4 * x[0] - 3.6 * x[1]) + np.random.uniform(-1, 1) * eps,
            x[2] * (3.6 - 0.4 * x[1] - 3.6 * x[2]) + np.random.uniform(-1, 1) * eps,
            x[3] * (3.8 - 0.35 * x[2] - 3.8 * x[3]) + np.random.uniform(-1, 1) * eps 
        ]
        X = np.r_[X, np.array(x_new).reshape(1, -1)]
    return X[5000:]


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
    
    # ---- 数据生成 ---------------------------------------------------------------------------------

    # NOTE: 对应 0->1, 1->2, 2->3, 3->1
    arr = gen_data(5000)

    N, D = arr.shape

    plt.figure(figsize=(3, 3))
    for i in range(D):
        plt.subplot(D, 1, i + 1)
        plt.plot(arr[:500, i], "k", linewidth=0.5)
        ax = plt.gca()
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_color("none")
        plt.xticks([], [])
        plt.yticks([], [])
    plt.tight_layout()

    # ---- 检测分析 ---------------------------------------------------------------------------------

    idx_x, idx_y = 3, 1
    x, y = arr[:, idx_x], arr[:, idx_y]

    m = 3
    sub_sample_size, rounds = 500, 3  # 参数提醒

    # 时间尺度
    tau_max = 20
    taus = np.arange(1, tau_max + 1, 1)

    # 时延
    lags = np.arange(-10, (10 + 1) * 1, 1)

    te_mtx = np.zeros((len(taus), len(taus)))
    lag_mtx = np.zeros((len(taus), len(taus)))
    for i, tau_x in enumerate(taus):
        for j, tau_y in enumerate(taus):
            print("\r" + f"processing tau_x {tau_x}, tau_y {tau_y}", end="")

            x_sym = symbolize(x, tau_x, m, tau_max=20)
            y_sym = symbolize(y, tau_y, m, tau_max=20)

            self = TransferEntropy(x_sym, y_sym, tau_x, tau_y)

            _td_te = []
            for lag in lags:
                te_mean, te_std, te_lst = self.cal_td_te(
                    lag * tau_x, sub_sample_size=sub_sample_size, rounds=rounds)
                _td_te.append(te_mean)

            te_mtx[i, j] = np.max(_td_te)
            lag_mtx[i, j] = lags[np.argmax(_td_te)] * tau_x

    plt.figure()
    sns.heatmap(te_mtx, cmap="rainbow")
    sns.heatmap(lag_mtx, cmap="RdBu_r")
    
    # ---- 特征时间参数 ------------------------------------------------------------------------------
    
    from core.acf_test import SelfAssoc, show_td_analysis_results
    
    td_lags = np.arange(0, 10 + 1, 1)
    self = SelfAssoc(x, sub_sample_size, rounds)
    td_assocs, td_assocs_srg = self.cal_td_assoc_dists(td_lags)
    avg_td_assocs = np.array([np.nanmean(p) for p in td_assocs])
    
    plt.figure()
    plt.plot(avg_td_assocs)