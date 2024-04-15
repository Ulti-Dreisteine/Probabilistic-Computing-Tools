from sklearn.preprocessing import MinMaxScaler
from itertools import permutations
import scipy.io as scio
import seaborn as sns
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.time_series.transfer_entropy import TransferEntropy  # 已经针对STE进行了修改


def load_data(normalize: bool = True, h=100):
    data = scio.loadmat(f"{BASE_DIR}/dataset/kuehrt_cstr/data.mat")
    cols = ["T_fl", "cin", "cA", "cB", "cC"]
    
    X = None
    idxs = np.arange(0, 700000, h)  # note 当间隔取小时, CMI的surrog背景值偏低, 可能跟样本不满足iid有关
    for col in cols:
        x = data[col][0, idxs].T
        X = x if X is None else np.c_[X, x]
    
    # 归一化
    if normalize:
        X = MinMaxScaler().fit_transform(X)
    
    return X, cols


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
    arr, cols = load_data(h=3)
    
    arr += np.random.normal(0, 0.03, arr.shape)
    
    arr = arr[-10000:, :]
    N, D = arr.shape
    
    plt.figure(figsize=(3, 3))
    for i in range(D):
        plt.subplot(D, 1, i + 1)
        plt.plot(arr[:1000, i], "k")
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xticks([], [])
        plt.yticks([], [])
    plt.tight_layout()
    
    # ---- 因果检测 ---------------------------------------------------------------------------------
    
    idx_x, idx_y = 1, 3
    x, y = arr[:, idx_x], arr[:, idx_y]
    tau_x, tau_y = 1, 1
    h = np.min([tau_x, tau_y])
    # h = 40
    
    m = 2
    sub_sample_size, rounds = 500, 3  # 参数提醒
    
    len_lags = 30
    tau_max = h * 50
    taus = np.arange(h, tau_max + h, h)  # 时间尺度
    
    te_mtx = np.zeros((len_lags * 2 + 1, len(taus)))
    
    for i, tau in enumerate(taus):
        print("\r" + "processing tau %s" % (tau), end="")
            
        x_sym = symbolize(x, tau, m, tau_max=tau_max)
        y_sym = symbolize(y, tau, m, tau_max=tau_max)  # NOTE: 自适应符号化
        
        self = TransferEntropy(x_sym, y_sym, tau, tau)
        
        # <<<<<<<<
        # lags = np.arange(-len_lags * tau, (len_lags + 1) * tau, tau)
        # >>>>>>>>
        lags = np.arange(-len_lags * h, (len_lags + 1) * h, h)
        
        for j, lag in enumerate(lags):
            kwargs = {"sub_sample_size": sub_sample_size, "rounds": rounds}
            te_mean, te_std, _ = self.cal_td_te(lag, **kwargs)
            te_mtx[j, i] = te_mean
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(te_mtx, cmap="rainbow")
    plt.xlabel(r"time scale $\tau$")
    plt.xticks([0 + 0.5, 9 + 0.5, 19 + 0.5], [1, 10, 20], rotation=0)
    plt.ylabel(r"time delay $X \rightarrow Y$")
    plt.yticks([0 + 0.5, 10 + 0.5, 20 + 0.5], [r"-10$\times\tau$", 0, r"10$\times\tau$"], rotation=0)
    plt.title(r"Transfer Entropy $X \rightarrow Y$ at different time scales and delays", fontsize=14)
    plt.hlines(10 + 0.5, 0, 160, colors="black", linestyles="dashed")
    plt.tight_layout()
    
    # plt.savefig("te_mtx.png", dpi=450)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x[:100])
    plt.ylabel(r"$X$", font="Times New Roman")
    plt.grid(linewidth=0.8)
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")

    plt.subplot(2, 1, 2)
    plt.plot(y[:100])
    plt.ylabel(r"$Y$", font="Times New Roman")
    plt.grid(linewidth=0.8)
    plt.xlabel("time", font="Times New Roman")
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")

    plt.tight_layout()
    # plt.savefig("time_series.png", dpi=450)
    
    # ---- 特征时间参数 ------------------------------------------------------------------------------
    
    from core.acf_test import SelfAssoc, show_td_analysis_results
    
    td_lags = np.arange(0, 10 + 1, 1)
    self = SelfAssoc(y, sub_sample_size, rounds)
    td_assocs, td_assocs_srg = self.cal_td_assoc_dists(td_lags)
    avg_td_assocs = np.array([np.nanmean(p) for p in td_assocs])
    
    plt.figure()
    plt.plot(avg_td_assocs)