import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.time_series.util import continuously_symbolize, parse_peaks
from giefstat.time_series.transfer_entropy import TransferEntropy

if __name__ == "__main__":
    
    # ---- 载入数据 ---------------------------------------------------------------------------------
    
    arr = np.load(f"{BASE_DIR}/dataset/time_delayed/cyclic_symbols.npy")
    
    # ---- 分析 -------------------------------------------------------------------------------------
    
    idx_x, idx_y = 3, 1
    tau_x, tau_y = 1, 1
    
    x, y = arr[:, idx_x], arr[:, idx_y]
    
    te = TransferEntropy(x, y, tau_x, tau_y)
    
    # 时延检测
    td_lags = np.arange(-12, 12 + 1, 1)
    td_te_info = []
    
    for td_lag in td_lags:
        print(td_lag)
        te_mean, te_std, _ = te.cal_td_te(td_lag)
        td_te_info.append((te_mean, te_std))
    
    # 背景值
    te_bg_mean, te_bg_std = te.cal_bg_te(rounds=100)
    ci_bg_ub = te_bg_mean + 3 * te_bg_std  # 均值 + 整数倍标准差  # 单侧检验
    
    # 解析峰值
    peak_idxs, peak_taus, peak_strengths, peak_stds, peak_signifs = \
        parse_peaks(tau_x, td_lags, td_te_info, ci_bg_ub)
    
    # 画图
    td_te_means = [p[0] for p in td_te_info]
    bounds = [0, np.max(peak_strengths) * 1.1] if peak_strengths else [0, 0.1]  # 画图的上下边界
    
    plt.figure(figsize=(3, 3))
    plt.bar(td_lags, td_te_means)
    plt.hlines(ci_bg_ub, td_lags.min(), td_lags.max(), linestyles="-", linewidth=0.5, colors="r")
    plt.ylim(*bounds)
    plt.vlines(0, *bounds, colors="k", linewidth=1.0, zorder=2)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$T_{X \rightarrow Y@\tau}^{\rm s}$")
    plt.grid(True, zorder=-1)
    plt.tight_layout()
    