import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.time_series.td_assoc_analysis import acf_test
from giefstat.time_series.util import continuously_symbolize, parse_peaks
from giefstat.time_series.transfer_entropy import TransferEntropy

if __name__ == "__main__":
    
    # ---- 载入数据 ---------------------------------------------------------------------------------
    
    data = pd.read_csv(f"{BASE_DIR}/dataset/time_delayed/siso.csv")
    
    label = "nonlinear_static"
    x = data[f"x_{label}"].values
    y = data[f"y_{label}"].values
    
    # ---- 获得特征时间参数 --------------------------------------------------------------------------
    
    # taus = np.arange(-20, 20 + 1, 1)
    # _ = acf_test(x, taus, "c", show=True, rounds=100)
    # _ = acf_test(y, taus, "c", show=True, rounds=100)
    
    tau_x, tau_y = 1, 1
    
    # ---- 构造符号化序列 ----------------------------------------------------------------------------
    
    x, y = continuously_symbolize(x, y, m=3, tau_x=tau_x, tau_y=tau_y)
    
    # ---- 时延传递熵计算 ----------------------------------------------------------------------------
    
    te = TransferEntropy(x, y, tau_x, tau_y)
    
    
    td_lags = np.arange(-10, 10 + 1, 1)
    td_te_info = []
    
    for td_lag in td_lags:
        print(td_lag)
        te_mean, te_std, _ = te.cal_td_te(td_lag)
        td_te_info.append((te_mean, te_std))
    
    # 背景值
    te_bg_mean, te_bg_std =te.cal_bg_te(rounds=50)
    ci_bg_ub = te_bg_mean + 3 * te_bg_std  # 均值 + 整数倍标准差  # 单侧检验
    
    # 解析峰值
    peak_idxs, peak_taus, peak_strengths, peak_stds, peak_signifs = parse_peaks(tau_x, td_lags, td_te_info, ci_bg_ub)
    
    # 画图
    td_te_means = [p[0] for p in td_te_info]
    bounds = [0, np.max(peak_strengths) * 1.1]  # 画图的上下边界
    
    plt.figure(figsize=(3, 3))
    plt.fill_between(td_lags, td_te_means, np.zeros_like(td_te_means))
    plt.hlines(ci_bg_ub, td_lags.min(), td_lags.max(), linestyles="-", linewidth=0.5, colors="r")
    plt.ylim(*bounds)
    plt.vlines(0, *bounds, colors="k", linewidth=1.0, zorder=2)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$T_{X \rightarrow Y@\tau}^{\rm s}$")
    plt.grid(True, zorder=-1)
    plt.tight_layout()
    
    
    
    