from sklearn.utils import resample, shuffle
from typing import List, Tuple
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.util.adapt_symbolize import adapt_symbolize
from giefstat.util import build_td_series
from giefstat.coefficient import cal_assoc
from giefstat.pyitlib import discrete_random_variable as drv


def build_iid_td_samples(x: np.ndarray, y: np.ndarray, td_lag: int, tau_iid: int = 20, 
                         iid_size: int = None) -> np.ndarray:
    """
    构建独立同分布的时延样本
    """
    
    # 构建时延样本
    x_td, y_td =  build_td_series(x, y, td_lag)
    arr = np.c_[x_td, y_td]

    # 采集独立同分布样本
    iid_size = len(arr) // tau_iid - 1 if iid_size is None else iid_size
    idxs = np.arange(0, tau_iid * (iid_size + 1), tau_iid)

    return arr[idxs, :]


def cal_bootstrap_coeffs(x_iid: np.ndarray, y_iid: np.ndarray, bt_rounds: int = 100, method: str = "drv") -> \
    Tuple[float, float, List[float]]:
    """
    基于Bootstrap采样计算相关系数的均值和标准差
    
    Params:
    -------
    x_iid: x的独立同分布样本
    y_iid: y的独立同分布样本
    bt_rounds: Bootstrap采样次数
    method: 相关系数计算方法
    """
    
    arr_iid = np.c_[x_iid, y_iid]
    
    coeff_lst = []
    idxs = np.arange(x_iid.shape[0])
    
    for _ in range(bt_rounds):
        # Bootstrap采样
        _idxs_bt = resample(idxs, replace=True)
        _arr = arr_iid.copy()[_idxs_bt, :]
        
        if method == "drv":
            coeff = drv.information_mutual(_arr[:, 0], _arr[:, 1])  # NOTE：如果是连续值则将此函数替换为Kraskov方法
        elif method == "MI-GIEF":
            coeff = cal_assoc(_arr[:, 0].astype(np.float32), _arr[:, 1].astype(np.float32), "MI-GIEF", "c", "c")
        else:
            raise ValueError("method参数错误，不被支持")
        
        coeff_lst.append(coeff)
        
    coeff_mean = np.nanmean(coeff_lst)
    coeff_std = np.nanstd(coeff_lst)
    
    return coeff_mean, coeff_std, coeff_lst


class TimeDelayedAssoc(object):
    """
    时间延迟关联分析
    
    这段代码定义了一个名为TimeDelayedAssoc的类，用于进行时间延迟关联分析。该类具有以下几个方法：

    __init__(self, x: np.ndarray, y: np.ndarray, method: str = "drv") -> None：初始化方法，接收两个参数x和y，分别表示输入的两个序列。该方法会检查x和y的数据类型是否为整数，如果不是整数则会抛出RuntimeError异常。然后将x和y展平为一维数组，并保存到实例变量中。

    cal_bootstrap_assoc(self, td_lag: int, tau_iid: int, iid_size: int = None, bt_rounds: int = 100)：计算时延关联值的方法。该方法接收四个参数：td_lag表示时延的长度，tau_iid表示独立同分布样本的长度，iid_size表示独立同分布样本的大小（可选，默认为None），bt_rounds表示Bootstrap采样的轮数（可选，默认为100）。该方法首先调用build_iid_td_samples函数构建独立同分布的时延样本，然后调用cal_bootstrap_coeffs函数进行Bootstrap采样，计算关联系数的均值、标准差和列表，并将结果返回。

    cal_bg_assoc(self, td_lag: int, tau_iid: int, iid_size: int = None, pm_rounds: int = 10, bt_rounds: int = 10)：计算时延关联值的方法。该方法与cal_bootstrap_assoc方法类似，不同之处在于它进行了多次的Bootstrap采样。首先，它调用build_iid_td_samples函数构建独立同分布的时延样本。然后，它使用shuffle函数对其中一个序列进行随机重排，构建新的样本。接下来，它使用cal_bootstrap_coeffs函数进行Bootstrap采样，计算关联系数的均值、标准差和列表。重复这个过程多次（由pm_rounds参数指定），并将每次的结果保存到一个列表中。最后，它计算关联系数列表的均值和标准差，并将结果返回。

    这些方法都依赖于一些未定义的函数，如build_iid_td_samples和cal_bootstrap_coeffs。你需要确保这些函数在代码中的其他位置被定义和实现。
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, method: str = "drv") -> None:
        try:
            assert "int" in str(x.dtype)
            assert "int" in str(y.dtype)
        except Exception as _:
            raise RuntimeError("x和y序列值必须均为整数") from _
        
        self.x = x.flatten()
        self.y = y.flatten()
        self.N = len(self.x)
        self.method = method
        
    def cal_bootstrap_assoc(self, td_lag: int, tau_iid: int, iid_size: int = None, 
                            bt_rounds: int = 100):
        """
        计算时延关联值
        """
            
        # 构建独立同分布的时延样本
        arr_iid = build_iid_td_samples(self.x, self.y, td_lag, tau_iid, iid_size)
        
        # 进行Bootstrap采样
        coeff_mean, coeff_std, coeff_lst = cal_bootstrap_coeffs(arr_iid[:, 0], arr_iid[:, 1], bt_rounds, self.method)
            
        return coeff_mean, coeff_std, coeff_lst
    
    def cal_bg_assoc(self, td_lag: int, tau_iid: int, iid_size: int = None, 
                     pm_rounds: int = 10, bt_rounds: int = 10):
        """
        计算时延关联值
        """
            
        # 构建独立同分布的时延样本
        arr_iid = build_iid_td_samples(self.x, self.y, td_lag, tau_iid, iid_size)
        
        coeff_lst = []
        
        for _ in range(pm_rounds):
            x_shuff = shuffle(arr_iid[:, 0])
            _arr_shuff = np.c_[x_shuff, arr_iid[:, 1]]
        
            # 进行Bootstrap采样
            _, _, _coeff_lst = cal_bootstrap_coeffs(_arr_shuff[:, 0], _arr_shuff[:, 1], bt_rounds, self.method)
            
            coeff_lst.extend(_coeff_lst)
            
        coeff_mean = np.nanmean(coeff_lst)
        coeff_std = np.nanstd(coeff_lst)
            
        return coeff_mean, coeff_std, coeff_lst
        
        
if __name__ == "__main__":
    
    # ---- 载入数据 ---------------------------------------------------------------------------------
    
    data = pd.read_csv(f"{BASE_DIR}/dataset/chemical_process/petro_data.csv")
    
    cols = ["Y"] + [f"X{i}" for i in range(1, 18 + 1)]
    data = data.iloc[:-1][cols].astype(float)
    
    # 使用pandas对data进行插值
    for col in data.columns:
        data[col] = data[col].interpolate(method="linear", limit_direction="both")
    
    # ---- 时延关联检测 ------------------------------------------------------------------------------
    
    for i in range(1, 19):
    
        x, y = data[f"X{i}"].values, data["Y"].values
        
        # ---- 自适应离散化 --------------------------------------------------------------------------
        
        q = 20
        x_sym, q_x = adapt_symbolize(x, q=q)
        y_sym, q_y = adapt_symbolize(y, q=q)
        
        # ---- ACF检验 -----------------------------------------------------------------------------
        
        self = TimeDelayedAssoc(x_sym, y_sym, method="MI-GIEF")
        
        tau_iid = 30
        h = 10
        td_lags = np.arange(-200, 200 + h, h)
        
        coeff_mean_lst, coeff_std_lst = [], []
        coeff_bg_mean_lst, coeff_bg_std_lst = [], []
        
        for td_lag in td_lags:
            print(f"td_lag: {td_lag} ...", end="\r")
            
            # 计算关联值
            coeff_mean, coeff_std, _ = self.cal_bootstrap_assoc(td_lag, tau_iid=tau_iid, bt_rounds=100)
            coeff_mean_lst.append(coeff_mean)
            coeff_std_lst.append(coeff_std)
            
            # 计算背景值
            coeff_mean, coeff_std, _ = self.cal_bg_assoc(td_lag, tau_iid=tau_iid, pm_rounds=10, bt_rounds=10)
            coeff_bg_mean_lst.append(coeff_mean)
            coeff_bg_std_lst.append(coeff_std)
        
        plt.figure(figsize=(6, 4))
        plt.errorbar(td_lags, coeff_mean_lst, yerr=3 * np.array(coeff_std_lst), fmt="o", capsize=3, 
                    color="b", alpha=0.5, label="bootstrap")
        plt.errorbar(td_lags, coeff_bg_mean_lst, yerr=3 * np.array(coeff_bg_std_lst), fmt="o", capsize=3, 
                    color="r", alpha=0.5, label="background")
        plt.grid(linewidth=0.5, alpha=0.5)
        # plt.ylim([-0.1, 1.1])
        plt.title(f"X{i} vs Y")
        plt.show()