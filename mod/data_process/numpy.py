from typing import Optional
from math import factorial
import pandas as pd
import numpy as np

EPS = 1e-6


# ---- 数据集划分 -----------------------------------------------------------------------------------

def train_test_split(X, y, seed: int = None, test_ratio: float = 0.3):
    X, y = X.copy(), y.copy()
    assert X.shape[0] == y.shape[0]
    assert 0 <= test_ratio < 1

    if seed is not None:
        np.random.seed(seed)
        shuffled_indexes = np.random.permutation(range(len(X)))
    else:
        shuffled_indexes = np.random.permutation(range(len(X)))

    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    
    return X[train_index], X[test_index], y[train_index], y[test_index]


# ---- 高维数据压缩, 计算互信息时要用. ----------------------------------------------------------------

def _reorder_z_series(z_compress: np.ndarray):
    z_compress = z_compress.copy()
    map_ = dict(zip(set(z_compress), list(range(len(set(z_compress))))))
    vfunc = np.vectorize(lambda x: map_[x])
    z_compress = vfunc(z_compress)
    return z_compress


def compress_z_data(z_arr: np.ndarray) -> np.ndarray:
    """
    对高维数据进行压缩
    
    Params:
    -------
    z_arr: Z数组, 注意shape = (D, N)而不是(N, D)
    """
    
    D = z_arr.shape[0]
    z_label_ns = [len(set(z_arr[i, :])) for i in range(z_arr.shape[0])]

    if D == 0:
        raise ValueError("empty array z")

    if D == 1:
        z_compress = z_arr.flatten()
        z_compress = _reorder_z_series(z_compress)
    else:
        i = 2
        z_compress = np.ravel_multi_index(z_arr[:i, :], z_label_ns[:i], mode="wrap")
        z_compress = _reorder_z_series(z_compress)

        while True:
            if i == D:
                break
            else:
                _arr = np.vstack((z_compress, z_arr[i, :]))
                z_compress = np.ravel_multi_index(_arr, (len(set(z_compress)), z_label_ns[i]), mode="wrap")
                z_compress = _reorder_z_series(z_compress)
                i += 1

    return z_compress


# ---- 数据离散化 -----------------------------------------------------------------------------------

def discretize_series(x: np.ndarray, n: int = 100, method="qcut") -> Optional[np.ndarray]:
    """
    对数据序列采用等频分箱
    """
    
    q = int(len(x) // n)
    x_enc = None
    
    if method == "qcut":
        x_enc = pd.qcut(x, q, labels=False, duplicates="drop").flatten()  # 等频分箱
    elif method == "cut":
        x_enc = pd.cut(x, q, labels=False, duplicates="drop").flatten()  # 等宽分箱
    else:
        pass
    
    return x_enc


def discretize_arr(X: np.ndarray, n: int = 100, method: str = "qcut") -> np.ndarray:
    """
    逐列离散化
    """
    
    X = X.copy()
    
    for i in range(X.shape[1]):
        X[:, i] = discretize_series(X[:, i], n, method)
        
    return X.astype(int)


# ---- 计算R2 ----------------------------------------------------------------------------------------

def cal_r2(y, y_pred) -> float:
    SStot = np.sum((y_pred - np.mean(y_pred)) ** 2)
    SSreg = np.sum((y - np.mean(y_pred)) ** 2)
    r2 = SSreg / SStot
    return r2


# ---- 滤波 ----------------------------------------------------------------------------------------

def savitzky_golay(y, window_size, order, deriv = 0, rate = 1) -> np.ndarray:
    """
    savitzky_golay滤波
    
    Reference:
    ----------
    链接地址：https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except Exception:
        raise ValueError("window_size and order have to be of type int")
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    
    # 预计算系数.
    b = np.mat([[k ** i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    
    # pad the signal at the extremes with values taken from the signal itself.
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve(m[::-1], y, mode = "valid")