import warnings

warnings.filterwarnings("ignore")

from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Union, Tuple, List
from sklearn.metrics import mean_squared_error as mse, explained_variance_score as evs,\
    r2_score as r2, f1_score as f1
from scipy.special import gamma
import pandas as pd
import numpy as np


####################################################################################################
# 数据处理 
####################################################################################################

# ---- 数据标准化 -----------------------------------------------------------------------------------

def normalize(X: np.ndarray):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.copy())
    return X


def _convert_1d_series2int(x: np.ndarray):
    """
    将一维数据按照标签进行编码为连续整数
    """
    
    x = x.flatten()
    x_unique = np.unique(x)

    if len(x_unique) > 100:
        raise RuntimeWarning(
            f"too many labels: {len(x_unique)} for the discrete data")

    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    
    return x


def _convert_arr2int(arr: np.ndarray):
    """
    将一维数据按照标签进行编码为连续整数
    """
    
    _, D = arr.shape
    
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    
    return arr.astype(int)


def stdize_values(x: np.ndarray, dtype: str, eps: float = 1e-10) -> np.ndarray:
    """
    数据预处理: 标签值整数化、连续值归一化, 将连续和离散变量样本处理为对应的标准格式用于后续分析
    """
    
    x = x.copy()
    x = x.reshape(x.shape[0], -1)  # (N, ) 转为 (N, 1), (N, D) 转为 (N, D)
    
    if dtype == "c":
        # 连续值加入噪音并归一化
        x += eps * np.random.random_sample(x.shape)
        return normalize(x)
    elif dtype == "d":
        # 将标签值转为连续的整数值
        x = _convert_arr2int(x)
        return x
    

# ---- 数据离散化 -----------------------------------------------------------------------------------

def discretize_series(x: np.ndarray, n: int = None, method: str = "qcut") -> np.ndarray:
    """
    对数据序列采用等频分箱
    
    Params:
    -------
    n: 每个箱子里的样本数（平均值，因为pandas的结果有小幅波动）
    """
    
    n = len(x) // 15 if n is None else n
    q = int(len(x) // n)
    
    if method == "qcut":
        return pd.qcut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等频分箱
    elif method == "cut":
        return pd.cut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等宽分箱
    else:
        raise ValueError(f"Invalid method {method}")


def discretize_arr(X: np.ndarray, n: int = None, method: str = "qcut") -> np.ndarray:
    """
    数组逐列离散化
    """
    
    if n is None:
        n = X.shape[0] // 20
        
    X = X.copy()
    
    for i in range(X.shape[1]):
        X[:, i] = discretize_series(X[:, i], n, method)
        
    return X.astype(int)
    

# ---- K近邻查询 ------------------------------------------------------------------------------------
    
def build_tree(x: np.ndarray, metric: str = "chebyshev") -> Union[BallTree, KDTree]:
    """
    建立近邻查询树. 低维用具有欧式距离特性的KDTree; 高维用具有更一般距离特性的BallTree
    """
    
    x = x.reshape(len(x), -1)
    
    return BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(x, metric=metric)


def query_neighbors_dist(tree: Union[BallTree, KDTree], x: Union[np.ndarray, list], k: int) -> np.ndarray:
    """
    求得x样本在tree上的第k个近邻样本
    
    Note:
    -----
    如果tree的样本中包含了x, 则返回结果中也会含有x
    """
    
    x = np.array(x).reshape(1, len(x))
    
    # 返回在x处，tree的样本中距离其最近的k个样本信息
    nbrs_info = tree.query(x, k=k)
    
    return nbrs_info[0][:, -1]


# ---- 空间球体积 -----------------------------------------------------------------------------------

def get_unit_ball_volume(d: int, metric: str = "euclidean") -> float:
    """
    d维空间中按照euclidean或chebyshev距离计算所得的单位球体积
    """
    
    if metric == "euclidean":
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == "chebyshev":
        return 1
    else:
        raise ValueError(f"unsupported metric {metric}")
    

# ---- 构造时延序列 ---------------------------------------------------------------------------------

def build_td_series(x: np.ndarray, y: np.ndarray, tau: int, max_len: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成时延序列样本
    
    Params:
    -------
    x: 样本x数组
    y: 样本y数组
    tau: 时间平移样本点数
        * 若 tau > 0, 则 x 对应右方 tau 个样本点后的 y; 
        * 若 tau < 0, 则 y 对应右方 tau 个样本点后的 x
    """
    
    x_td, y_td = x.flatten(), y.flatten()
    
    if len(x_td) != len(y_td):
        raise ValueError("length of x is not equal to y")
    
    N = len(x_td)
    
    lag_remain = np.abs(tau) % N
    
    if lag_remain != 0:
        if tau > 0:
            y_td = y_td[lag_remain:]
            x_td = x_td[:-lag_remain]
        else:
            x_td = x_td[lag_remain:]
            y_td = y_td[:-lag_remain]

    # 当样本量过高时执行降采样, 降低计算复杂度
    if len(x_td) > max_len:
        idxs = np.arange(len(x_td))
        np.random.shuffle(idxs)
        idxs = idxs[:max_len]
        x_td, y_td = x_td[idxs], y_td[idxs]

    return x_td, y_td


####################################################################################################
# 模型评估 
####################################################################################################

def cal_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> Optional[float]:
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    
    if metric == "r2":
        return r2(y_true, y_pred)
    elif metric == "evs":
        return evs(y_true, y_pred)
    elif metric == "mse":
        return mse(y_true, y_pred)
    elif metric == "mape":
        idxs = np.where(y_true != 0)
        y_true, y_pred = y_true[idxs], y_pred[idxs]
        return np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)
    elif metric == "f1":
        return f1(y_true, y_pred, average="micro")  # 使用微平均
    else:
        raise ValueError(f"Unsupported metric {metric}")


# 模型评价
def exec_model_test(X: np.ndarray, y: np.ndarray, model, metric: str="r2", test_ratio: float=0.3, 
                    rounds: int=10) -> Tuple[float, List[float]]:
    """
    执行随机交叉验证模型测试
    """
    
    X, y = X.copy(), y.copy()
    N = X.shape[0]
    test_size = int(N * test_ratio)
    metrics = []
    
    for _ in range(rounds):
        shuffled_indexes = np.random.permutation(range(N))
        train_idxs = shuffled_indexes[test_size:]
        test_idxs = shuffled_indexes[:test_size]

        X_train, X_test = X[train_idxs, :], X[test_idxs, :]
        y_train, y_test = y[train_idxs], y[test_idxs]

        model.fit(X_train, y_train)
        m = cal_metric(y_test, model.predict(X_test), metric)
        metrics.append(m)
    
    return np.mean(metrics), metrics