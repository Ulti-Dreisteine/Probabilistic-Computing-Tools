from typing import Optional, Tuple, List
from sklearn.utils import resample, shuffle
import numpy as np

from ..prob_est import cal_discrete_prob
from .util import build_td_series


class TransferEntropy(object):
    """
    传递熵成对时延因果检验
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, tau_x: int, tau_y: int, alpha: float = 0.01, 
                 eps: float = 1e-12):
        """
        Params:
        -------
        x: X的序列样本
        y: Y的序列样本
        tau_x: X的特征时间参数
        tau_y: Y的特征时间参数
        alpha: 显著性水平
        # sub_sample_size: 子采样数
        # rounds: 随机测试轮数
        
        Note:
        -----
        x和y序列值都必须为整数
        """
        
        try:
            assert "int" in str(x.dtype)
            assert "int" in str(y.dtype)
        except Exception as _:
            raise RuntimeError("x和y序列值必须均为整数") from _
        
        self.x = x.flatten()
        self.y = y.flatten()
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.alpha = alpha
        self.eps = eps
        self.N = len(self.x)
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1  # X和Y的窗口样本数以及Y的超前时间
        
    @staticmethod
    def _sum(_arr_yyx, _states_yyx, eps) -> float:
        """通过离散加和计算TE值"""
        _te = 0.0
        
        for _state in _states_yyx:
            prob_yyx = cal_discrete_prob(_arr_yyx, [0, 1, 2], _state)
            prob_y_yx = cal_discrete_prob(_arr_yyx, [0], _state[[0]], [1, 2], _state[[1, 2]])
            prob_y_y = cal_discrete_prob(_arr_yyx, [0], _state[[0]], [1], _state[[1]])
            
            prob = prob_yyx * np.log2(prob_y_yx / (prob_y_y + eps) + eps)
            
            if not np.isnan(prob):
                _te += prob
        
        return _te
    
    def _build_iid_samples(self, x_td: np.ndarray, y_td: np.ndarray, iid_size: Optional[int] = None) \
        -> np.ndarray:
        """
        构建用于计算传递熵的独立同分布样本。

        Params:
        -------
        x_td: 变量 x 的时间序列数据。
        y_td: 变量 y 的时间序列数据。
        iid_size: 生成的独立同分布样本的大小。

        Return:
        -------
        np.ndarray: 独立同分布样本的数组。
        """
        
        # 构造联合分布样本
        idxs = np.arange(0, len(x_td) - self.h * self.tau_y - 1)
        xk = x_td[idxs].reshape(-1, 1)
        yk = y_td[idxs].reshape(-1, 1)
        yh = yk.copy()

        arr_yyx = np.concatenate([
            yh[self.h * self.tau_y:, :],
            yk[: -self.h * self.tau_y, :],
            xk[self.h * self.tau_y:, :]
            ], axis = 1)  # type: np.ndarray

        # 选择独立采样时间间隔，确保样本在x和y序列上都无关
        tau_iid = np.max([self.tau_x, self.tau_y])
        
        # 根据总样本量和独立采样时间间隔，自适应设置iid_size采集iid样本
        iid_size = len(arr_yyx) // tau_iid - 1 if iid_size is None else iid_size

        idxs = np.arange(0, tau_iid * (iid_size + 1), tau_iid)
        return arr_yyx[idxs, :]
    
    def cal_td_te(self, td_lag: int, bt_rounds: int = 200, iid_size: int = None) \
        -> Tuple[float, float, List[float]]:
        """
        计算X->Y@td_lag处的传递熵
        
        Params:
        -------
        td_lag: X到Y的时延
        bt_rounds: Bootstrap抽样轮数
        
        Returns:
        --------
        te_mean: 传递熵均值
        te_std: 传递熵标准差
        te_lst: 传递熵的Boostrap采样列表
        """
        
        x_td, y_td =  build_td_series(self.x, self.y, td_lag) # type: ignore
        
        # 构造iid样本
        arr_td_iid = self._build_iid_samples(x_td, y_td, iid_size=iid_size)
        
        # 进行多轮bootstrap检验
        te_lst = []
        idxs = np.arange(len(arr_td_iid))
        
        for _ in range(bt_rounds):
            # Bootstrap采样
            _idxs_bt = resample(idxs, replace=True)
            _arr = arr_td_iid.copy()[_idxs_bt, :]
            
            # 计算传递熵
            _states_yyx = np.unique(_arr, axis = 0)
            _te = self._sum(_arr, _states_yyx, self.eps)
            
            te_lst.append(_te)
            
        te_mean = np.nanmean(te_lst)
        te_std = np.nanstd(te_lst)
        
        return te_mean, te_std, te_lst
    
    def cal_bg_te(self, pm_rounds: int = 10, bt_rounds: int = 10, iid_size: int = None) -> \
        Tuple[float, float, List[float]]:
        """
        计算背景值
        
        Params:
        -------
        pm_rounds: Permutation轮数
        iid_size: iid样本大小
        
        Returns:
        --------
        te_mean: 背景传递熵均值
        te_std: 背景传递熵标准差
        te_lst: 背景传递熵的Permutation采样列表
        """
        
        te_lst = []
        
        for _ in range(pm_rounds):
            x_shuff = shuffle(self.x)
            
            # 构造iid样本
            arr_iid = self._build_iid_samples(x_shuff, self.y, iid_size=iid_size)
            
            for _ in range(bt_rounds):
                _arr = resample(arr_iid, replace=True)
            
                # 计算传递熵
                _states_yyx = np.unique(_arr, axis = 0)
                _te = self._sum(_arr, _states_yyx, self.eps)
            
                te_lst.append(_te)
        
        te_mean, te_std = np.nanmean(te_lst), np.nanstd(te_lst)
        
        return te_mean, te_std, te_lst
