from typing import Tuple
import arviz as az
import numpy as np
import logging
import random

from ..coefficient import cal_general_assoc, cal_assoc, cal_cond_assoc

# Bootstrapping和Downsampling方法

BT_METHODS = [
    "PearsonCorr", "SpearmanCorr", "MI-model", "MI-cut", "MI-qcut", "MI-Darbellay", "CMI-model", 
    "CMI-cut", "CMI-qcut", "DRV"]
DS_METHODS = [
    "DistCorr", "MI-GIEF", "MI-KDE", "MIC", "RMIC", "CMI-GIEF", "CMIC", "CRMIC"]


# 系数计算
# NOTE：注意iid条件


def _deter_bootstrap_size(x: np.ndarray, size_bt: int, max_size_bt: int) -> int:
    """
    确定随机抽取样本量
    
    Note:
    -----
    如果样本量小，则全部代入关联计算; 否则使用max_size对数据进行截断, 减少计算负担
    """
    
    if size_bt is None:
        if len(x) <= max_size_bt:
            return len(x)
        
        logging.warning(
            f"exec_surrog_indep_test: 采用默认max_size_bt={max_size_bt}作为size_bt, 用于限制代用数据重采样\
                规模")
        return max_size_bt
    
    return size_bt


def exec_surrog_indep_test(x: np.ndarray, y: np.ndarray, method: str, z: np.ndarray = None, 
                           xtype: str = None, ytype: str = None, ztype: str = None, rounds: int = 100, 
                           alpha: float = 0.05, max_size_bt: int = 1000, size_bt: int = None, 
                           show: bool = False, **kwargs) -> Tuple[float, Tuple[float, bool, np.ndarray]]:
    """
    基于替代数据的独立性检验
    
    Params:
    -------
    x: np.ndarray, 一维
    y: np.ndarray, 一维
    method: str, 关联计算方法
    z: np.ndarray, 一维, 条件变量
    xtype: str, x的数值类型, "d"或者"c"
    ytype: str, y的数值类型, "d"或者"c"
    ztype: str, z的数值类型, "d"或者"c"
    rounds: int, BT抽样轮数
    """
    
    x, y = x.flatten(), y.flatten()
    assert len(x) == len(y)  # x and y must have the same length
    
    if z is not None:
        z = z.reshape(len(z), -1)  # z可能是多维变量
        assert len(x) == len(z)  # x and z must have the same length
    
    N = len(x)
    idxs = list(np.arange(N))
    
    # 确定随机抽样样本量, 对于对重复样本敏感的方法, 样本量减半
    size_bt = _deter_bootstrap_size(x, size_bt, max_size_bt)
    
    if method in DS_METHODS:
        size_bt //= 2
        
        
    # 进行bootstrap抽样计算
    bt_records = {"bt": np.array([]), "srg": np.array([])}
    
    for _ in range(rounds):
        print(f"\rround {_}", end="")
        
        # 根据方法对重复样本敏感与否, 选择有放回或无放回抽样
        if method in BT_METHODS:
            idxs_bt = random.choices(idxs, k=size_bt)  # 有放回抽样，就是Bootstrap
        elif method in DS_METHODS:
            idxs_bt = random.sample(idxs, k=size_bt)  # 无放回抽样
        else:
            raise ValueError(f"method {method} not supported")
        
        x_bt, y_bt = x[idxs_bt], y[idxs_bt]
        z_bt = z[idxs_bt] if z is not None else None
        
        assoc_bt = cal_general_assoc(x_bt, y_bt, z_bt, method, xtype, ytype, ztype, **kwargs)
        
        # 对x_bt进行随机重排, 计算x_bt重排样本x_srg和y_bt之间的系数值
        x_srg = np.random.permutation(x_bt)
        assoc_srg = cal_general_assoc(x_srg, y_bt, z_bt, method, xtype, ytype, ztype, **kwargs)
        
        bt_records["bt"] = np.append(bt_records["bt"], np.abs(assoc_bt))
        bt_records["srg"] = np.append(bt_records["srg"], np.abs(assoc_srg))
        
    if show:
        # 计算参考值
        if z is None:
            assoc_ref = cal_assoc(
                x[:size_bt], y[:size_bt], method, xtype, ytype, **kwargs)
        else:
            assoc_ref = cal_cond_assoc(
                x[:size_bt], y[:size_bt], z[:size_bt], method, xtype, ytype, ztype, **kwargs)
            
        assoc_ref = np.abs(assoc_ref)
        
        import matplotlib.pyplot as plt
        
        plt.figure()
        
        # 画图显示计算结果
        plt.subplot(1, 2, 1)
        az.plot_posterior(
            {f"{method}": bt_records["bt"]}, 
            kind="hist", 
            bins=30, 
            ref_val=assoc_ref, 
            hdi_prob=1 - alpha * 2,
            show=False, ax=plt.gca())
        
        plt.subplot(1, 2, 2)
        az.plot_posterior(
            {f"{method}_Surrog": bt_records["srg"]}, 
            kind="hist",
            bins=30,
            ref_val=assoc_ref,
            hdi_prob=1 - alpha * 2,
            show=False, ax=plt.gca())

        plt.tight_layout()
        plt.show()
        
    # 计算结果
    assocs_bt = bt_records["bt"]
    assocs_srg = bt_records["srg"]
    assoc = np.mean(assocs_bt)
    
    # 计算显著性: 单侧检验
    p = len(assocs_srg[assocs_srg >= assoc]) / rounds
    indep = p >= alpha
    
    return assoc, (p, indep, assocs_srg)