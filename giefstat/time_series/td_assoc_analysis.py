# -*- coding: utf-8 -*-
"""
Created on 2023/06/05 15:44:42

@File -> td_assoc_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时延关联检测
"""

from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import logging

from ..util import build_td_series
from ..indep_test.surrog_indep_test import exec_surrog_indep_test


def _deter_size_bt(N: int, taus: Union[np.ndarray, List[int]], max_size_bt: int) -> int:
    """
    确定随机测试样本量
    """
    
    N_drop_tau_max = N - np.max(np.abs(taus))  # 时延计算中的最低样本量
    
    if N_drop_tau_max <= max_size_bt:
        size_bt = N_drop_tau_max
    else:
        logging.warning(f"采用默认max_size_bt={max_size_bt}作为size_bt, 用于限制代用数据重采样规模")
        size_bt = max_size_bt

    return size_bt


def _show_results(taus, td_assocs, td_indeps, method):
    plt.figure()
    plt.scatter(
        taus, 
        td_assocs, 
        edgecolors=["k" if p==1 else "r" for p in td_indeps], # type: ignore
        c="w", 
        lw=2, 
        zorder=1)
    plt.plot(taus, td_assocs, c="k", lw=1.5, zorder=0)
    plt.grid(alpha=0.3, zorder=-1)
    plt.xlabel("$\\tau$")
    plt.ylabel(method)


def measure_td_assoc(x: np.ndarray, y: np.ndarray, taus: Union[np.ndarray, List[int]], 
                     xtype: str = "c", ytype: str = "c", max_size_bt: int = 1000, method: str = "MI-GIEF", 
                     show: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    双变量时延关联检测

    Params:
    -------
    x: x样本
    y: y样本
    taus: 待计算时延数组
    xtype: x的数值类型
    ytype: y的数值类型
    max_size_bt: 重采样计算中的最大样本数
    method: 所使用的关联度量方法
    show: 是否显示结果
    **kwargs: 如alpha, rounds等, 见independence_test.surrog_indep_test.exec_surrog_indep_test
    """
    
    x, y = x.flatten(), y.flatten()
    N = len(x)

    # 检查样本长度
    try:
        assert N == len(y)
    except Exception as e:
        raise RuntimeError("len(x) is not equal to len(y).") from e
    
    size_bt = _deter_size_bt(N, taus, max_size_bt)
    
    # 逐时延计算
    td_assocs = np.array([])
    td_indeps = np.array([])
    
    for tau in taus:
        # 构造时延序列
        x_td, y_td = build_td_series(x, y, tau)
        
        # 进行关联计算
        assoc, (_, indep, _) = exec_surrog_indep_test(
            x_td, y_td, method, xtype=xtype, ytype=ytype, size_bt=size_bt, **kwargs)
        
        td_assocs = np.append(td_assocs, assoc)
        td_indeps = np.append(td_indeps, indep)
        
    # 显示结果, 红色点表示显著
    if show:
        _show_results(taus, td_assocs, td_indeps, method)
    
    return td_assocs, td_indeps


def acf_test(x: np.ndarray, taus: Union[np.ndarray, List[int]], xtype: str = "c", 
             max_size_bt: int = 1000, method: str = "MI-GIEF", show: bool = False, **kwargs):
    """
    自相关函数检验
    
    Params:
    -------
    see measure_td_assoc
    """
    
    y, ytype = x.copy(), xtype
    
    return measure_td_assoc(x, y, taus, xtype, ytype, max_size_bt, method, show, **kwargs)
    
