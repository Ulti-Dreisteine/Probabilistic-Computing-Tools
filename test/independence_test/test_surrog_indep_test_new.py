# -*- coding: utf-8 -*-
"""
Created on 2024/03/05 13:16:24

@File -> test_surrog_indep_test_new.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试基于代用数据的独立性检验
"""

from scipy.stats import pearsonr
from typing import Union, List, Tuple
import arviz as az
import numpy as np
import logging
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.coefficient.mi_gief import MutualInfoGIEF
from giefstat.coefficient.mic import MIC
# from giefstat.independence_test.surrog_indep_test import exec_surrog_indep_test
from dataset.bivariate.data_generator import DataGenerator


def gen_test_data(func, N, scale):
    data_gener = DataGenerator()
    x, y, _, _ = data_gener.gen_data(N, func, normalize=False)
    y_range = np.max(y) - np.min(y)
    noise = np.random.uniform(-scale * y_range, scale * y_range, y.shape)
    y_noise = y.copy() + noise
    return x, y_noise


# 系数计算

def cal_assoc(x, y, xtype, ytype, method, **kwargs):
    if method == "PearsonCorr":
        return pearsonr(x, y)[0]
    elif method == "MI-GIEF":
        return MutualInfoGIEF(x, xtype, y, ytype)(**kwargs)
    elif method == "MIC":
        return MIC(x, y)(method="mic", **kwargs)
    else:
        raise ValueError(f"method {method} not supported")


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
        else:
            logging.warning(
                f"exec_surrog_indep_test: 采用默认max_size_bt={max_size_bt}作为size_bt, \
                用于限制代用数据重采样规模")
            return max_size_bt
    else:
        return size_bt


# def exec_surrog_indep_test(x: np.ndarray, y: np.ndarray, method: str, z: np.ndarray = None, 
#                            xtype: str = None, ytype: str = None, ztype: str = None, rounds: int = 100, 
#                            alpha: float = 0.05, max_size_bt: int = 1000, size_bt: int = None, 
#                            **kwargs) -> Tuple[float, Tuple[float, bool, np.ndarray]]:
#     """
#     这段代码实现了一个基于代理数据独立性检验的函数exec_surrog_indep_test，通过随机抽样和假设检验来评估两个数据集（x和y）之间的独立性。
#     计算步骤如下：
#     1. 确定随机抽样样本量：如果size_bt为None，则使用max_size_bt作为随机抽样样本量；否则，使用size_bt作为随机抽样样本量；
#     2. 计算关联系数：使用随机抽样的方法计算x和y之间的关联值，如果z不为None，则使用z数据进行计算；
#     3. 计算背景系数值：使用蒙特卡洛方法，对x和y进行重复抽样，计算抽样之间的关联值，并将所有抽样得到的关联值存储在assocs_srg数组中；
#     4. 计算显著性：基于单侧检验计算P值并判断两个数据集是否独立。
    
#     Params:
#     -------
#     x: x数据
#     y: y数据
#     method: 选用的关联度量方法
#     z: z数据
#     xtype: x的数值类型
#     :ytype: y的数值类型
#     :ztype: y的数值类型
#     rounds: 重复轮数
#     alpha: 显著性阈值
#     max_size_bt: 用于自助重采样的最大样本数
#     size_bt: 自行设定的重采样规模
#     kwargs: cal_general_assoc方法中的关键字参数, 见coefficient.__init__.cal_general_assoc
    
#     Returns:
#     --------
#     assoc: 关联值, float
#     p: P值, float
#     indep: 是否独立, bool
#     assocs_srg: 代理数据关联值数组, np.ndarray
#     """
    
#     # 确定随机抽样样本量
#     size_bt = _deter_bootstrap_size(x, size_bt, max_size_bt)
        
#     # 计算关联系数
#     # TODO: 通过随机抽样获得关联值分布, 有放回or无放回?
#     if z is None:
#         assoc = cal_general_assoc(x[:size_bt], y[:size_bt], None, method, xtype, ytype, ztype, **kwargs)
#     else:
#         assoc = cal_general_assoc(x[:size_bt], y[:size_bt], z[:size_bt], method, xtype, ytype, ztype, **kwargs)

#     # 计算背景值
#     idxs = np.arange(len(x))
#     assocs_srg = np.array([])
    
#     for _ in range(rounds):
#         # 基于代用数据获得背景值
#         idxs_bt = random.choices(idxs, k=size_bt)   # 有放回抽样
#         x_bt, y_bt = x[idxs_bt], y[idxs_bt]
#         x_srg = np.random.permutation(x_bt)         # 随机重排获得代用数据，NOTE，只对x进行重排
        
#         if z is None:
#             assoc_srg = cal_general_assoc(x_srg, y_bt, None, method, xtype, ytype, ztype, **kwargs)
#         else:
#             z_bt = z[idxs_bt]
#             assoc_srg = cal_general_assoc(x_srg, y_bt, z_bt, method, xtype, ytype, ztype, **kwargs)

#         assocs_srg = np.append(assocs_srg, assoc_srg)

#     # 计算显著性: 单侧检验
#     p = len(assocs_srg[assocs_srg >= assoc]) / rounds
#     indep = p >= alpha
    
#     return assoc, (p, indep, assocs_srg)


if __name__ == "__main__":
    
    # ---- 生成数据 ---------------------------------------------------------------------------------
    
    x, y = gen_test_data("sin_low_freq", 800, 0.1)
    
    plt.scatter(x, y, color="w", edgecolor="k")
    
    # ---- 计算系数值和显著性 -------------------------------------------------------------------------
    
    x, y = x.flatten(), y.flatten()
    
    # 参数设置
    rounds = 100  # BT抽样轮数
    alpha = 0.05  # 显著性阈值
    
    size_bt = None
    max_size_bt = 5000
    
    # method = "PearsonCorr"
    # method = "MI-GIEF"
    method = "MIC"
    
    """计算原理：
    1. 进行多次Bootstrap抽样或降采样, 在每一轮抽样计算中;
      1.1 计算抽样样本x_bt和y_bt的系数值;
      1.2 对x_bt进行随机重排, 计算x_bt重排样本x_srg和y_bt之间的系数值;
    2. 汇总所有系数值和背景值, 计算显著性;
    """
    idxs = list(np.arange(len(x)))
    
    # 确定随机抽样样本量
    size_bt = _deter_bootstrap_size(x, size_bt, max_size_bt) // 2
    
    # assoc = np.abs(cal_assoc(x, y, "c", "c", method))
    assoc = np.abs(cal_assoc(x[:size_bt], y[:size_bt], "c", "c", method))
    
    # 进行bootstrap抽样计算
    bt_records = {"bt": [], "srg": []}
    
    for _ in range(rounds):
        print(f"\rround {_}", end="")
        
        # TODO: Bootstrap抽样 or Subsampling???
        # NOTE: 基于KNN、KDE等方法对于重复样本很敏感
        
        # idxs_bt = random.choices(idxs, k=size_bt)  # 有放回抽样
        idxs_bt = random.sample(idxs, k=size_bt)  # 无放回抽样
        x_bt, y_bt = x[idxs_bt], y[idxs_bt]
        
        assoc_bt = cal_assoc(x_bt, y_bt, "c", "c", method)
        
        # 对x_bt进行随机重排, 计算x_bt重排样本x_srg和y_bt之间的系数值
        x_srg = np.random.permutation(x_bt)
        assoc_srg = cal_assoc(x_srg, y_bt, "c", "c", method)
        
        bt_records["bt"].append(np.abs(assoc_bt))
        bt_records["srg"].append(np.abs(assoc_srg))
    
    # 画图显示计算结果
    az.plot_posterior(
        {f"{method}": bt_records["bt"]}, 
        kind="hist", 
        bins=30, 
        ref_val=assoc, 
        hdi_prob=1 - alpha * 2)
    
    az.plot_posterior(
        {f"{method}_Surrog": bt_records["srg"]}, 
        kind="hist",
        bins=30,
        ref_val=assoc,
        hdi_prob=1 - alpha * 2)
    