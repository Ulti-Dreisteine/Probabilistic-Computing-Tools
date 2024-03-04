# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2023/04/26 16:45:20

@File -> surrog_indep_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于代用数据的独立性测试
"""

__doc__ = """
    计算关联系数并通过代用数据重采样获得结果的显著性信息
"""

from typing import Tuple
import numpy as np
import logging
import random

from ..coefficient import cal_general_assoc


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


def exec_surrog_indep_test(x: np.ndarray, y: np.ndarray, method: str, z: np.ndarray = None, 
                           xtype: str = None, ytype: str = None, ztype: str = None, rounds: int = 100, 
                           alpha: float = 0.05, max_size_bt: int = 1000, size_bt: int = None, 
                           **kwargs) -> Tuple[float, Tuple[float, bool, np.ndarray]]:
    """
    这段代码实现了一个基于代理数据独立性检验的函数exec_surrog_indep_test，通过随机抽样和假设检验来评估两个数据集（x和y）之间的独立性。
    计算步骤如下：
    1. 确定随机抽样样本量：如果size_bt为None，则使用max_size_bt作为随机抽样样本量；否则，使用size_bt作为随机抽样样本量；
    2. 计算关联系数：使用随机抽样的方法计算x和y之间的关联值，如果z不为None，则使用z数据进行计算；
    3. 计算背景系数值：使用蒙特卡洛方法，对x和y进行重复抽样，计算抽样之间的关联值，并将所有抽样得到的关联值存储在assocs_srg数组中；
    4. 计算显著性：基于单侧检验计算P值并判断两个数据集是否独立。
    
    Params:
    -------
    x: x数据
    y: y数据
    method: 选用的关联度量方法
    z: z数据
    xtype: x的数值类型
    :ytype: y的数值类型
    :ztype: y的数值类型
    rounds: 重复轮数
    alpha: 显著性阈值
    max_size_bt: 用于自助重采样的最大样本数
    size_bt: 自行设定的重采样规模
    kwargs: cal_general_assoc方法中的关键字参数, 见coefficient.__init__.cal_general_assoc
    
    Returns:
    --------
    assoc: 关联值, float
    p: P值, float
    indep: 是否独立, bool
    assocs_srg: 代理数据关联值数组, np.ndarray
    """
    
    # 确定随机抽样样本量
    size_bt = _deter_bootstrap_size(x, size_bt, max_size_bt)
        
    # 计算关联系数
    # TODO: 通过随机抽样获得关联值分布, 有放回or无放回?
    if z is None:
        assoc = cal_general_assoc(x[:size_bt], y[:size_bt], None, method, xtype, ytype, ztype, **kwargs)
    else:
        assoc = cal_general_assoc(x[:size_bt], y[:size_bt], z[:size_bt], method, xtype, ytype, ztype, **kwargs)

    # 计算背景值
    idxs = np.arange(len(x))
    assocs_srg = np.array([])
    
    for _ in range(rounds):
        # 基于代用数据获得背景值
        idxs_bt = random.choices(idxs, k=size_bt)   # 有放回抽样
        x_bt, y_bt = x[idxs_bt], y[idxs_bt]
        x_srg = np.random.permutation(x_bt)         # 随机重排获得代用数据，NOTE，只对x进行重排
        
        if z is None:
            assoc_srg = cal_general_assoc(x_srg, y_bt, None, method, xtype, ytype, ztype, **kwargs)
        else:
            z_bt = z[idxs_bt]
            assoc_srg = cal_general_assoc(x_srg, y_bt, z_bt, method, xtype, ytype, ztype, **kwargs)

        assocs_srg = np.append(assocs_srg, assoc_srg)

    # 计算显著性: 单侧检验
    p = len(assocs_srg[assocs_srg >= assoc]) / rounds
    indep = p >= alpha
    
    return assoc, (p, indep, assocs_srg)