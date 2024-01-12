# -*- coding: utf-8 -*-
"""
Created on 2024/01/11 15:58:39

@File -> discrete_data.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 离散数据概率估计
"""

from typing import Union, List
import numpy as np


def _cal_prob(arr: np.ndarray, dims: List[int], state: Union[np.ndarray, list]) -> float:
    """
    计算概率
    
    Params:
    -------
    arr: 联合分布样本数组
    dims: 指定的变量维数列表, 对应于列索引编号
    state: 对应dims维数上的特定标签值状态
    
    Note:
    -----
    arr各元素值均为整数标签
    """
    
    N, _ = arr.shape
    state = np.array(state).flatten()
    prob = (arr[:, dims] == state).all(axis = 1).sum() / N
    
    return prob


def cal_discrete_prob(arr: np.ndarray, dims: List[int], state: Union[np.ndarray, list], 
                      cdims: List[int] = None, cstate: Union[np.ndarray, list] = None) -> float:
    """
    计算离散概率和条件概率
    
    Params:
    -------
    arr: 联合分布样本数组
    dims: 指定的变量维数列表, 对应于列索引编号
    state: 对应dims维数上的特定标签值状态
    cdims: 条件变量维数列表, 对应于列索引编号
    cstate: 条件变量对应cdims维数上的特定标签值状态
    
    Note:
    -----
    arr各元素值均为整数标签
    
    Example:
    --------
    arr.shape = (-1, 3)  # X0, X1, X2
    
    # 一维边际概率
    P(X1 = x1) = cal_discrete_prob(arr, [1], [x1])
    
    # 二维边际概率
    P(X1 = x1, X2 = x2) = cal_discrete_prob(arr, [1, 2], [x1, x2])
    
    # 条件概率
    P(X1 = x1, X2 = x2 | X0 = x0) = cal_discrete_prob(arr, [1, 2], [x1, x2], [0], [x0])
    """
    
    assert "int" in str(arr.dtype)
    
    if (cdims is None) or (cstate is None):
        return _cal_prob(arr, dims, state) 
    else:
        # 判断待计算变量集合与条件变量集合不同
        try:
            assert not bool(set(dims) & set(cdims))
        except Exception as _:
            raise ValueError(f"待计算变量索引集 {dims} 与条件索引集 {cdims} 有重叠") from _
        
        # 从联合分布样本arr中提取满足条件状态的样本, 保留所有列
        mask = arr[:, cdims] == np.array(cstate)  # type: np.ndarray
        # mask = np.apply_along_axis(all, axis=1, arr=mask)
        carr = arr.copy()[mask.all(axis=1), :]
        return _cal_prob(carr, dims, state)
    
    
if __name__ == "__main__":
    # 初始化联合分布样本数组
    arr = np.array([
        [1, 1, 2],
        [1, 1, 1],
        [1, 2, 1],
        [2, 1, 1],
        [2, 2, 1],
        [1, 1, 2],
        [1, 2, 2],
        [2, 1, 2],
        [2, 2, 2],
    ])
    
    print(f"arr.shape = {arr.shape}")
    
    # ---- 概率 -------------------------------------------------------------------------------------
    
    print("\n概率:")
    print(cal_discrete_prob(arr, [0, 1], [1, 1]))
    
    print("\n条件概率:")
    print(cal_discrete_prob(arr, [0, 1], [1, 1], [2], [1]))
    
    print("\n条件概率:")
    print(cal_discrete_prob(arr, [0], [1], [1, 2], [1, 1]))
    
    cdims = [1, 2]
    cstate = [1, 2]
    mask = arr[:, cdims] == np.array(cstate)
    
    
    
    
    
    
    
    
    
    