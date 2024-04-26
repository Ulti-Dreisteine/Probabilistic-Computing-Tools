# -*- coding: utf-8 -*-
"""
Created on 2021/02/27 17:03:09

@File -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

__all__ = ["search_nearest_neighbors_in_list"]

import bisect


def search_nearest_neighbors_in_list(lst, x):
    """
    寻找x在有序lst中的两侧(或单侧)邻点值.
    
    params:
    -------
    x: 用于查询的值
    lst: 有序列表
    
    return:
    -------
    neighbors, tuple(left_neighbor, right_neighbor)
    """
    
    if x in lst:
        return [x]
    else:
        if x <= lst[0]:
            neighbors = [lst[0]]
        elif x >= lst[-1]:
            neighbors = [lst[-1]]
        else:
            left_idx = bisect.bisect_left(lst, x) - 1
            right_idx = left_idx + 1
            neighbors = [lst[left_idx], lst[right_idx]]
        return neighbors