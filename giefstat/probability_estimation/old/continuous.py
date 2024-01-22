# -*- coding: utf-8 -*-
"""
Created on 2024/01/15 14:56:10

@File -> continuous.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 连续变量概率估计
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.stats import gaussian_kde
from typing import Optional, Union
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from giefstat.util import stdize_values, build_tree, query_neighbors_dist, get_unit_ball_volume


# K-Nerest Neighbors (KNN)
def cal_knn_prob_dens(x: Union[np.ndarray, list], X: Optional[np.ndarray] = None, 
                      tree: Union[BallTree, KDTree, None] = None, k: int = 3, 
                      metric: str = "chebyshev") -> float:
    """
    这段代码实现了KNN方法用于连续型变量的概率密度计算, 包括了以下几个步骤:
    - 构建距离树：使用Scikit-learn的BallTree或KDTree类构建距离树，如果总体样本集不为空，则使用KDTree进行构建。
    - 查询距离：使用距离树查询近邻，获取k nearest neighbors距离。
    - 计算概率密度：使用单位球体体积和k nearest neighbors距离的D次幂进行积分，得到概率密度
    
    Params:
    -------
    x: 待计算位置
    X: 总体样本集
    tree: 使用总体样本建立的距离树
    k: 近邻数
    metric: 距离度量指标
    
    Note:
    -----
    - x和X可以为一维或多维
    - X和tree中必须有一个赋值, 如果有tree则优先使用tree
    """
    
    x = np.array(x).flatten()               # 转为一维序列
    x = stdize_values(x, "c").flatten()     # 标准化
    
    # 构建距离树
    if tree is not None:
        N, D = tree.get_arrays()[0].shape
    elif X is not None:
        X = stdize_values(X.reshape(len(X), -1), "c")
        N, D = X.shape
        tree = build_tree(X)
    else:
        raise ValueError("Either X or tree must be specified.")
    
    k_dist = query_neighbors_dist(tree, x, k)[0]  # type: float
    v = get_unit_ball_volume(D, metric)
    p = (k / N) / (v * k_dist**D)
    
    return p


# Kernel Density Estimation (KDE)
def cal_kde_prob_dens(x: Union[np.ndarray, list], X: Optional[np.ndarray] = None, 
                      kde: Optional[gaussian_kde] = None) -> float:
    """
    这段代码实现了基于高斯核密度估计的方法用于连续型变量的概率密度计算, 包括了以下几个步骤:
    - 标准化：使用stdize_values函数将输入变量x进行标准化，使其分布更加接近标准正态分布。
    - 构建高斯核密度估计模型：如果总体样本集X不为空，则使用gaussian_kde类进行建模，否则抛出一个ValueError异常。
    - 计算概率密度：使用kde对象对输入变量x进行密度估计，并返回结果
    
    Params:
    -------
    x: 待计算位置
    X: 总体样本集
    kde: 高斯核密度估计对象实例
    
    Note:
    -----
    - x和X可以为一维或多维
    - X和kde中必须有一个赋值, 如果有kde则优先使用kde
    """
    
    x = np.array(x).flatten()               # 转为一维序列
    x = stdize_values(x, "c")
    
    if kde is not None:
        pass
    elif X is not None:
        X = stdize_values(X.reshape(len(X), -1), "c")
        kde = gaussian_kde(X.T)
    else:
        raise ValueError("Either X or kde must be specified.")
        
    return kde(x.T)[0]


def _cal_prob_dens(x, method, X, tree_x, kde_x, **kwargs) -> float:
    if method == "knn":
        return cal_knn_prob_dens(x, X=X, tree=tree_x, **kwargs)
    elif method == "kde":
        return cal_kde_prob_dens(x, X=X, kde=kde_x)
    else:
        raise ValueError(f"Unsupported method {method}.")
    

def _if_models_exist(tree_xz, kde_xz, tree_z, kde_z) -> bool:
    """
    如果对应模型均存在, 则返回True
    """
    
    if all(
        [(tree_xz is not None) | (kde_xz is not None), 
         (tree_z is not None) | (kde_z is not None)]):
        return True
    else:
        return False
    

def cal_prob_dens(x: Union[np.ndarray, list], method: str, X: Optional[np.ndarray] = None,
                  z: Union[np.ndarray, list] = None, Z: Optional[np.ndarray] = None, 
                  tree_x: Union[BallTree, KDTree, None] = None, kde_x: Optional[gaussian_kde] = None,
                  tree_xz: Union[BallTree, KDTree, None] = None, kde_xz: Optional[gaussian_kde] = None,
                  tree_z: Union[BallTree, KDTree, None] = None, kde_z: Optional[gaussian_kde] = None,
                  **kwargs) -> float:
    """
    计算连续变量的概率密度或条件概率密度
    
    Params:
    -------
    x: 待计算位置
    method: 计算方法
    X: X的总体样本
    z: 待计算条件位置
    XZ: XZ的总体样本
    tree_x: X总体样本构建的距离树
    kde_x: X的高斯核密度估计对象实例
    tree_xz: XZ总体样本构建的距离树
    kde_xz: XZ的高斯核密度估计对象实例
    kwargs:
        k: 近邻数, 见cal_knn_prob_dens函数
        metric: 距离度量指标, 见cal_knn_prob_dens函数
    
    Notes:
    ------
    - x和X可以为一维或多维
    - z和Z可以为一维或多维
    - X和kde中必须有一个赋值, 如果有kde则优先使用kde
    """
    
    assert method in {"knn", "kde"}

    if z is None:
        return _cal_prob_dens(x, method, X, tree_x, kde_x, **kwargs)
    else:
        xz = np.append(np.array(x).flatten(), np.array(z).flatten())
        
        if _if_models_exist(tree_xz, kde_xz, tree_z, kde_z):
            prob_xz = _cal_prob_dens(xz, method, None, tree_xz, kde_xz, **kwargs)
            prob_z = _cal_prob_dens(z, method, None, tree_z, kde_z, **kwargs)
        else:
            assert X is not None
            assert Z is not None
            
            X = X.copy().reshape(len(X), -1)
            Z = Z.copy().reshape(len(Z), -1)
            XZ = np.c_[X, Z]
            
            prob_xz = _cal_prob_dens(xz, method, XZ, None, None, **kwargs)
            prob_z = _cal_prob_dens(z, method, Z, None, None, **kwargs)
        
        return prob_xz / prob_z
    
    

# if __name__ == "__main__":
    
#     # ---- 载入数据 ---------------------------------------------------------------------------------

#     from dataset.trivariate.data_generator import DataGenerator
    
#     N = 100
#     func = "M6"
#     x, y, z = DataGenerator().gen_data(N, func)
    
#     # ---- 功能代码 ----------------------------------------------------------------------------------
    
#     X = np.c_[x, y]
#     Z = np.c_[z]
#     XZ = np.c_[X, Z]
    
#     # 输入参数
#     xz: Union[np.ndarray, list] = [0.5, 0.5, 0.3]
#     z: Union[np.ndarray, list] = [0.3]
    
#     # ---- 计算代码 ---------------------------------------------------------------------------------
    
#     XY = np.c_[x, y]
#     xy = [0.34, 0.57]
#     tree = build_tree(XY)
#     print(cal_knn_prob_dens(xy, XY, k=5))
#     print(cal_knn_prob_dens(xy, tree=tree, k=5))
    
#     kde = gaussian_kde(XY.T)
#     print(cal_kde_prob_dens(xy, XY))
#     print(cal_kde_prob_dens(xy, kde=kde))
    
    