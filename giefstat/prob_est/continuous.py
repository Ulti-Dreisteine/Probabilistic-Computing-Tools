# -*- coding: utf-8 -*-
"""
Created on 2024/01/22 13:16:25

@File -> continuous.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 连续变量概率密度估计
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.stats import gaussian_kde
from typing import Optional, Union
import numpy as np

from ..util import stdize_values, build_tree, query_neighbors_dist, get_unit_ball_volume


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
    return (k / N) / (v * k_dist**D)


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


####################################################################################################
# 通用计算方法                                                                                       #
####################################################################################################


def cal_non_cond_prob(x: Union[np.ndarray, list], method: str = None, X: np.ndarray = None, 
                      model: Optional[Union[BallTree, KDTree, gaussian_kde]] = None, **kwargs):
    # sourcery skip: merge-else-if-into-elif, swap-if-else-branches, switch
    """
    计算非条件的概率密度。
    函数首先尝试断言模型model或者样本X必须存在一个：
    - 如果二者均不存在，则抛出ValueError
    - 如果模型存在，则直接使用该模型计算非条件的概率密度
    - 如果模型不存在，则使用总体样本计算非条件的概率密度。具体实现方法根据 method 参数的不同而有所不同：
    - 如果 method 为 "knn"，则使用 BallTree 或 KDTree 模型计算非条件的概率密度；
    - 如果 method 为 "kde"，则使用 gaussian_kde 模型计算非条件的概率密度；
    - 否则，抛出 ValueError。
    
    Params:
    -------
    x: 待计算位置
    method: 所选择的方法
    X: 总体样本集
    model: 所使用的模型
    **kwargs: 其他可选参数
    """

    try:
        # 断言模型或者样本必须存在一个
        assert (model is not None) | (X is not None)
    except Exception as _:
        raise ValueError("Either model or X must be specified.") from _

    # 如果对应方法模型存在, 则直接使用该模型计算; 否则使用总体样本计算
    if model is not None:
        if type(model) in [BallTree, KDTree]:
            return cal_knn_prob_dens(x, tree=model, **kwargs)
        elif type(model) in [gaussian_kde]:
            return cal_kde_prob_dens(x, kde=model)
        else:
            raise ValueError("model type must be in [BallTree, KDTree, gaussian_kde]")
    else:
        if method == "knn":
            return cal_knn_prob_dens(x, X=X, **kwargs)
        elif method == "kde":
            return cal_kde_prob_dens(x, X=X)
        else:
            raise ValueError("method must be in ['knn', 'kde']")
        

# TODO: 计算条件概率cal_cond_prob(...)
        

# if __name__ == "__main__":
#     # 生成一些样本数据
#     data = np.random.rand(100, 2)

#     # 创建一个BallTree for the sample data
#     tree = BallTree(data)

#     # 计算给定值的非条件概率密度
#     x = np.array([0.5, 0.5])
#     p_x = cal_non_cond_prob(x, model=tree)
#     print("Probability density of x:", p_x)

#     #  Alternatively, you can provide the input data and计算非条件概率密度
#     X = np.random.rand(100, 2)
#     p_X = cal_non_cond_prob(x, method="kde", X=X)
#     print(f"Probability density of x with respect to X: {p_X}")