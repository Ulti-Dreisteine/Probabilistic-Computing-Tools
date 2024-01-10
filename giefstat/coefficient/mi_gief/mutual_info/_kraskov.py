# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2021/08/07 16:38:23

@File -> mie_kraskov.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用Kraskov方法计算互信息
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma
import numpy.linalg as la
from typing import Union, Tuple
from numpy import log
import numpy as np
import warnings

__doc__ = """
    (补充文献)
"""


def entropy(x, k=3, base=2) -> float:
    """
    The classic K-L k-nearest neighbor continuous entropy estimator.
    x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    
    # Set k smaller than num.samples - 1
    assert k <= len(x) - 1
    
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    
    return (const + n_features * np.log(nn).mean()) / log(base)


def centropy(x, y, k=3, base=2) -> float:
    """
    The classic K-L k-nearest neighbor continuous entropy estimator for the entropy of X conditioned
    on Y
    """
    
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k, base=base)
    entropy_y = entropy(y, k=k, base=base)
    
    return entropy_union_xy - entropy_y


def tc(xs, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropy(col, k=k, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropy(xs, k, base)


def ctc(xs, y, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropy(col, y, k=k, base=base)
                         for col in xs_columns]
    return np.sum(centropy_features) - centropy(xs, y, k, base)


def kraskov_mi(x, y, z=None, k=3, base=np.e, alpha=0) -> float:
    """
    Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    
    # arrays should have same length
    assert len(x) == len(y)
    
    # set k smaller than num.samples - 1
    assert k <= len(x) - 1
    
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    
    points = np.hstack(points)
    
    # find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(
            y, dvec), digamma(k), digamma(len(x))
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(
            yz, dvec), avgdigamma(z, dvec), digamma(k)
        
    return (-a - b + c + d) / log(base)


def kldiv(x, xp, k=3, base=2) -> float:
    """
    KL Divergence between p and q for x ~ p(x), xp ~ q(x)
    x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    
    # set k smaller than num. samples - 1
    assert k < min(len(x), len(xp))
    
    # two distributions must have same dim
    assert len(x[0]) == len(xp[0])
    
    x, xp = np.asarray(x), np.asarray(xp)
    x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
    
    d = len(x[0])
    n = len(x)
    m = len(xp)
    const = log(m) - log(n - 1)
    
    tree = build_tree(x)
    treep = build_tree(xp)
    nn = query_neighbors(tree, x, k)
    nnp = query_neighbors(treep, x, k - 1)
    
    return (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / log(base)


def lnc_correction(tree, points, k, alpha):
    """
    局部非正态校正 (local Non-normality Correction)
    """
    
    e = 0
    n_sample = points.shape[0]
    
    for point in points:
        # find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
        knn_points = points[knn]
        
        # substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        
        # calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        
        # calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        
        # calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    
    return e


# 离散估计器

def entropyd(sx, base=2) -> float:
    """
    discrete entropy estimator
    
    Params:
    -------
    sx: a list of samples
    """
    
    _, count = np.unique(sx, return_counts=True, axis=0)
    
    # convert to float as otherwise integer division results in all 0 for proba
    proba = count.astype(float) / len(sx)
    
    # avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy
    # estimate as 0 * log(1/0) = 0
    proba = proba[proba > 0.0]
    
    return np.sum(proba * np.log(1. / proba)) / log(base)


def midd(x, y, base=2) -> float:
    """
    discrete mutual information estimator given a list of samples which can be any hashable object
    """
    
    # arrays should have same lengths
    assert len(x) == len(y)
    
    return entropyd(x, base) - centropyd(x, y, base)


def cmidd(x, y, z, base=2) -> float:
    """
    discrete mutual information estimator given a list of samples which can be any hashable object
    """
    
    # arrays should have same lengths
    assert len(x) == len(y) == len(z)
    
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    
    return entropyd(xz, base) + entropyd(yz, base) - entropyd(xyz, base) - entropyd(z, base)


def centropyd(x, y, base=2) -> float:
    """
    the classic K-L k-nearest neighbor continuous entropy estimator for the entropy of X 
    conditioned on Y.
    """
    
    xy = np.c_[x, y]
    
    return entropyd(xy, base) - entropyd(y, base)


def tcd(xs, base=2) -> float:
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropyd(col, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropyd(xs, base)


def ctcd(xs, y, base=2) -> float:
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropyd(col, y, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropyd(xs, y, base)


def corexd(xs, ys, base=2) -> float:
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [midd(col, ys, base=base) for col in xs_columns]
    return np.sum(cmi_features) - midd(xs, ys, base)


# 混合估计器

def micd(x, y, k=3, base=2, warning=True) -> float:
    """
    if x is continuous and y is discrete, compute mutual information
    """
    
    # arrays should have same length
    assert len(x) == len(y)
    
    entropy_x = entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    
    return abs(entropy_x - entropy_x_given_y)  # units already applied


def midc(x, y, k=3, base=2, warning=True) -> float:
    return micd(y, x, k, base, warning)


def centropycd(x, y, k=3, base=2, warning=True) -> float:
    """
    conditional entropy between continuous X and discrete Y
    """
    
    return entropy(x, base) - micd(x, y, k, base, warning)


def centropydc(x, y, k=3, base=2, warning=True) -> float:
    return centropycd(y, x, k=k, base=base, warning=warning)


def ctcdc(xs, y, k=3, base=2, warning=True):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropydc(
        col, y, k=k, base=base, warning=warning) for col in xs_columns]
    return np.sum(centropy_features) - centropydc(xs, y, k, base, warning)


def ctccd(xs, y, k=3, base=2, warning=True):
    return ctcdc(y, xs, k=k, base=base, warning=warning)


def corexcd(xs, ys, k=3, base=2, warning=True):
    return corexdc(ys, xs, k=k, base=base, warning=warning)


def corexdc(xs, ys, k=3, base=2, warning=True):
    return tcd(xs, base) - ctcdc(xs, ys, k, base, warning)


# 工具函数

def add_noise(x, intens=1e-10):
    """ 
    small noise to break degeneracy
    """
    
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    """
    finds number of neighbors in some radius in the marginal space
    returns expectation value of <psi(nx)>
    """
    
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    
    return np.mean(digamma(num_points))


def build_tree(points) -> Union[BallTree, KDTree]:
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")


# 测试

def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs) -> \
    Tuple[float, Tuple[float, float]]:
    """
    shuffle test: repeatedly shuffle the x-values and then estimate measure(x, y, [z]), returns the 
    mean and conf. interval ("ci=0.95" default) over "ns" runs. "measure" could me mi, cmi, e.g. 
    keyword arguments can be passed. MI and CMI should have a mean near zero.
    """
    
    # a copy for shuffle
    x_clone = np.copy(x)  
    outputs = []
    
    for _ in range(ns):
        np.random.shuffle(x_clone)
        
        if z:
            outputs.append(measure(x_clone, y, z, **kwargs))
        else:
            outputs.append(measure(x_clone, y, **kwargs))
            
    outputs.sort()
    
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])