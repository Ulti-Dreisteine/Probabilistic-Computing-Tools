# -*- coding: utf-8 -*-
"""
Created on 2024/01/22 14:04:28

@File -> bayesian_net_cond_indep_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于贝叶斯网络的条件独立性检验
"""

from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import arviz as az
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.independence_test.surrog_indep_test import exec_surrog_indep_test


def _normalize(x: np.ndarray):
    return MinMaxScaler().fit_transform(x.reshape(len(x), -1)).flatten()


def gen_data(func: str, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    产生数据
    """
    
    x1 = np.random.normal(0, 1, N)
    x2 = np.random.normal(0, 1, N)
    e1 = np.random.random(N) * 1e-6
    e2 = np.random.random(N) * 1e-6
    z = np.random.normal(0, 1, N)
    
    if func == "M1":
        x = x1 + z + e1
        y = x2 + z + e2
    elif func == "M2":
        x = x1 + z + e1
        y = np.power(z, 2) + e2
    elif func == "M3":
        x = x1 + z + e1
        y = 0.5 * np.sin(x1 * np.pi) + z + e2
    elif func == "M4":
        x = x1 + z + e1
        y = x1 + x2 + z + e2
    elif func == "M5":
        x = np.sqrt(np.abs(x1 * z)) + z + e1
        y = 0.25 * (x1 ** 2) * (x2 ** 2) + x2 + z + e2
    elif func == "M6":
        x = np.log(np.abs(x1 * z) + 1) + z + e1
        y = 0.5 * (x1 ** 2 * z) + x2 + z + e2
    else:
        raise ValueError(f"unknown func {func}")
    
    x, y, z = _normalize(x), _normalize(y), _normalize(z)
    
    return x, y, z


if __name__ == "__main__":
    funcs = ["M1", "M2", "M3", "M4", "M5", "M6"]
    N = 500
    
    method = "CMI-GIEF"
    rounds = 300
    alpha = 0.01

    _, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, func in enumerate(funcs):
        x, y, z = gen_data(func, N)
        assoc, (p, indep, assocs_srg) = exec_surrog_indep_test(
            x, y, method, z=z, xtype="c", ytype="c", ztype="c", rounds=rounds, alpha=alpha)

        # 画图
        ax = axs[i // 2, i % 2]
        az.plot_posterior(
            {f"{method}_Surrog": assocs_srg}, 
            kind="hist",
            bins=50,
            ref_val=assoc,
            hdi_prob=1 - alpha * 2,
            ax=ax)
        ax.set_title(f"dataset: {func}, independence detected: {indep}", fontsize=18)
        ax.set_xlabel("CMI value")
    plt.tight_layout()
    
    plt.show()