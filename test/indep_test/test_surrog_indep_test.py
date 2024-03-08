# -*- coding: utf-8 -*-
"""
Created on 2024/03/05 13:16:24

@File -> test_surrog_indep_test_new.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试基于代用数据的独立性检验
"""

from scipy.stats import pearsonr
from typing import Union, List, Tuple, Optional
import arviz as az
import numpy as np
import logging
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from dataset.bivariate.data_generator import DataGenerator
from giefstat.indep_test.surrog_indep_test import exec_surrog_indep_test


def gen_test_data(func, N, scale):
    data_gener = DataGenerator()
    x, y, _, _ = data_gener.gen_data(N, func, normalize=False)
    y_range = np.max(y) - np.min(y)
    noise = np.random.uniform(-scale * y_range, scale * y_range, y.shape)
    y_noise = y.copy() + noise
    return x, y_noise


if __name__ == "__main__":
    # 生成数据
    x, y = gen_test_data("sin_low_freq", 800, 0.0)
    
    plt.scatter(x, y, color="w", edgecolor="k")
    
    # 计算系数值和显著性
    assoc, (p, indep, assocs_srg) = exec_surrog_indep_test(
        x, y, "MI-GIEF", xtype="c", ytype="c", rounds=100, show=True)
    
    