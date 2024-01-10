# -*- coding: utf-8 -*-
"""
Created on 2022/01/05 13:53:45

@File -> setting.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 项目配置
"""

import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.append(BASE_DIR)

PROJ_CMAP = {
    "blue": "#1f77b4",  # 蓝色
    "orange": "#ff7f0e",  # 黄橙色
    "green": "#2ca02c",  # 绿色
    "red": "#d62728",  # 红色
    "purple": "#9467bd",  # 紫色
    "cyan": "#17becf",  # 青色
    "grey": "#7f7f7f",  # 灰色
    "black": "k",  # 黑色
    "white": "w",

    # 类似色搭配互补色, 同一色系list中颜色由亮到暗排列.
    "similar-complement-cmap": {
            "greens": ["#5ED1BA", "#34D1B2", "#00A383", "#1F7A68", "#006A55"],
            "reds": ["#F97083", "#F93E58", "#F30021", "#B62E40s", "#9E0016"],
            "yellows": ["#FFCB73", "#FFB840", "#FFA100", "#BF8A30", "#A66900"],
            "oranges": ["#FFAA73", "#FF8B40", "#FF6400", "#BF6830", "#A64100"],
    }
}

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 12

plt.rc("font", size=BIGGER_SIZE, family="Times New Roman")
plt.rc("axes", titlesize=BIGGER_SIZE)
plt.rc("axes", labelsize=BIGGER_SIZE  + 2)
plt.rc("xtick", labelsize=BIGGER_SIZE)
plt.rc("ytick", labelsize=BIGGER_SIZE)
plt.rc("legend", fontsize=BIGGER_SIZE)
plt.rc("figure", titlesize=20)
plt.rc("mathtext", fontset="cm")

# ---- 定义环境变量 ---------------------------------------------------------------------------------

# ---- 定义模型参数 ---------------------------------------------------------------------------------

# ---- 定义测试参数 ---------------------------------------------------------------------------------

# ---- 定义通用函数 ---------------------------------------------------------------------------------