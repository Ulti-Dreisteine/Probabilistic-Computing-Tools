# -*- coding: utf-8 -*-
"""
Created on 2024/04/08 11:48:02

@File -> stft_petro.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 短时傅里叶变换
"""

from scipy.signal import stft
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    
    # ---- 载入数据 ---------------------------------------------------------------------------------
    
    # data = pd.read_csv(f"{BASE_DIR}/dataset/chemical_process/petro_data.csv")
    # y = data["Y"].values
    
    # 生成数据
    fs = 256
    t = np.linspace(0, 1, fs, endpoint=False) # 时间
    
    signal_1 = np.sin(2 * np.pi * 50 * t) # 50Hz正弦波
    signal_2 = 0.5 * np.sin(2 * np.pi * 100 * t) # 100Hz正弦波
    signal_3 = 0.5 * np.sin(2 * np.pi * 200 * t) # 200Hz正弦波
    
    y = np.concatenate([signal_1, signal_2, signal_3])
    
    # ---- STFT分析 ---------------------------------------------------------------------------------
    
    f, t, spectrum = stft(y, fs=1, nperseg=32)
    
    # 画图
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, f, np.abs(spectrum), shading="gouraud")