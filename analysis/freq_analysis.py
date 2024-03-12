# -*- coding: utf-8 -*-
"""
Created on 2024/03/12 14:39:22

@File -> freq_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时间序列频域分析
"""

from scipy.fftpack import fft
import numpy as np
import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    data = pd.read_csv(f"{BASE_DIR}/dataset/chemical_process/petro_data.csv")
    
    y = data["X12"].values
    y_bg = np.random.normal(0, 1, len(y))
    
    # plt.plot(y)
    
    # ---- 频谱分析 ---------------------------------------------------------------------------------
    
    fft_y = fft(y)
    fft_y_bg = fft(y_bg)
    
    N = len(y)
    x = np.arange(N)
    abs_y, angle_y = np.abs(fft_y) / N, np.angle(fft_y)
    abs_y_bg, angle_y_bg = np.abs(fft_y_bg) / N, np.angle(fft_y_bg)
    
    # plt.plot(x[: N // 2], abs_y[: N // 2])
    plt.plot(x[: 300], abs_y[: 300])
    plt.plot(x[: 300], abs_y_bg[: 300])
    # plt.plot(x[:100], angle_y[:100])
    
    plt.ylim([0, 1])
    
    # plt.psd(y, NFFT=256, Fs=5, scale_by_freq=True)