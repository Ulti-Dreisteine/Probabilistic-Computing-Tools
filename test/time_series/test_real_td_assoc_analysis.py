import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.time_series.td_assoc_analysis import measure_td_assoc

if __name__ == "__main__":
    data = pd.read_csv(f"{BASE_DIR}/dataset/chemical_process/data.csv")
    
    x_cols = [f"X{i}" for i in range(1, 10)]
    y_col = ["Y"]
    
    X = data[x_cols].values
    y = data[y_col].values.flatten()
    
    taus = {
        "Y": 1, "X1": 4, "X2": 1, "X3": 1, "X4": 150, "X5": 150, "X6": 23, "X7": 5, "X8": 5, "X9": 4}
    
    for i in range(1, 10):
        print(f"i = {i}")
        x = X[:, i - 1]
        tau_x = taus[f"X{i}"]
        td_lags = np.arange(-20 * tau_x, 20 * tau_x + tau_x, tau_x)
        _ = measure_td_assoc(x, y, td_lags, show=True, alpha=0.01, rounds=10)
        plt.show()