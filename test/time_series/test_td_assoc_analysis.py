import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from dataset.time_delayed.data_generator import gen_four_species
from giefstat.time_series.td_assoc_analysis import measure_td_assoc

if __name__ == "__main__":
       
    # ---- 载入测试数据 -----------------------------------------------------------------------------

    samples = gen_four_species(N=6000)
    x, y = samples[:, 1], samples[:, 2]

    # ---- 时延关联检测 -----------------------------------------------------------------------------

    taus = np.arange(-20, 20, 1)
    _ = measure_td_assoc(x, y, taus, show=True, alpha=0.01, rounds=10)