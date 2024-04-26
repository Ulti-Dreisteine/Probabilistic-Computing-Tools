from numpy import random
import itertools
import numpy as np


def f(s, a, b) -> float:
    return a * s**2 / (1 - b * s**2)


def gen_data():
    """含有回路的时延过程"""
    lags = [2, 2, 2, 2]
    As = [.5, .5, 1, 10]
    Bs = [0.8] * 5

    N = 10000
    D = 4
    arr = (np.random.rand(N, D) - 0.5) * 2

    # 每个时间步t
    for t, idx in itertools.product(range(10, len(arr)), range(1, D)):
        lag, a, b = lags[idx - 1], As[idx - 1], Bs[idx - 1]
        s = f(arr[t - lag, idx - 1], a, b)

        if idx == 1:
            # 来自idx{-1}->1的反馈
            lag, a, b = lags[-1], As[-1], Bs[-1]
            s += f(arr[t - lag, -1], a, b)

        arr[t, idx] = s + random.normal(0, 0.01)

    arr = arr[-2010:, :]
    return arr


# if __name__ == "__main__":
#     import sys
#     import os
    
#     BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
#     sys.path.insert(0, BASE_DIR)
    
#     from mod.dir_file_op import save_json_file
    
#     arr = gen_data()
#     cols = [f"X{i}" for i in range(1, arr.shape[1])]
    
#     # 保存结果
#     save_json_file({"cols": cols}, f"{BASE_DIR}/case_cyclic_benchmark/file/cols.json")
#     np.save(f"{BASE_DIR}/case_cyclic_benchmark/data/arr.npy", arr)