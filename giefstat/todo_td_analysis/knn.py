from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class PairwiseKNN(object):
    """MIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
    
    def cal_knn(self, n_neighbors: int = 10):
        
        def _approx_y(y_arr, x_idxs):
            return np.mean(y_arr[x_idxs[1:], :])
    
        # 寻找K近邻点.
        knn = KNeighborsRegressor(n_neighbors = n_neighbors)
        knn.fit(self.x.reshape(-1, 1), self.y)
        knn_values, knn_idxs = knn.kneighbors(self.x.reshape(-1, 1), n_neighbors = n_neighbors)
    
        # 计算目标近似值.
        y_hat = np.apply_along_axis(lambda x_idxs: _approx_y(self.y.reshape(-1, 1), x_idxs), axis = 1, arr = knn_idxs)
    
        res = np.var(self.y - y_hat)
        res = 1 / (np.power(res * n_neighbors / (n_neighbors + 1), 0.5) + 1)
        return res