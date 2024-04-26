from sklearn.metrics import mean_absolute_percentage_error as cal_mape
from sklearn.metrics import mean_absolute_error as cal_mae
from sklearn.metrics import mean_squared_error as cal_mse
from sklearn.metrics import r2_score as cal_r2
from typing import List
import numpy as np



def exec_model_test(X: np.ndarray, y: np.ndarray, model, metric: str = "r2", 
                    test_ratio: float = 0.3, rounds: int = 50) -> List[float]:
    X, y = X.copy(), y.copy()
    N = X.shape[0]
    test_size = int(N * test_ratio)
    
    metrics = []
    
    for _ in range(rounds):
        shuffled_indexes = np.random.permutation(range(N))
        train_idxs = shuffled_indexes[test_size:]
        test_idxs = shuffled_indexes[:test_size]

        X_train, X_test = X[train_idxs, :], X[test_idxs, :]
        y_train, y_test = y[train_idxs], y[test_idxs]

        model.fit(X_train, y_train)
        
        if metric == "r2":
            m = cal_r2(y_test, model.predict(X_test))
        elif metric == "mse":
            m = cal_mse(y_test, model.predict(X_test))
        elif metric == "mae":
            m = cal_mae(y_test, model.predict(X_test))
        elif metric == "mape":
            m = cal_mape(y_test, model.predict(X_test))
        else:
            raise ValueError(f"Invalid metric: {metric}")
        
        metrics.append(m)
        
    return metrics
