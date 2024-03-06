import numpy as np
import unittest
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from giefstat.coefficient.mi_kde.mi_kde import MutualInfoKDE, MargEntropy


class MutualInfoKDETest(unittest.TestCase):
    """
    测试MutualInfoKDE
    """
    
    def setup(self):
        print("testing task setup")
        
    def test_mutual_info_kde(self):
        x = np.array([1, 2, 3, 4]).astype(float)
        y = np.array([1, 2, 3, 4.1]).astype(float)
        
        mi_xy = MutualInfoKDE(x, y)()
        self.assertIsInstance(mi_xy, float)
        self.assertGreater(mi_xy, 0.0)
        
    def test_marg_entropy(self):
        x = np.array([1, 2, 3, 4]).astype(float)
        y = np.array([1, 2, 3, 4.1]).astype(float)
        
        hx, hy = MargEntropy(x)(), MargEntropy(y)()
        self.assertIsInstance(hx, float)
        self.assertGreater(hx, 0.0)
        self.assertIsInstance(hy, float)
        self.assertGreater(hy, 0.0)
        

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)