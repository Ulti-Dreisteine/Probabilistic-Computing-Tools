import numpy as np
import unittest
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from giefstat.coefficient.mic import MIC, CMIC


class MaxInfoCoeffTest(unittest.TestCase):
    """
    测试MIC
    """
    
    def setup(self):
        print("testing task setup")
    
    def test_mic(self):
        # X, Y强相关
        x = np.random.normal(0, 1, 1000)
        y = x + np.random.normal(0, 0.01, 1000)
        
        mi_xy = MIC(x, y)()
        print(f"strongly correlated: MI between continous X and continous Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertGreater(mi_xy, 0.0)
        
        # X, Y不相关
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        
        mi_xy = MIC(x, y)()
        print(f"irrelevant: MI between continous X and continous Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertAlmostEqual(mi_xy, 0.0, delta=1e0)
        
    def test_cmic(self):
        # X, Y, Z均相关
        x = np.random.normal(0, 10, 1000)
        y = x + np.random.normal(0, 0.01, 1000)
        z = y + np.random.normal(0, 0.01, 1000)
        
        cmi_xyz = CMIC(x, y, z)()
        print(f"weakly correlated: CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e-1)
        
        # X和Y相关; X, Y与Z无关
        x = np.random.normal(0, 1, 100)
        y = x + np.random.normal(0, 0.1, 100)
        z = np.random.normal(0, 1, 100)
        
        cmi_xyz = CMIC(x, y, z)()
        print(f"strongly correlated: CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertGreater(cmi_xyz, 0.0)
        
        # X和Z相关; X, Z与Y无关
        x = np.random.normal(0, 1, 10000)
        y = np.random.normal(0, 1, 10000)
        z = x + np.random.normal(0, 0.1, 10000)
        
        cmi_xyz = CMIC(x, y, z)()
        print(f"irrelevant: CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e1)


if __name__ == "__main__":
    unittest.main()