import numpy as np
import unittest
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from giefstat.coefficient.mi_gief import MutualInfoGIEF, CondMutualInfoGIEF


class MutualInfoGIEFTest(unittest.TestCase):
    """
    测试MutualInfoGIEF
    """
    
    def setup(self):
        print("testing task setup")
    
    def test_mutual_info_gief(self):
        # X, Y强相关
        x = np.random.normal(0, 1, 1000)
        y = x + np.random.normal(0, 0.01, 1000)
        
        mi_xy = MutualInfoGIEF(x, "c", y, "c")()
        print(f"strongly correlated: MI between continous X and continous Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertGreater(mi_xy, 0.0)
        
        # X, Y不相关
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        
        mi_xy = MutualInfoGIEF(x, "c", y, "c")()
        print(f"irrelevant: MI between continous X and continous Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertAlmostEqual(mi_xy, 0.0, delta=1e0)
        
        # X, Y不相关
        x = np.random.normal(0, 1, 1000)
        y = np.random.randint(0, 2, 1000)
        
        mi_xy = MutualInfoGIEF(x, "c", y, "d")()
        print(f"irrelevant: MI between continous X and discrete Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertAlmostEqual(mi_xy, 0.0, delta=1e0)
    
    def test_cond_mutual_info_gief(self):
        # X, Y, Z均相关
        x = np.random.normal(0, 10, 1000)
        y = x + np.random.normal(0, 0.01, 1000)
        z = y + np.random.normal(0, 0.01, 1000)
        
        cmi_xyz = CondMutualInfoGIEF(x, "c", y, "c", z, "c")()
        print(f"weakly correlated: CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e-1)
        
        # X和Y相关; X, Y与Z无关
        x = np.random.normal(0, 1, 100)
        y = x + np.random.normal(0, 0.1, 100)
        z = np.random.normal(0, 1, 100)
        
        cmi_xyz = CondMutualInfoGIEF(x, "c", y, "c", z, "c")()
        print(f"strongly correlated: CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertGreater(cmi_xyz, 0.0)
        
        # X和Y相关; X, Y与Z无关
        x = np.random.normal(0, 1, 100)
        y = x + np.random.normal(0, 0.1, 100)
        z = np.random.randint(0, 2, 100)
        
        cmi_xyz = CondMutualInfoGIEF(x, "c", y, "c", z, "d")()
        print(f"strongly correlated: CMI between continous X and continous Y conditioned on discrete Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertGreater(cmi_xyz, 0.0)
        
        # X和Z相关; X, Z与Y无关
        x = np.random.normal(0, 1, 10000)
        y = np.random.normal(0, 1, 10000)
        z = x + np.random.normal(0, 0.1, 10000)
        
        cmi_xyz = CondMutualInfoGIEF(x, "c", y, "c", z, "c")()
        print(f"irrelevant: CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e-1)
        
        # X和Z相关; X, Z与Y无关
        x = np.random.normal(0, 1, 10000)
        y = np.random.randint(0, 2, 10000)
        z = x + np.random.normal(0, 0.1, 10000)
        
        cmi_xyz = CondMutualInfoGIEF(x, "c", y, "d", z, "c")()
        print(f"irrelevant: CMI between continous X and discrete Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e-1)


if __name__ == "__main__":
    unittest.main()