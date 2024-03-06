from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import unittest
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from giefstat.coefficient.mi_model import MutualInfoModel, CondMutualInfoModel


class MutualInfoModelTest(unittest.TestCase):
    """
    测试MutualInfoModel
    """
    
    def setup(self):
        print("testing task setup")
        
    def test_mutual_info_model(self):
        # Y为连续变量
        x = np.arange(100).astype(float)
        y = np.arange(100).astype(float)
        model = RandomForestRegressor()
        
        mi_xy = MutualInfoModel(x, "c", y, "c")(model)
        print(f"MI between continous X and continous Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertGreater(mi_xy, 0.0)
        
        # Y为类别变量
        x = np.arange(100).astype(float)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier()
        
        mi_xy = MutualInfoModel(x, "c", y, "d")(model)
        print(f"MI between continous X and discrete Y: {mi_xy}")
        self.assertIsInstance(mi_xy, float)
        self.assertGreater(mi_xy, 0.0)
        
    def test_cond_mutual_info_model(self):
        # X, Y, Z均相关
        x = np.arange(100).astype(float)
        y = np.arange(100).astype(float)
        z = np.arange(100).astype(float)
        model = RandomForestRegressor()
        
        cmi_xyz = CondMutualInfoModel(x, "c", y, "c", z, "c")(model)
        print(f"CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e-2)
        
        # X和Y相关; X, Y与Z无关
        x = np.arange(100).astype(float)
        y = x + np.random.normal(0, 0.01, 100)
        z = np.random.normal(0, 1, 100).astype(float)
        model = RandomForestRegressor()
        
        cmi_xyz = CondMutualInfoModel(x, "c", y, "c", z, "c")(model)
        print(f"CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertGreater(cmi_xyz, 0.0)
        
        # X和Z相关; X, Z与Y无关
        x = np.arange(1000).astype(float)
        y = np.random.normal(0, 1, 1000).astype(float)
        z = x + np.random.normal(0, 0.1, 1000)
        model = RandomForestRegressor()
        
        cmi_xyz = CondMutualInfoModel(x, "c", y, "c", z, "c")(model)
        print(f"CMI between continous X and continous Y conditioned on continuous Z: {cmi_xyz}")
        self.assertIsInstance(cmi_xyz, float)
        self.assertAlmostEqual(cmi_xyz, 0.0, delta=1e-1)
        

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)