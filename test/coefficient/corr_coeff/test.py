# -*- coding: utf-8 -*-
"""
Created on 2024/03/06 15:38:07

@File -> test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: END
"""

import numpy as np
import unittest
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from giefstat.coefficient.corr_coeff.coeff import cal_dist_corr, cal_pearson_corr, cal_spearman_corr


class CorrCoeffTest(unittest.TestCase):
    """
    测试CorrCoeff
    """
    
    def setup(self):
        print("testing task setup")
        
    def test_pearson_corr(self):
        x = np.array([1, 2, 3, 4]).astype(float)
        y = np.array([1, 2, 3, 4.1]).astype(float)
        self.assertIsInstance(cal_pearson_corr(x, y), float)
        self.assertGreater(cal_pearson_corr(x, y), 0.0)
        self.assertLessEqual(cal_pearson_corr(x, y), 1.0)
        
    def test_dist_corr(self):
        x = np.array([1, 2, 3, 4]).astype(float)
        y = np.array([1, 2, 3, 4.1]).astype(float)
        self.assertIsInstance(cal_dist_corr(x, y), float)
        self.assertGreater(cal_dist_corr(x, y), 0.0)
        self.assertLessEqual(cal_dist_corr(x, y), 1.0)
        
    def test_cal_spearman_corr(self):
        x = np.array([1, 2, 3, 4]).astype(float)
        y = np.array([1, 2, 3, 4.1]).astype(float)
        self.assertIsInstance(cal_spearman_corr(x, y), float)
        self.assertGreater(cal_spearman_corr(x, y), 0.0)
        self.assertLessEqual(cal_spearman_corr(x, y), 1.0)
        

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)