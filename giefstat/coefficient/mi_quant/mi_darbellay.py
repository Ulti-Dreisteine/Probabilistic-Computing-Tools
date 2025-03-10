import numpy as np

from ...util import stdize_values
from ._quant_darbellay import exec_partition, Cell


class MutualInfoDarbellay(object):
    """
    基于Darbellay自适应分箱的互信息计算
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Note:
        -----
        由于本项目中的Darbellay离散化算法设置, 需要对变量数据进行标准化至0-1区间
        """
        
        self.x_norm = stdize_values(x, "c")
        self.y_norm = stdize_values(y, "c")
        
        try:
            assert self.x_norm.shape[1] == 1
            assert self.y_norm.shape[1] == 1
        except Exception as e:
            raise ValueError("不支持非一维变量间的MI计算") from e
    
    def __call__(self) -> float:
        leaf_cells, arr_norm = exec_partition(self.x_norm, self.y_norm)
        N_total, _ = arr_norm.shape
        
        # 计算互信息
        n_leafs = len(leaf_cells)

        mi = 0.0
        
        for i in range(n_leafs):
            cell = leaf_cells[i]  # type: Cell
            (xl, xu), (yl, yu) = cell.bounds

            Nxy = len(cell.arr)
            Nx = len(
                np.where((arr_norm[:, 0] >= xl) & (arr_norm[:, 0] < xu))[0])
            Ny = len(
                np.where((arr_norm[:, 1] >= yl) & (arr_norm[:, 1] < yu))[0])
            
            gain = Nxy * np.log(Nxy / Nx / Ny)
            mi += gain
        
        mi = round(mi / N_total + np.log(N_total), 4)
        
        return mi