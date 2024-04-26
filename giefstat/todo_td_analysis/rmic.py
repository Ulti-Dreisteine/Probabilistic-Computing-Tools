from minepy import MINE
import numpy as np

ALPHA = 0.6
C = 10


def _cal_base_mic(n):
    """经过排数值实验测试所得"""
    return -0.044 * np.log(n) + 0.4418


class PairwiseRMIC(object):
    """MIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
        self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
        if mic_params is None:
            mic_params = {"alpha": ALPHA, "c": C, "est": "mic_approx"}
        self.mic_params = mic_params
    
    def cal_rmic(self, rebase: bool = True):
        mine = MINE(**self.mic_params)
        mine.compute_score(self.x, self.y)
        mic = mine.mic()
        
        if not rebase:
            return mic
        else:
            base_mic = _cal_base_mic(len(self.x))
            if mic > base_mic:
                return (mic - base_mic) / (1.0 - base_mic)
            else:
                return 0.0