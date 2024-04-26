from minepy import MINE
import numpy as np

ALPHA = 0.6
C = 10


class PairwiseMIC(object):
    """MIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
        self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
        if mic_params is None:
            mic_params = {"alpha": ALPHA, "c": C, "est": "mic_approx"}
        self.mic_params = mic_params
    
    def cal_mic(self):
        mine = MINE(**self.mic_params)
        mine.compute_score(self.x, self.y)
        return mine.mic()