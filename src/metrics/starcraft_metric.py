import numpy as np
from src.metrics.base import Metric

class StarcraftMetric(Metric):
    """将环境返回的奖励数组转换为适应度 rho"""
    def __init__(self, transform: str = "win_rate", eps: float = 1e-8):
        self.transform = transform
        self.eps = eps

    def compute_rho(self, rewards: np.ndarray) -> float:
        """实现基类的抽象方法"""
        if self.transform == "win_rate":
            # 假设 rewards 为多个 episode 的胜率（0/1）或累积奖励
            return float(np.mean(rewards))
        elif self.transform == "identity":
            return float(rewards[0])
        else:
            raise ValueError(f"Unknown transform {self.transform}")

    def __call__(self, rewards: np.ndarray) -> float:
        return self.compute_rho(rewards)