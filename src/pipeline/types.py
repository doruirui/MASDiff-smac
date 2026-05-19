from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class Individual:
    params: np.ndarray   # 扁平参数向量
    rewards: np.ndarray  # 原始奖励数组（可多维）
    rho: float           # 适应度值

Population = List[Individual]