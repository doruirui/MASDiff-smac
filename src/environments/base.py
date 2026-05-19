from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from src.pipeline.types import Individual

class Environment(ABC):
    @abstractmethod
    def evaluate(self, individual: Individual) -> Individual:
        """给一个个体的参数，返回评估后的个体（含奖励和 rho）"""
        pass
class Environment(ABC):
    """
    环境/仿真器抽象接口（用户自定义实现）。

    主流程中，环境主要负责两类仿真：
    - 收集经验库 + Tau（奖励先留空）
    - 给定策略/奖励等输入跑一次，产出“仿真数据”（结构与 Q 相同/可对齐），用于计算 ρ
    """

    @abstractmethod
    def simulate_collect(self, planner_input: Any) -> tuple[list[list[Any]], Any]:
        """
        用给定输入仿真一次，收集每个智能体的经验库与 Tau。

        返回值：
        - experience_buffers: 长度为 N 的列表；每个元素是一个经验序列。
          每条经验建议形如 [s, a, r]，其中 r 在收集时留空（例如 None）。
          s 建议形如 [当前状态(如道路id、坐标等), [特征1, 特征2, ...]]（特征大小由配置决定）
        - tau: Tau（用户自定义结构），用于作为扩散模型的条件生成奖励 R
        """

    @abstractmethod
    def simulate_evaluate(self, planner_input: Any) -> Any:
        """
        用给定输入仿真一次，返回用于评估的“仿真数据”（结构建议与 Q 相同或可对齐）。

        返回值用途：
        - 与 Q 一起输入到 metric 计算 ρ（适应度/指标，越大越好）
        """

