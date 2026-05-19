from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any, Optional

try:
    torch = importlib.import_module("torch")
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 torch。`src/metrics/sumo_ryl_metric.py` 需要 PyTorch 来计算张量 MSE。"
    ) from e

from src.metrics.base import Metric


class SumoRylMetric(Metric):
    """
    sumo_ryl 的 rho 计算：
    1. 对 q 和 simulation_data 计算每条路的误差
    2. 可选地根据 q 为每条路生成更大的权重
    3. 将整体误差通过单调变换映射成 rho（越大越好）
    """

    def __init__(
        self,
        *,
        transform: str = "inv1p",
        eps: float = 1e-8,
        temperature: float = 1.0,
        rho_scale: float = 1.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        weighted_by_q: bool = False,
        weight_alpha: float = 1.5,
        weight_base: float = 0.3,
        weight_use_log1p: bool = True,
        weight_csv_path: str = "",
    ) -> None:
        self.transform = str(transform).lower()
        self.eps = float(eps)
        self.temperature = float(temperature)
        self.rho_scale = float(rho_scale)
        self.clip_min = clip_min if clip_min is None else float(clip_min)
        self.clip_max = clip_max if clip_max is None else float(clip_max)

        self.weighted_by_q = bool(weighted_by_q)
        self.weight_alpha = float(weight_alpha)
        self.weight_base = float(weight_base)
        self.weight_use_log1p = bool(weight_use_log1p)
        self.weight_csv_path = str(weight_csv_path)

    def compute_rho(self, q: Any, simulation_data: Any) -> float:
        q_t = self._to_tensor(q, name="q")
        s_t = self._to_tensor(simulation_data, name="simulation_data")
        q_t, s_t = self._align_shapes(q_t, s_t)

        if q_t.shape != s_t.shape:
            raise ValueError(f"q 与 simulation_data 形状必须一致，但得到 q={tuple(q_t.shape)}，sim={tuple(s_t.shape)}")

        q_road = self._reduce_to_per_road(q_t, name="q")
        s_road = self._reduce_to_per_road(s_t, name="simulation_data")
        abs_err_road = torch.abs(q_road - s_road)

        if self.weighted_by_q:
            road_weights = torch.clamp(q_road, min=0.0)
            mse = torch.sum(road_weights * abs_err_road) / torch.clamp(road_weights.sum(), min=1e-6)
        else:
            road_weights = torch.ones_like(q_road)
            mse = torch.mean(abs_err_road)

        self._maybe_write_weight_csv(
            q_road=q_road,
            simulation_road=s_road,
            road_weights=road_weights,
            abs_err_road=abs_err_road,
        )

        if self.transform == "neg":
            rho = -mse
        elif self.transform == "inv":
            rho = 1.0 / (mse + max(self.eps, 1e-12))
        elif self.transform == "inv1p":
            rho = 1.0 / (1.0 + mse)
        elif self.transform == "exp":
            temp = max(self.temperature, 1e-12)
            rho = torch.exp(-mse / temp)
        else:
            raise ValueError(f"未知 transform={self.transform}(支持: neg/inv/inv1p/exp)")

        rho = rho * self.rho_scale
        if self.clip_min is not None or self.clip_max is not None:
            rho = torch.clamp(
                rho,
                min=self.clip_min if self.clip_min is not None else -float("inf"),
                max=self.clip_max if self.clip_max is not None else float("inf"),
            )

        return float(rho.detach().cpu().item())

    def _to_tensor(self, x: Any, *, name: str) -> "torch.Tensor":
        if isinstance(x, torch.Tensor):
            t = x.detach().float().cpu()
        else:
            t = torch.tensor(x, dtype=torch.float32)
        if t.numel() == 0:
            raise ValueError(f"{name} 不能为空")
        return t

    def _align_shapes(self, q_t: "torch.Tensor", s_t: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        if q_t.shape == s_t.shape:
            return q_t, s_t

        q_squeezed = q_t.squeeze()
        s_squeezed = s_t.squeeze()
        if q_squeezed.shape == s_squeezed.shape:
            return q_squeezed, s_squeezed

        return q_t, s_t

    def _reduce_to_per_road(self, x_t: "torch.Tensor", *, name: str) -> "torch.Tensor":
        if x_t.numel() == 0:
            raise ValueError(f"{name} 不能为空")
        if x_t.dim() == 0:
            return x_t.reshape(1)
        if x_t.dim() == 1:
            return x_t
        reduce_dims = tuple(range(x_t.dim() - 1))
        return torch.mean(x_t, dim=reduce_dims)

    def _maybe_write_weight_csv(
        self,
        *,
        q_road: "torch.Tensor",
        simulation_road: "torch.Tensor",
        road_weights: "torch.Tensor",
        abs_err_road: "torch.Tensor",
    ) -> None:
        csv_path = str(self.weight_csv_path or "").strip()
        if not csv_path:
            return None

        if not (
            q_road.shape == simulation_road.shape == road_weights.shape == abs_err_road.shape
        ):
            raise ValueError(
                "q_road、simulation_road、road_weights、abs_err_road 的 shape 必须一致，"
                f"但得到 {tuple(q_road.shape)}, {tuple(simulation_road.shape)}, "
                f"{tuple(road_weights.shape)}, {tuple(abs_err_road.shape)}"
            )

        out_path = Path(csv_path)
        if out_path.parent != Path("."):
            out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["road_index", "q_value", "simulation_value", "weight", "abs_err_road"])
            for road_idx, (q_value, sim_value, weight, abs_err) in enumerate(
                zip(
                    q_road.tolist(),
                    simulation_road.tolist(),
                    road_weights.tolist(),
                    abs_err_road.tolist(),
                )
            ):
                writer.writerow(
                    [int(road_idx), float(q_value), float(sim_value), float(weight), float(abs_err)]
                )
        return None

