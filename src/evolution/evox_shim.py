from __future__ import annotations

import importlib
from typing import Any

try:
    torch = importlib.import_module("torch")
except Exception as e:  # pragma: no cover
    raise ImportError("evox_shim 需要 torch。请先安装 torch。") from e


class Problem:
    """Minimal Problem base class used by local fallback workflow."""

    def evaluate(self, pop: Any) -> Any:  # pragma: no cover - interface only
        raise NotImplementedError()


class OpenES:
    """Lightweight OpenES fallback when `evox` package is unavailable."""

    def __init__(
        self,
        *,
        pop_size: int,
        center_init: Any,
        learning_rate: float = 0.05,
        noise_stdev: float = 0.1,
        mirrored_sampling: bool = True,
        device: str | None = None,
        **_: Any,
    ) -> None:
        center = center_init.detach().float() if hasattr(center_init, "detach") else torch.tensor(center_init)
        center = center.reshape(-1)
        if device is not None:
            center = center.to(device)

        self.center = center
        self.pop_size = max(1, int(pop_size))
        self.learning_rate = float(learning_rate)
        self.noise_stdev = max(1.0e-8, float(noise_stdev))
        self.mirrored_sampling = bool(mirrored_sampling)

        self._last_noise: Any | None = None

    def ask(self) -> Any:
        dim = int(self.center.numel())
        dtype = self.center.dtype
        device = self.center.device

        if self.mirrored_sampling:
            half = (self.pop_size + 1) // 2
            eps = torch.randn((half, dim), dtype=dtype, device=device)
            noise = torch.cat([eps, -eps], dim=0)[: self.pop_size]
        else:
            noise = torch.randn((self.pop_size, dim), dtype=dtype, device=device)

        self._last_noise = noise
        return self.center.unsqueeze(0) + self.noise_stdev * noise

    def tell(self, fitness: Any, *, maximize: bool = True) -> None:
        if self._last_noise is None:
            return None

        fit = fitness.detach().float() if hasattr(fitness, "detach") else torch.tensor(fitness, dtype=torch.float32)
        fit = fit.reshape(-1)
        if fit.numel() != self._last_noise.shape[0]:
            raise ValueError("fitness size 与种群数量不一致。")

        fit = (fit - fit.mean()) / (fit.std(unbiased=False) + 1.0e-8)
        grad = torch.matmul(fit.to(self._last_noise.device), self._last_noise)
        grad = grad / (float(self.pop_size) * self.noise_stdev)

        direction = 1.0 if maximize else -1.0
        self.center = self.center + direction * self.learning_rate * grad.to(self.center.device)
        return None


class DE:
    """Lightweight DE-style fallback for environments without `evox` package."""

    def __init__(
        self,
        *,
        pop_size: int,
        lb: Any,
        ub: Any,
        mean: Any | None = None,
        stdev: Any | None = None,
        device: str | None = None,
        **_: Any,
    ) -> None:
        lb_t = lb.detach().float() if hasattr(lb, "detach") else torch.tensor(lb, dtype=torch.float32)
        ub_t = ub.detach().float() if hasattr(ub, "detach") else torch.tensor(ub, dtype=torch.float32)

        if device is not None:
            lb_t = lb_t.to(device)
            ub_t = ub_t.to(device)

        if mean is None:
            center = (lb_t + ub_t) / 2.0
        else:
            center = mean.detach().float() if hasattr(mean, "detach") else torch.tensor(mean, dtype=torch.float32)
            if device is not None:
                center = center.to(device)

        if stdev is None:
            noise_stdev = 0.1
        else:
            st = stdev.detach().float() if hasattr(stdev, "detach") else torch.tensor(stdev, dtype=torch.float32)
            noise_stdev = max(1.0e-6, float(st.abs().mean().item()))

        self.lb = lb_t.reshape(-1)
        self.ub = ub_t.reshape(-1)
        self._openes = OpenES(
            pop_size=int(pop_size),
            center_init=center.reshape(-1),
            learning_rate=0.05,
            noise_stdev=noise_stdev,
            mirrored_sampling=False,
            device=device,
        )

    def ask(self) -> Any:
        pop = self._openes.ask()
        return torch.max(torch.min(pop, self.ub.unsqueeze(0)), self.lb.unsqueeze(0))

    def tell(self, fitness: Any, *, maximize: bool = True) -> None:
        self._openes.tell(fitness, maximize=maximize)
        self._openes.center = torch.max(torch.min(self._openes.center, self.ub), self.lb)
        return None


class StdWorkflow:
    """Minimal workflow compatible with runner usage (`init_step` / `step`)."""

    def __init__(self, algorithm: Any, problem: Any, *, opt_direction: str = "max", device: str | None = None) -> None:
        self.algorithm = algorithm
        self.problem = problem
        self.opt_direction = str(opt_direction).lower()
        self.device = device

    def _one_step(self) -> None:
        pop = self.algorithm.ask()
        fitness = self.problem.evaluate(pop)
        self.algorithm.tell(fitness, maximize=(self.opt_direction != "min"))

    def init_step(self) -> None:
        self._one_step()

    def step(self) -> None:
        self._one_step()
