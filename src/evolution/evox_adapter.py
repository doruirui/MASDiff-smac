from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

from src.config.schema import EvoXConfig
from src.pipeline.types import Individual, Population
from src.utils.import_utils import import_symbol


def _ordered_unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _repo_root_candidates(repo_path: str | None) -> list[Path]:
    roots: list[Path] = []

    def _add(path: Path) -> None:
        roots.append(path.expanduser())

    if repo_path:
        raw = Path(repo_path)
        _add(raw)
        if not raw.is_absolute():
            _add(Path.cwd() / raw)
            try:
                argv0 = Path(sys.argv[0])
                if argv0.parts:
                    _add(argv0.resolve().parent / raw)
            except Exception:
                pass

    env_repo = os.environ.get("EVOX_REPO_PATH")
    if env_repo:
        env_raw = Path(env_repo)
        _add(env_raw)
        if not env_raw.is_absolute():
            _add(Path.cwd() / env_raw)

    return _ordered_unique_paths(roots)


def _attempted_evox_paths(repo_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add_root(root: Path) -> None:
        for candidate in (root, root / "src"):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    for root in _repo_root_candidates(repo_path):
        _add_root(root)

    anchor = Path(__file__).resolve()
    common_names = ["evox-main-only框架", "evox-main-only", "evox"]
    parent_chain = list(anchor.parents)
    cwd_chain = [Path.cwd(), *list(Path.cwd().parents)[:5]]
    for base in _ordered_unique_paths([anchor.parent, *parent_chain[:6], *cwd_chain]):
        for name in common_names:
            _add_root(base / name)

    return candidates


def _candidate_evox_paths(repo_path: str | None) -> list[Path]:
    return [p for p in _attempted_evox_paths(repo_path) if p.exists()]


def _format_paths(paths: list[Path]) -> str:
    if not paths:
        return "<none>"
    return ", ".join(str(p) for p in paths)


def ensure_evox_importable(repo_path: str | None) -> list[Path]:
    injected: list[Path] = []
    for candidate in _candidate_evox_paths(repo_path):
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
            injected.append(candidate)
    return injected


def import_torch_and_evox(repo_path: str | None, *, allow_shim_fallback: bool = True) -> tuple[Any, type[Any]]:
    injected = ensure_evox_importable(repo_path)
    attempted = _attempted_evox_paths(repo_path)
    try:
        torch = importlib.import_module("torch")
    except Exception as e:
        raise ImportError("运行 EvoX 进化需要先安装 torch。") from e

    try:
        problem_base = importlib.import_module("evox.core").Problem
    except Exception as e:
        if not allow_shim_fallback:
            searched_text = _format_paths(attempted)
            injected_text = _format_paths(injected)
            raise ImportError(
                "未检测到真实 EvoX（evox.core）。当前配置禁止 shim 回退。"
                " 请安装 evox 或修正 `evox.repo_path`。"
                f" 已尝试路径: {searched_text}; 已注入 sys.path: {injected_text}"
            ) from e

        try:
            from src.evolution.evox_shim import Problem as problem_base

            warnings.warn(
                "未检测到 evox 包，已回退到本地轻量 evox_shim（仅保证基本 OpenES 工作流）。",
                RuntimeWarning,
            )
            return torch, problem_base
        except Exception:
            pass

        searched_text = _format_paths(attempted)
        injected_text = _format_paths(injected)
        raise ImportError(
            "无法导入 EvoX。请先安装 evox，或把 evox 仓库路径写到配置 `evox.repo_path`。"
            f" 已尝试路径: {searched_text}; 已注入 sys.path: {injected_text}"
        ) from e
    return torch, problem_base


def _import_algorithm_class(class_path: str, *, allow_shim_fallback: bool = True) -> type[Any]:
    try:
        return import_symbol(class_path)
    except Exception:
        pass

    if class_path.startswith("evox.algorithms:"):
        if not allow_shim_fallback:
            raise ImportError(
                f"无法导入真实 EvoX 算法: {class_path}。当前配置禁止 shim 回退。"
            )
        algo_name = class_path.split(":", 1)[1].rsplit(".", 1)[-1]
        from src.evolution import evox_shim

        algo_cls = getattr(evox_shim, algo_name, None)
        if algo_cls is not None:
            warnings.warn(
                f"算法 `{algo_name}` 使用本地 evox_shim 回退实现。",
                RuntimeWarning,
            )
            return algo_cls

    raise ImportError(f"无法导入进化算法: {class_path}")


def get_evox_std_workflow_cls(repo_path: str | None, *, allow_shim_fallback: bool = True) -> type[Any]:
    ensure_evox_importable(repo_path)
    try:
        return importlib.import_module("evox.workflows").StdWorkflow
    except Exception:
        if not allow_shim_fallback:
            searched = _attempted_evox_paths(repo_path)
            searched_text = _format_paths(searched)
            raise ImportError(
                "未检测到真实 EvoX workflow（evox.workflows.StdWorkflow）。"
                " 当前配置禁止 shim 回退。"
                f" 已尝试路径: {searched_text}"
            )

        from src.evolution.evox_shim import StdWorkflow

        warnings.warn(
            "未检测到 evox.workflows，已回退到本地 StdWorkflow shim。",
            RuntimeWarning,
        )
        return StdWorkflow


def _flatten_reward(rewards: Any, torch_mod: Any) -> Any:
    if hasattr(rewards, "detach"):
        tensor = rewards.detach()
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        return tensor.reshape(-1)
    return torch_mod.tensor(rewards, dtype=torch_mod.float32).reshape(-1)


def _population_reward_vectors(initial_population: Population, torch_mod: Any) -> Any:
    if len(initial_population) == 0:
        raise ValueError("initial_population 不能为空，无法初始化 EvoX 算法。")
    return torch_mod.stack([_flatten_reward(ind.rewards, torch_mod) for ind in initial_population], dim=0)


def build_evox_algorithm(evox_cfg: EvoXConfig, initial_population: Population) -> Any:
    torch, _ = import_torch_and_evox(
        evox_cfg.repo_path,
        allow_shim_fallback=evox_cfg.allow_shim_fallback,
    )
    algo_cls = _import_algorithm_class(
        evox_cfg.algorithm.class_path,
        allow_shim_fallback=evox_cfg.allow_shim_fallback,
    )
    kwargs = dict(evox_cfg.algorithm.kwargs or {})

    # ========== 关键修改：从种群个体参数直接获取维度 ==========
    if len(initial_population) == 0:
        raise ValueError("initial_population 不能为空，无法初始化 EvoX 算法。")
    dim = initial_population[0].params.shape[0]
    print(f"[build_evox_algorithm] 从种群参数推断维度: {dim}")
    # =======================================================

    pop_size = int(kwargs.get("pop_size", max(1, len(initial_population))))
    kwargs["pop_size"] = pop_size

    device = kwargs.get("device", evox_cfg.device)
    # 准备用于初始化算法的张量：中心点、上下界等
    all_params = torch.stack([torch.from_numpy(ind.params).float() for ind in initial_population])
    if device is not None:
        all_params = all_params.to(device)
        print(f"[build_evox_algorithm] 已将张量移动到设备: {device}")

    mean_center = all_params.mean(dim=0)
    best_idx = max(range(len(initial_population)), key=lambda idx: float(initial_population[idx].rho))
    best_center = all_params[best_idx]

    lb = torch.full((dim,), float(evox_cfg.reward_min), dtype=torch.float32, device=device)
    ub = torch.full((dim,), float(evox_cfg.reward_max), dtype=torch.float32, device=device)

    center = best_center if evox_cfg.init_strategy == "best" else mean_center

    algo_name = evox_cfg.algorithm.class_path.split(":")[-1].rsplit(".", 1)[-1]
    if algo_name == "OpenES":
        kwargs.setdefault("center_init", center)
    elif algo_name in {"DE", "SHADE", "CoDE", "SaDE", "ODE", "JaDE"}:
        kwargs.setdefault("lb", lb)
        kwargs.setdefault("ub", ub)
        kwargs.setdefault("mean", mean_center)
        stdev = all_params.std(dim=0)
        kwargs.setdefault("stdev", torch.clamp(stdev, min=1e-3))
    elif algo_name in {"PSO", "CLPSO", "CSO", "FSPSO", "DMSPSOEL", "SLPSOGS", "SLPSOUS"}:
        kwargs.setdefault("lb", lb)
        kwargs.setdefault("ub", ub)
    else:
        raise NotImplementedError(
            f"当前只支持接入 OpenES / DE 类 / PSO 类 EvoX 算法，暂不支持 `{algo_name}`。"
        )

    if device is not None:
        kwargs.setdefault("device", device)
    return algo_cls(**kwargs)


def build_evox_problem(
    *,
    evox_cfg: EvoXConfig,
    reward_shape: tuple[int, int],
    batch_evaluator: Callable[[Any], Population],
) -> Any:
    torch, problem_base = import_torch_and_evox(
        evox_cfg.repo_path,
        allow_shim_fallback=evox_cfg.allow_shim_fallback,
    )
    num_car, num_road = reward_shape

    class RewardPopulationProblem(problem_base):
        def __init__(self) -> None:
            super().__init__()
            self.last_population: Population = []
            self.best_individual: Individual | None = None

        def evaluate(self, pop: Any) -> Any:
            population = batch_evaluator(pop)
            self.last_population = list(population)
            if population:
                best = max(population, key=lambda ind: float(ind.rho))
                if (self.best_individual is None) or (float(best.rho) > float(self.best_individual.rho)):
                    self.best_individual = best

            fitness = [float(ind.rho) for ind in population]
            return torch.tensor(
                fitness,
                dtype=getattr(pop, "dtype", torch.float32),
                device=getattr(pop, "device", None),
            )

        @property
        def expected_reward_shape(self) -> tuple[int, int]:
            return num_car, num_road

    return RewardPopulationProblem()