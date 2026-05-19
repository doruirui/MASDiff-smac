from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config.schema import (
    AlgorithmConfig,
    EliteConfig,
    EvoXConfig,
    LoggingConfig,
    MasDiffConfig,
    TruncatedDiffusionConfig,
)
from src.utils.import_utils import ModuleSpec


def _as_modulespec(d: dict[str, Any], *, key: str) -> ModuleSpec:
    if not isinstance(d, dict):
        raise TypeError(f"配置字段 `{key}` 需要是 dict，但得到 {type(d)}")
    class_path = d.get("class_path")
    if not class_path:
        raise ValueError(f"配置字段 `{key}.class_path` 不能为空")
    kwargs = d.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise TypeError(f"配置字段 `{key}.kwargs` 需要是 dict，但得到 {type(kwargs)}")
    return ModuleSpec(class_path=str(class_path), kwargs=kwargs)


def _optional_modulespec(raw: Any, *, key: str) -> ModuleSpec | None:
    if raw is None:
        return None
    if isinstance(raw, dict) and not raw:
        return None
    return _as_modulespec(raw, key=key)


def load_config(path: str | Path) -> MasDiffConfig:
    """
    读取一套 YAML 配置并转换为 MasDiffConfig。

    注意：这是“每套模块一个 yaml”，不是每个模块一个 yaml。
    """
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("配置文件顶层必须是 dict")

    algo_raw = raw.get("algorithm") or {}
    elite_raw = raw.get("elite") or {}
    trunc_raw = raw.get("truncated_diffusion") or {}
    logging_raw = raw.get("logging") or {}
    evox_raw = raw.get("evox") or {}
    evox_algo_raw = evox_raw.get("algorithm") or {}

    cfg = MasDiffConfig(
        seed=int(raw.get("seed", 0)),
        algorithm=AlgorithmConfig(
            M=int(algo_raw.get("M", 1)),
            N=int(algo_raw.get("N", 1)),
            K=int(algo_raw.get("K", 1)),
        ),
        q_provider=_as_modulespec(raw.get("q_provider") or {}, key="q_provider"),
        environment=_as_modulespec(raw.get("environment") or {}, key="environment"),
        dqn_module=_optional_modulespec(raw.get("dqn_module"), key="dqn_module"),
        diffusion_model=_optional_modulespec(raw.get("diffusion_model"), key="diffusion_model"),
        elite_selector=_optional_modulespec(raw.get("elite_selector"), key="elite_selector"),
        metric=_as_modulespec(raw.get("metric") or {}, key="metric"),
        parallel_executor=_as_modulespec(raw.get("parallel_executor") or {}, key="parallel_executor"),
        elite=EliteConfig(elite_count=int(elite_raw.get("elite_count", 1))),
        truncated_diffusion=TruncatedDiffusionConfig(
            add_noise_steps=int(trunc_raw.get("add_noise_steps", 1)),
            denoise_steps=int(trunc_raw.get("denoise_steps", 1)),
        ),
        evox=EvoXConfig(
            algorithm=_as_modulespec(evox_algo_raw, key="evox.algorithm"),
            reward_min=float(evox_raw.get("reward_min", 0.0)),
            reward_max=float(evox_raw.get("reward_max", 5.0)),
            repo_path=(str(evox_raw["repo_path"]) if evox_raw.get("repo_path") else None),
            init_strategy=str(evox_raw.get("init_strategy", "best")),
            device=(str(evox_raw["device"]) if evox_raw.get("device") else None),
            allow_shim_fallback=bool(evox_raw.get("allow_shim_fallback", True)),
        ),
        logging=LoggingConfig(
            best_rho_csv_path=str(logging_raw.get("best_rho_csv_path", "outputs/best_rho_history.csv")),
            initial_population_path=(
                str(logging_raw["initial_population_path"])
                if logging_raw.get("initial_population_path")
                else None
            ),
        ),
        extra=dict(raw.get("extra") or {}),
    )

    # 轻量校验
    if cfg.algorithm.M <= 0:
        raise ValueError("algorithm.M 必须 > 0")
    if cfg.algorithm.N <= 0:
        raise ValueError("algorithm.N 必须 > 0")
    if cfg.algorithm.K < 0:
        raise ValueError("algorithm.K 必须 >= 0")
    if cfg.elite.elite_count <= 0:
        raise ValueError("elite.elite_count 必须 > 0")
    if cfg.evox.reward_max < cfg.evox.reward_min:
        raise ValueError("evox.reward_max 必须 >= evox.reward_min")
    if cfg.evox.init_strategy not in {"best", "mean"}:
        raise ValueError("evox.init_strategy 仅支持 best / mean")

    return cfg

