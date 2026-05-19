from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.dqn.base import DqnModule
from src.environments.base import Environment
from src.metrics.base import Metric
from src.q.base import QProvider


_CACHE_VERSION = 2


@dataclass
class InitialPopulationSeedData:
    """初始种群中单个个体需要缓存的数据。"""

    tau: Any
    experience_buffers: list[list[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def _maybe_seed(seed: int | None) -> None:
    if seed is None:
        return

    try:
        seed = int(seed)
    except Exception:
        return

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def _resolve_ray_value(value: Any) -> Any:
    try:
        import ray  # type: ignore
    except Exception:
        return value

    object_ref_type = getattr(ray, "ObjectRef", None)
    if object_ref_type is not None and isinstance(value, object_ref_type):
        return ray.get(value)
    return value


def _safe_ray_put(value: Any) -> Any:
    # 直接返回原对象，避免 ray.put 在当前环境中触发 Broken pipe。
    return value


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)

    if hasattr(value, "__dict__"):
        return {
            "__class__": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "attrs": _to_jsonable(vars(value)),
        }

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return {"__class__": type(value).__name__, "shape": list(shape)}
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass

    return repr(value)


def _build_cache_identity(
    environment: Environment,
    dqn_module: DqnModule,
    *,
    population_size: int,
    num_agents: int,
    seed: int,
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "population_size": int(population_size),
        "num_agents": int(num_agents),
        "seed": int(seed),
        "environment": _to_jsonable(environment),
        "dqn_module": _to_jsonable(dqn_module),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _default_initial_population_path(
    environment: Environment,
    dqn_module: DqnModule,
    *,
    population_size: int,
    num_agents: int,
    seed: int,
) -> Path:
    cache_id = _build_cache_identity(
        environment,
        dqn_module,
        population_size=population_size,
        num_agents=num_agents,
        seed=seed,
    )[:12]
    return Path("outputs") / "fitness_initial_population" / (
        f"{environment.__class__.__name__}_{dqn_module.__class__.__name__}"
        f"_M{population_size}_N{num_agents}_{cache_id}.pkl"
    )


def _infer_num_agents(single_reward: Any) -> int:
    shape = getattr(single_reward, "shape", None)
    if shape is not None:
        try:
            if len(shape) >= 1:
                return int(shape[0])
        except Exception:
            pass

    try:
        return int(len(single_reward))
    except Exception as e:
        raise TypeError("无法从单个奖励 R 推断智能体数量 N。") from e


def _normalize_rewards_batch(rewards_batch: Sequence[Any], population_size: int) -> tuple[list[Any], int]:
    rewards_list = list(rewards_batch)
    if population_size <= 0:
        raise ValueError("population_size 必须 > 0。")
    if len(rewards_list) != population_size:
        raise ValueError(
            f"rewards_batch 数量({len(rewards_list)})必须与 population_size({population_size})一致。"
        )
    if not rewards_list:
        raise ValueError("rewards_batch 不能为空。")

    num_agents = _infer_num_agents(rewards_list[0])
    if num_agents <= 0:
        raise ValueError("从 rewards_batch 推断得到的智能体数量 N 必须 > 0。")

    for idx, rewards in enumerate(rewards_list):
        current_num_agents = _infer_num_agents(rewards)
        if current_num_agents != num_agents:
            raise ValueError(
                f"第 {idx} 个奖励的智能体数量为 {current_num_agents}，"
                f"与第 0 个奖励的智能体数量 {num_agents} 不一致。"
            )

    return rewards_list, num_agents


def _coerce_seed_population_item(item: Any, *, index: int) -> InitialPopulationSeedData:
    if isinstance(item, InitialPopulationSeedData):
        return item

    if isinstance(item, dict):
        if "experience_buffers" not in item:
            raise TypeError(f"缓存中的第 {index} 个个体缺少 experience_buffers。")
        return InitialPopulationSeedData(
            tau=item.get("tau"),
            experience_buffers=item["experience_buffers"],
            metadata=dict(item.get("metadata") or {}),
        )

    if hasattr(item, "tau") and hasattr(item, "experience_buffers"):
        return InitialPopulationSeedData(
            tau=getattr(item, "tau"),
            experience_buffers=getattr(item, "experience_buffers"),
            metadata=dict(getattr(item, "metadata", {}) or {}),
        )

    raise TypeError(f"缓存中的第 {index} 个个体类型不受支持: {type(item).__name__}")


def _try_load_initial_population(
    population_path: Path,
    *,
    cache_identity: str,
    population_size: int,
    num_agents: int,
) -> list[InitialPopulationSeedData] | None:
    if not population_path.exists():
        return None

    with population_path.open("rb") as f:
        cached = pickle.load(f)

    if isinstance(cached, list):
        if len(cached) != population_size:
            return None
        return [_coerce_seed_population_item(item, index=i) for i, item in enumerate(cached)]

    if not isinstance(cached, dict):
        return None

    meta = cached.get("meta") or {}
    if meta.get("cache_identity") != cache_identity:
        return None
    if int(meta.get("population_size", -1)) != population_size:
        return None
    if int(meta.get("num_agents", -1)) != num_agents:
        return None

    population = cached.get("population")
    if not isinstance(population, list) or len(population) != population_size:
        return None

    return [_coerce_seed_population_item(item, index=i) for i, item in enumerate(population)]


def _save_initial_population(
    population_path: Path,
    *,
    cache_identity: str,
    population_size: int,
    num_agents: int,
    population: list[InitialPopulationSeedData],
) -> None:
    population_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _CACHE_VERSION,
        "meta": {
            "cache_identity": cache_identity,
            "population_size": int(population_size),
            "num_agents": int(num_agents),
        },
        "population": population,
    }
    with population_path.open("wb") as f:
        pickle.dump(payload, f)


def _ensure_ray_initialized(ray_init_kwargs: dict[str, Any] | None = None) -> bool:
    import ray  # type: ignore

    if ray.is_initialized():
        return False

    ray.init(**(ray_init_kwargs or {}))
    return True


def _task_options_for(
    task_name: str,
    *,
    default_task_options: dict[str, Any] | None = None,
    task_options_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    options = dict(default_task_options or {})
    if task_options_by_name and task_name in task_options_by_name:
        options.update(task_options_by_name[task_name])
    return options


def _ray_map(
    fn: Any,
    tasks: list[dict[str, Any]],
    *,
    task_name: str,
    default_task_options: dict[str, Any] | None = None,
    task_options_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    import ray  # type: ignore

    if not tasks:
        return []

    options = _task_options_for(
        task_name,
        default_task_options=default_task_options,
        task_options_by_name=task_options_by_name,
    )
    remote_fn = ray.remote(fn).options(**options) if options else ray.remote(fn)
    refs = [remote_fn.remote(task) for task in tasks]
    return list(ray.get(refs))


def build_fitness_initial_seed_data(task: dict[str, Any]) -> InitialPopulationSeedData:
    """
    Ray 并行任务：
    随机初始化策略，仿真一次，收集 experience_buffers 和 tau。
    """
    _maybe_seed(task.get("seed"))

    import ray  # type: ignore

    environment: Environment = _resolve_ray_value(task["environment_ref"])
    dqn_module: DqnModule = _resolve_ray_value(task["dqn_module_ref"])
    num_agents = int(task["num_agents"])
    index = int(task["index"])

    init_policies = dqn_module.init_random_policies(num_agents)
    experience_buffers, tau = environment.simulate_collect(init_policies)

    return InitialPopulationSeedData(
        tau=tau,
        experience_buffers=experience_buffers,
        metadata={"initial_index": index},
    )


def evaluate_fitness_for_reward(task: dict[str, Any]) -> float:
    """
    Ray 并行任务：
    用给定奖励 R 填充经验，训练 DQN，再仿真并计算 rho。
    """
    _maybe_seed(task.get("seed"))

    import ray  # type: ignore

    environment: Environment = _resolve_ray_value(task["environment_ref"])
    dqn_module: DqnModule = _resolve_ray_value(task["dqn_module_ref"])
    metric: Metric = _resolve_ray_value(task["metric_ref"])
    q = _resolve_ray_value(task["q_ref"])

    seed_data = _coerce_seed_population_item(task["seed_data"], index=int(task["index"]))
    rewards = task["rewards"]

    training_data = dqn_module.build_training_data(seed_data.experience_buffers, rewards)
    trained_policies = dqn_module.train_per_agent(training_data)
    simulation_data = environment.simulate_evaluate(trained_policies)
    rho = metric.compute_rho(q, simulation_data)
    return float(rho)


def calculate_fitness_based_on_rewards(
    environment: Environment,
    rewards_batch: Sequence[Any],
    population_size: int,
    *,
    dqn_module: DqnModule,
    metric: Metric,
    q: Any | None = None,
    q_provider: QProvider | None = None,
    initial_population_path: str | Path | None = None,
    ray_init_kwargs: dict[str, Any] | None = None,
    default_task_options: dict[str, Any] | None = None,
    task_options_by_name: dict[str, dict[str, Any]] | None = None,
    seed: int = 0,
) -> list[float]:
    """
    输入：
    - environment
    - rewards_batch: 批量奖励 R
    - population_size: 种群规模 M

    输出：
    - 每个 R 对应的适应度列表

    说明：
    - 不再依赖 steps.py / executor；
    - 只直接调用必要对象自身的方法：
      - q_provider.load_or_create_q
      - dqn_module.init_random_policies / build_training_data / train_per_agent
      - environment.simulate_collect / simulate_evaluate
      - metric.compute_rho
    - 初始数据创建、DQN 训练、仿真评估都使用 Ray 并行。
    """
    if not isinstance(environment, Environment):
        raise TypeError("environment 必须是 Environment 实例。")
    if not isinstance(dqn_module, DqnModule):
        raise TypeError("dqn_module 必须是 DqnModule 实例。")
    if not isinstance(metric, Metric):
        raise TypeError("metric 必须是 Metric 实例。")

    rewards_list, num_agents = _normalize_rewards_batch(rewards_batch, population_size)
    cache_identity = _build_cache_identity(
        environment,
        dqn_module,
        population_size=population_size,
        num_agents=num_agents,
        seed=seed,
    )
    population_path = (
        Path(initial_population_path)
        if initial_population_path is not None
        else _default_initial_population_path(
            environment,
            dqn_module,
            population_size=population_size,
            num_agents=num_agents,
            seed=seed,
        )
    )

    owns_ray = _ensure_ray_initialized(ray_init_kwargs)
    try:
        if q is None:
            if q_provider is None:
                raise ValueError("未传入 q 时，必须提供 q_provider。")
            q = q_provider.load_or_create_q(environment)

        import ray  # type: ignore

        initial_population = _try_load_initial_population(
            population_path,
            cache_identity=cache_identity,
            population_size=population_size,
            num_agents=num_agents,
        )
        if initial_population is not None:
            print(f"读取初始种群缓存: {population_path}")
        else:
            print(f"未找到可用初始种群缓存，开始创建: {population_path}")
            environment_ref = _safe_ray_put(environment)
            dqn_module_ref = _safe_ray_put(dqn_module)
            init_tasks = [
                {
                    "index": i,
                    "num_agents": num_agents,
                    "seed": int(seed) + i,
                    "environment_ref": environment_ref,
                    "dqn_module_ref": dqn_module_ref,
                }
                for i in range(population_size)
            ]
            initial_population = _ray_map(
                build_fitness_initial_seed_data,
                init_tasks,
                task_name="build_fitness_initial_seed_data",
                default_task_options=default_task_options,
                task_options_by_name=task_options_by_name,
            )
            _save_initial_population(
                population_path,
                cache_identity=cache_identity,
                population_size=population_size,
                num_agents=num_agents,
                population=initial_population,
            )
            print(f"初始种群缓存已保存: {population_path}")

        environment_ref = _safe_ray_put(environment)
        dqn_module_ref = _safe_ray_put(dqn_module)
        metric_ref = _safe_ray_put(metric)
        q_ref = _safe_ray_put(q)
        fitness_tasks = [
            {
                "index": i,
                "seed": int(seed) + 100000 + i,
                "environment_ref": environment_ref,
                "dqn_module_ref": dqn_module_ref,
                "metric_ref": metric_ref,
                "q_ref": q_ref,
                "seed_data": initial_population[i],
                "rewards": rewards_list[i],
            }
            for i in range(population_size)
        ]

        fitness_list = _ray_map(
            evaluate_fitness_for_reward,
            fitness_tasks,
            task_name="evaluate_fitness_for_reward",
            default_task_options=default_task_options,
            task_options_by_name=task_options_by_name,
        )
        return [float(v) for v in fitness_list]
    finally:
        if owns_ray:
            try:
                import ray  # type: ignore

                ray.shutdown()
            except Exception:
                pass


__all__ = [
    "InitialPopulationSeedData",
    "build_fitness_initial_seed_data",
    "evaluate_fitness_for_reward",
    "calculate_fitness_based_on_rewards",
]
