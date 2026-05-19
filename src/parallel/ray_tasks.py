from __future__ import annotations

import random
import time
from typing import Any

from src.pipeline.types import Individual
from src.pipeline.steps import (
    step_3_init_random_rewards,
    step_4_evaluate_only_astar_reward,
    step_4_build_only_astar_runtime_individual,
    step_5_3_1_truncated_diffusion_mutate_reward,
    step_5_3_2_simulate_collect,
    step_5_3_3_generate_reward,
    step_5_3_4_simulate_and_compute_rho,
)
from src.utils.import_utils import ModuleSpec, instantiate


def _maybe_cleanup_cuda_cache() -> None:
    """Best-effort cleanup of worker-side Python and CUDA cached memory."""
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def extract_diffusion_state(diffusion_model: Any) -> dict[str, Any]:
    """
    从扩散模型中提取“可广播到 Ray worker”的轻量状态。

    目前主要支持：
    - SumoRylDiffusionModel: 保存 diffusion_model.net.state_dict()
    - 其它实现：若提供 state_dict() 也会尝试使用
    """
    # 优先：有 net（SumoRylDiffusionModel）
    net = getattr(diffusion_model, "net", None)
    if net is not None and hasattr(net, "state_dict") and callable(getattr(net, "state_dict")):
        return {"kind": "net_state_dict", "net": net.state_dict()}

    # 退化：直接 state_dict（若用户实现提供）
    if hasattr(diffusion_model, "state_dict") and callable(getattr(diffusion_model, "state_dict")):
        return {"kind": "state_dict", "state": diffusion_model.state_dict()}

    raise TypeError("扩散模型不支持提取 state_dict（无法在 Ray worker 中复现同一模型参数）。")


def load_diffusion_state(diffusion_model: Any, state: dict[str, Any]) -> None:
    kind = state.get("kind")
    if kind == "net_state_dict":
        net = getattr(diffusion_model, "net", None)
        if net is None or (not hasattr(net, "load_state_dict")):
            raise TypeError("目标扩散模型没有 net.load_state_dict，无法加载广播参数。")
        net.load_state_dict(state["net"])
        return None
    if kind == "state_dict":
        if not hasattr(diffusion_model, "load_state_dict"):
            raise TypeError("目标扩散模型没有 load_state_dict，无法加载广播参数。")
        diffusion_model.load_state_dict(state["state"])
        return None
    raise ValueError(f"未知 diffusion state kind={kind}")


def _maybe_seed(seed: int | None) -> None:
    if seed is None:
        return None
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    return None


def _resolve_ray_value(value: Any) -> Any:
    """Return the payload itself when Ray already auto-resolved the ObjectRef."""
    try:
        import ray  # type: ignore
    except Exception:
        return value

    object_ref_type = getattr(ray, "ObjectRef", None)
    if object_ref_type is not None and isinstance(value, object_ref_type):
        return ray.get(value)
    return value


def _prepare_only_astar_context(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment": instantiate(task["environment_spec"]),
        "metric": instantiate(task["metric_spec"]),
        "q": _resolve_ray_value(task["q_ref"]),
    }


def build_initial_individual(task: dict[str, Any]) -> Individual:
    """
    Ray task：构建初始种群的一个个体（对应 runner.py 中 build_initial_individual 的语义）。

    task 字段（由 runner 组装）：
    - i: 个体索引（仅用于 metadata）
    - N: 智能体数量
    - seed: 可选，任务随机种子
    - q_ref: ray ObjectRef（Q 可能很大，用 object store 传）
    - environment_spec/metric_spec: ModuleSpec（用于 worker 内实例化）
    """
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    i = int(task["i"])
    N = int(task["N"])

    result: Individual | None = None
    try:
        ctx = _prepare_only_astar_context(task)
        q = ctx["q"]
        env = ctx["environment"]
        metric = ctx["metric"]

        initial_rewards = step_3_init_random_rewards(
            q=q,
            num_agents=N,
            reward_min=float(task.get("reward_min", 0.0)),
            reward_max=float(task.get("reward_max", 5.0)),
        )

        result = step_4_evaluate_only_astar_reward(env, initial_rewards, q=q, metric=metric)
        return result
    finally:
        _maybe_cleanup_cuda_cache()


def evaluate_reward_candidate(task: dict[str, Any]) -> Individual:
    """Ray task：评估一个候选奖励矩阵 R，返回 only-astar 运行时个体。"""
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    result: Individual | None = None
    try:
        ctx = _prepare_only_astar_context(task)
        q = ctx["q"]
        env = ctx["environment"]
        metric = ctx["metric"]
        rewards = task["rewards"]
        result = step_4_evaluate_only_astar_reward(env, rewards, q=q, metric=metric)
        return result
    finally:
        _maybe_cleanup_cuda_cache()


def evaluate_reward_batch(task: dict[str, Any]) -> list[Individual]:
    """Ray task：批量评估候选奖励矩阵，减少 driver 提交任务次数。"""
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    results: list[Individual] = []
    try:
        ctx = _prepare_only_astar_context(task)
        q = ctx["q"]
        env = ctx["environment"]
        metric = ctx["metric"]
        rewards_batch_payload = task.get("rewards_batch_ref", task.get("rewards_batch"))
        rewards_batch = list(_resolve_ray_value(rewards_batch_payload) or [])

        for rewards in rewards_batch:
            ind = step_4_evaluate_only_astar_reward(env, rewards, q=q, metric=metric)
            results.append(ind)
        return results
    finally:
        _maybe_cleanup_cuda_cache()


def train_diffusion_on_population(task: dict[str, Any]) -> dict[str, Any]:
    """
    Ray task：在远程侧直接消费初始种群 refs，执行 5.1 扩散训练。

    返回：
    - diffusion_state: 训练后的扩散模型参数 state_dict
    """
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    diffusion_state = _resolve_ray_value(task["diffusion_state_ref"])
    population_refs = list(task["population_refs"])

    diffusion = instantiate(task["diffusion_model_spec"])
    load_diffusion_state(diffusion, diffusion_state)

    population = list(ray.get(population_refs))
    diffusion.train_on_population(population)

    return {
        "diffusion_state": extract_diffusion_state(diffusion),
        "population_count": len(population),
    }


class OnlyAstarMutateActor:
    """Persistent Ray actor for only-astar mutation work."""

    def __init__(
        self,
        *,
        q_ref: Any,
        environment_spec: ModuleSpec,
        diffusion_model_spec: ModuleSpec,
        metric_spec: ModuleSpec,
        actor_index: int = 0,
    ) -> None:
        self.actor_index = int(actor_index)
        self.q = _resolve_ray_value(q_ref)
        self.env = instantiate(environment_spec)
        self.diffusion = instantiate(diffusion_model_spec)
        self.metric = instantiate(metric_spec)

    def set_diffusion_state(self, state: dict[str, Any]) -> None:
        load_diffusion_state(self.diffusion, state)

    def mutate(self, task: dict[str, Any]) -> Individual:
        _maybe_seed(task.get("seed"))

        elite: Individual = task["elite"]
        k = int(task["iteration_k"])
        trunc = dict(task.get("truncated_diffusion") or {})
        add_noise_steps = int(trunc.get("add_noise_steps", 1))
        denoise_steps = int(trunc.get("denoise_steps", 1))

        timings: dict[str, float] = {}
        t_mut_total = time.perf_counter()

        t = time.perf_counter()
        mutated_rewards = step_5_3_1_truncated_diffusion_mutate_reward(
            self.diffusion,
            elite.tau,
            elite.rewards,
            add_noise_steps=add_noise_steps,
            denoise_steps=denoise_steps,
        )
        timings["5.3.1"] = time.perf_counter() - t

        t = time.perf_counter()
        _, tau_2 = step_5_3_2_simulate_collect(self.env, mutated_rewards)
        timings["5.3.2"] = time.perf_counter() - t

        t = time.perf_counter()
        rewards_2 = step_5_3_3_generate_reward(self.diffusion, tau_2)
        timings["5.3.3"] = time.perf_counter() - t

        t = time.perf_counter()
        simulation_data_2, rho_2 = step_5_3_4_simulate_and_compute_rho(
            self.env,
            rewards_2,
            q=self.q,
            metric=self.metric,
        )
        timings["5.3.4"] = time.perf_counter() - t

        mutant = step_4_build_only_astar_runtime_individual(
            tau=tau_2,
            rewards=rewards_2,
            rho=float(rho_2),
            simulation_data=simulation_data_2,
        )
        mutant.metadata["iteration_k"] = k
        mutant.metadata["parent_rho"] = float(elite.rho)
        mutant.metadata["actor_index"] = self.actor_index
        mutant.metadata["timing_mutation"] = {"total": time.perf_counter() - t_mut_total, **timings}
        return mutant

    def close(self) -> None:
        self.q = None
        self.env = None
        self.diffusion = None
        self.metric = None
        _maybe_cleanup_cuda_cache()


def mutate_one(task: dict[str, Any]) -> Individual:
    """
    Ray task：对一个精英个体做变异（对应 runner.py 中 mutate_one 的语义）。

    task 字段：
    - elite: Individual（注意：runner 已确保 elite.policies 为空，避免巨大传输）
    - iteration_k: 当前迭代 k
    - seed: 可选
    - q_ref: ray ObjectRef
    - diffusion_state_ref: ray ObjectRef（当前迭代训练后的扩散参数）
    - truncated_diffusion: dict(add_noise_steps, denoise_steps)
    - environment_spec/diffusion_model_spec/metric_spec: ModuleSpec
    """
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    elite: Individual = task["elite"]
    k = int(task["iteration_k"])
    trunc = dict(task.get("truncated_diffusion") or {})
    add_noise_steps = int(trunc.get("add_noise_steps", 1))
    denoise_steps = int(trunc.get("denoise_steps", 1))

    result: Individual | None = None
    try:
        q = _resolve_ray_value(task["q_ref"])
        diffusion_state = _resolve_ray_value(task["diffusion_state_ref"])

        env = instantiate(task["environment_spec"])
        diffusion = instantiate(task["diffusion_model_spec"])
        metric = instantiate(task["metric_spec"])
        load_diffusion_state(diffusion, diffusion_state)

        # =========================
        # 5.3 精英个体变异（对应 runner.py 的 mutate_one）
        # =========================
        # 5.3.1) Tau 条件下做“截断扩散”生成变异奖励 R'
        mutated_rewards = step_5_3_1_truncated_diffusion_mutate_reward(
            diffusion,
            elite.tau,
            elite.rewards,
            add_noise_steps=add_noise_steps,
            denoise_steps=denoise_steps,
        )

        # 5.3.2) 用变异奖励 R' + 纯 A* 仿真，重新收集经验库与 Tau
        _, tau_2 = step_5_3_2_simulate_collect(env, mutated_rewards)

        # 5.3.3) 把新 Tau 作为条件重新生成奖励 R
        rewards_2 = step_5_3_3_generate_reward(diffusion, tau_2)

        # 5.3.4) 用新奖励 R + 纯 A* 仿真并计算 ρ
        simulation_data_2, rho_2 = step_5_3_4_simulate_and_compute_rho(env, rewards_2, q=q, metric=metric)

        # 变异后的运行时对象同样不再携带 experience_buffers / policies。
        result = step_4_build_only_astar_runtime_individual(
            tau=tau_2,
            rewards=rewards_2,
            rho=float(rho_2),
            simulation_data=simulation_data_2,
        )
        return result
    finally:
        _maybe_cleanup_cuda_cache()

