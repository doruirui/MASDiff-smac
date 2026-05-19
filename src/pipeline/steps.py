from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from src.dqn.base import DqnModule
from src.diffusion.base import DiffusionModel
from src.environments.base import Environment
from src.evolution.base import EliteSelector
from src.metrics.base import Metric
from src.pipeline.types import Individual, Population
from src.q.base import QProvider


_INITIAL_POPULATION_CACHE_VERSION = 2


@dataclass
class _CachedInitialIndividual:
    tau: Any
    rewards: Any
    rho: float
    metadata: dict[str, Any]


def _strip_initial_population_metadata(
    population: Population,
) -> list[dict[str, Any]]:
    """
    Keep only fields that are actually consumed by the only-astar evolution path.

    Required later:
    - tau / rewards / rho
    - simulation_data only for the current best individual, because
      step_record_best_simulation_data_if_improved() reads it immediately after load.

    Explicitly dropped from cache:
    - experience_buffers
    - policies
    - non-essential metadata such as initial_index / parent_rho / iteration_k
    """
    if len(population) == 0:
        return []

    best_index = max(range(len(population)), key=lambda idx: float(population[idx].rho))
    cached: list[dict[str, Any]] = []
    for idx, ind in enumerate(population):
        metadata: dict[str, Any] = {}
        if idx == best_index:
            simulation_data = getattr(ind, "metadata", {}).get("simulation_data")
            if simulation_data is not None:
                metadata["simulation_data"] = simulation_data
        cached.append(
            {
                "tau": ind.tau,
                "rewards": ind.rewards,
                "rho": float(ind.rho),
                "metadata": metadata,
            }
        )
    return cached


def _restore_initial_population(payload: list[dict[str, Any]]) -> Population:
    population: Population = []
    for item in payload:
        population.append(
            Individual(
                tau=item["tau"],
                rewards=item["rewards"],
                rho=float(item["rho"]),
                experience_buffers=[],
                policies=[],
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return population


# =========================
# 1. 读取或创建 Q
# =========================
def step_1_load_or_create_q(q_provider: QProvider, environment: Environment) -> Any:
    """
    主流程第 1 步：
    1.读取或创建 Q（这是目标，如果没有就创建，创建函数用户自定义）
    """
    q = q_provider.load_or_create_q(environment)
    return q


# =========================
# 2. 随机初始化扩散模型
# =========================
def step_2_init_diffusion_model(diffusion: DiffusionModel) -> DiffusionModel:
    """
    主流程第 2 步：
    2.随机初始化扩散模型
    """
    diffusion.init_random()
    return diffusion


# =========================
# 3. 随机初始化策略（DQN）
# =========================
def step_3_init_random_policies(dqn: DqnModule, *, num_agents: int) -> list[Any]:
    """
    主流程第 3 步：
    3.随机初始化策略（dqn），数量：M*N（M是种群规模，N是智能体数量）
    """
    policies = dqn.init_random_policies(num_agents)
    return policies


def step_3_init_astar_inputs(*, num_agents: int) -> list[Any]:
    """
    当前主流程第 3 步：
    3. 初始化纯 A* 仿真的占位输入。

    说明：
    - 主流程已移除 DQN 训练/调用；
    - 对 SUMO 示例环境而言，传入 [None] * N 会触发“纯 A*”路径规划分支；
    - 后续评估阶段会直接把奖励 R 传入环境，由环境执行“R + 纯 A*”仿真。
    """
    return [None] * int(num_agents)


def step_3_init_random_rewards(
    *,
    q: Any,
    num_agents: int,
    reward_min: float = 0.0,
    reward_max: float = 5.0,
) -> Any:
    """
    当前主流程第 3 步：
    3. 随机初始化初始奖励矩阵 R，并在首次仿真时走 “R + A*” 分支。

    说明：
    - 奖励矩阵形状为 [num_agents, num_road]；
    - num_road 直接从 q 的最后一维推断，保证与评估数据维度一致；
    - 默认随机范围为 [0, 5]。
    """
    num_car = int(num_agents)
    if num_car <= 0:
        raise ValueError("num_agents 必须 > 0")

    q_shape = getattr(q, "shape", None)
    num_road: int | None = None
    if q_shape is not None and len(q_shape) >= 1:
        num_road = int(q_shape[-1])
    elif isinstance(q, (list, tuple)) and len(q) > 0:
        last = q[-1]
        if isinstance(last, (list, tuple)) and len(last) > 0:
            num_road = int(len(last))
        else:
            num_road = int(len(q))

    if num_road is None:
        raise ValueError("无法从 q 推断 num_road：q 缺少有效 shape")

    if num_road <= 0:
        raise ValueError("无法从 q 推断 num_road：最后一维必须 > 0")

    lo = float(reward_min)
    hi = float(reward_max)
    if hi < lo:
        raise ValueError("reward_max 必须 >= reward_min")

    try:
        import torch  # type: ignore

        return torch.rand((num_car, num_road), dtype=torch.float32) * (hi - lo) + lo
    except Exception:
        import random

        return [[random.uniform(lo, hi) for _ in range(num_road)] for _ in range(num_car)]


def step_3_4_load_initial_population(population_path: str) -> Population:
    """
    从缓存文件读取初始种群。
    """
    p = Path(population_path)
    with p.open("rb") as f:
        population = pickle.load(f)

    if isinstance(population, dict) and int(population.get("cache_version", -1)) == _INITIAL_POPULATION_CACHE_VERSION:
        cached_population = population.get("population")
        if not isinstance(cached_population, list):
            raise TypeError(f"初始种群文件内容无效: {p}")
        return _restore_initial_population(cached_population)

    if not isinstance(population, list) or not all(isinstance(ind, Individual) for ind in population):
        raise TypeError(f"初始种群文件内容无效: {p}")

    return population


def step_3_4_try_load_initial_population(population_path: str | None) -> Population | None:
    """
    若配置了初始种群缓存文件且文件存在，则直接读取；否则返回 None。
    """
    if not population_path:
        return None

    p = Path(population_path)
    if not p.exists():
        return None

    return step_3_4_load_initial_population(population_path)


def step_3_4_save_initial_population(population: Population, *, population_path: str) -> None:
    """
    将初始种群保存到缓存文件。
    """
    p = Path(population_path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("wb") as f:
        pickle.dump(
            {
                "cache_version": _INITIAL_POPULATION_CACHE_VERSION,
                "population": _strip_initial_population_metadata(population),
            },
            f,
        )


def _to_jsonable(value: Any) -> Any:
    """
    将常见对象递归转换为可 JSON 序列化的结构。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=repr)]

    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    return repr(value)


# =========================
# 4. 形成初始种群：单个个体的构建步骤
# =========================
def step_4_1_simulate_collect(env: Environment, planner_input: Any) -> tuple[list[list[Any]], Any]:
    """
    主流程第 4.1 步：
    4.1 用给定规划输入仿真一次，收集每个智能体的经验库（奖励留空）和 Tau
    """
    experience_buffers, tau = env.simulate_collect(planner_input)
    return experience_buffers, tau


def step_4_2_generate_reward(diffusion: DiffusionModel, tau: Any) -> Any:
    """
    主流程第 4.2 步：
    4.2 把 Tau 作为扩散模型的条件生成奖励 R
    """
    rewards = diffusion.generate_reward(tau)
    return rewards


def step_4_3_simulate_and_compute_rho(
    env: Environment,
    rewards: Any,
    *,
    q: Any,
    metric: Metric,
) -> tuple[Any, float]:
    """
    当前主流程第 4.3 步：
    4.3 直接使用奖励 R + 纯 A* 进行仿真，并计算 rho。
    """
    simulation_data = env.simulate_evaluate(rewards)
    _append_simulation_data_to_csv(simulation_data)
    print(f"仿真数据: {simulation_data}")
    print(f"Q 数据: {q}")
    rho = metric.compute_rho(q, simulation_data)
    return simulation_data, float(rho)


def step_4_3_build_dqn_training_data(dqn: DqnModule, experience_buffers: list[list[Any]], rewards: Any) -> Any:
    """
    主流程第 4.3 步：
    4.3 将经验库和 R 合并形成 dqn 训练数据（用奖励 R 填充经验中的 r）
    """
    training_data = dqn.build_training_data(experience_buffers, rewards)
    return training_data


def step_4_4_train_dqn_per_agent(dqn: DqnModule, training_data: Any) -> list[Any]:
    """
    主流程第 4.4 步：
    4.4 为每个智能体训练一个 dqn
    """
    trained_policies = dqn.train_per_agent(training_data)
    return trained_policies


def _append_simulation_data_to_csv(simulation_data: Any, csv_path: str = "outputs/simulation_data.csv") -> None:
    """
    将每次仿真得到的 simulation_data 追加写入 CSV。

    说明：
    - 使用 JSON 序列化保存任意结构数据，避免丢失嵌套信息。
    - 默认输出到 outputs/simulation_data.csv。
    """
    p = Path(csv_path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not p.exists()) or (p.stat().st_size == 0)
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["simulation_data"])
        writer.writerow([json.dumps(_to_jsonable(simulation_data), ensure_ascii=False)])


def _overwrite_simulation_data_csv(simulation_data: Any, csv_path: str) -> None:
    """
    覆盖写入单个 simulation_data 到 CSV 文件。

    输出格式与 outputs/simulation_data.csv 保持一致：
    - 仅包含一列表头 simulation_data
    - 第二行写入一条 JSON 字符串
    """
    p = Path(csv_path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation_data"])
        writer.writerow([json.dumps(_to_jsonable(simulation_data), ensure_ascii=False)])


def step_4_5_simulate_and_compute_rho(
    env: Environment,
    policies: list[Any],
    *,
    q: Any,
    metric: Metric,
) -> tuple[Any, float]:
    """
    主流程第 4.5 步：
    4.5 再用训练好的 dqn 仿真一次，得到仿真数据，计算 ρ（越大越好）
    """
    simulation_data = env.simulate_evaluate(policies)
    _append_simulation_data_to_csv(simulation_data)
    print(f"仿真数据: {simulation_data}")
    print(f"Q 数据: {q}")
    rho = metric.compute_rho(q, simulation_data)
    return simulation_data, float(rho)


def step_4_build_individual(
    *,
    tau: Any,
    rewards: Any,
    rho: float,
    experience_buffers: list[list[Any]],
    policies: list[Any],
) -> Individual:
    """
    汇总第 4 步结果形成一个个体：
    (Tau, R, ρ) + 保存经验库与 DQN（策略）。
    """
    return Individual(
        tau=tau,
        rewards=rewards,
        rho=rho,
        experience_buffers=experience_buffers,
        policies=policies,
        metadata={},
    )


def step_4_build_only_astar_runtime_individual(
    *,
    tau: Any,
    rewards: Any,
    rho: float,
    simulation_data: Any | None = None,
) -> Individual:
    """
    Build the lightweight runtime individual used by the only-astar path.

    This path only needs:
    - tau / rewards / rho for evolution and export
    - simulation_data for best-simulation CSV tracking

    It intentionally drops:
    - experience_buffers
    - policies
    - non-essential metadata
    """
    metadata: dict[str, Any] = {}
    if simulation_data is not None:
        metadata["simulation_data"] = simulation_data
    return Individual(
        tau=tau,
        rewards=rewards,
        rho=rho,
        experience_buffers=[],
        policies=[],
        metadata=metadata,
    )


def step_4_evaluate_only_astar_reward(
    env: Environment,
    rewards: Any,
    *,
    q: Any,
    metric: Metric,
) -> Individual:
    """
    使用给定奖励矩阵 R 评估一个 only-astar 运行时个体。

    流程：
    - 先跑一次 simulate_collect(...) 获取 Tau
    - 再跑 simulate_evaluate(...) 获取 simulation_data 与 rho
    - 返回轻量运行时个体
    """
    _, tau = step_4_1_simulate_collect(env, rewards)
    simulation_data, rho = step_4_3_simulate_and_compute_rho(env, rewards, q=q, metric=metric)
    return step_4_build_only_astar_runtime_individual(
        tau=tau,
        rewards=rewards,
        rho=float(rho),
        simulation_data=simulation_data,
    )


# =========================
# 5. 进化迭代：K 次
# =========================
def step_5_1_train_diffusion_with_population(diffusion: DiffusionModel, population: Population) -> DiffusionModel:
    """
    主流程第 5.1 步：
    5.1 用当前的种群训练扩散模型
    """
    diffusion.train_on_population(population)
    return diffusion


def step_5_2_select_elite_population(selector: EliteSelector, population: Population, *, elite_count: int) -> Population:
    """
    主流程第 5.2 步：
    5.2 根据种群个体的 ρ 选择精英种群
    """
    elites = selector.select_elites(population, elite_count=elite_count)
    return list(elites)


def step_5_3_1_truncated_diffusion_mutate_reward(
    diffusion: DiffusionModel,
    tau: Any,
    base_rewards: Any,
    *,
    add_noise_steps: int,
    denoise_steps: int,
) -> Any:
    """
    主流程第 5.3.1 步：
    5.3.1 把 Tau 作为条件，然后使用截断扩散生成 R（奖励变异）

    说明：
    - base_rewards 来自精英个体原有的 R（用于“围绕精英 R 做变异”）
    """
    mutated_rewards = diffusion.generate_reward_truncated(
        tau,
        base_rewards,
        add_noise_steps=add_noise_steps,
        denoise_steps=denoise_steps,
    )
    return mutated_rewards


def step_5_3_2_simulate_collect(env: Environment, rewards: Any) -> tuple[list[list[Any]], Any]:
    """
    当前主流程第 5.3.2 步：
    5.3.2 使用变异后的奖励 R' + 纯 A* 仿真，收集新的经验库与 Tau。
    """
    return env.simulate_collect(rewards)


def step_5_3_3_generate_reward(diffusion: DiffusionModel, tau: Any) -> Any:
    """
    当前主流程第 5.3.3 步：
    5.3.3 根据新的 Tau 再生成奖励 R。
    """
    return step_4_2_generate_reward(diffusion, tau)


def step_5_3_4_simulate_and_compute_rho(
    env: Environment,
    rewards: Any,
    *,
    q: Any,
    metric: Metric,
) -> tuple[Any, float]:
    """
    当前主流程第 5.3.4 步：
    5.3.4 使用新的奖励 R + 纯 A* 仿真，并计算 rho。
    """
    return step_4_3_simulate_and_compute_rho(env, rewards, q=q, metric=metric)


def step_5_3_2_build_dqn_training_data(dqn: DqnModule, experience_buffers: list[list[Any]], rewards: Any) -> Any:
    """
    主流程第 5.3.2 步：
    5.3.2 将经验库和 R 合并形成 dqn 训练数据（用奖励 R 填充经验中的 r）
    """
    return step_4_3_build_dqn_training_data(dqn, experience_buffers, rewards)


def step_5_3_3_train_dqn_per_agent(dqn: DqnModule, training_data: Any) -> list[Any]:
    """
    主流程第 5.3.3 步：
    5.3.3 为每个智能体训练一个 dqn
    """
    return step_4_4_train_dqn_per_agent(dqn, training_data)


def step_5_3_4_simulate_collect(env: Environment, policies: list[Any]) -> tuple[list[list[Any]], Any]:
    """
    主流程第 5.3.4 步：
    5.3.4 用训练的 dqn 仿真，收集经验库（奖励留空）和 Tau
    """
    return step_4_1_simulate_collect(env, policies)


def step_5_3_5_generate_reward(diffusion: DiffusionModel, tau: Any) -> Any:
    """
    主流程第 5.3.5 步：
    5.3.5 把 Tau 作为扩散模型的条件生成奖励 R
    """
    return step_4_2_generate_reward(diffusion, tau)


def step_5_3_6_build_dqn_training_data(dqn: DqnModule, experience_buffers: list[list[Any]], rewards: Any) -> Any:
    """
    主流程第 5.3.6 步：
    5.3.6 将经验库和 R 合并形成 dqn 训练数据（用奖励 R 填充经验中的 r）
    """
    return step_4_3_build_dqn_training_data(dqn, experience_buffers, rewards)


def step_5_3_7_train_dqn_per_agent(dqn: DqnModule, training_data: Any) -> list[Any]:
    """
    主流程第 5.3.7 步：
    5.3.7 为每个智能体训练一个 dqn
    """
    return step_4_4_train_dqn_per_agent(dqn, training_data)


def step_5_3_8_simulate_and_compute_rho(
    env: Environment,
    policies: list[Any],
    *,
    q: Any,
    metric: Metric,
) -> tuple[Any, float]:
    """
    主流程第 5.3.8 步：
    5.3.8 再用训练好的 dqn 仿真一次，得到仿真数据，计算 ρ
    """
    return step_4_5_simulate_and_compute_rho(env, policies, q=q, metric=metric)


def step_5_4_add_mutants_to_population(population: Population, mutants: Population) -> Population:
    """
    主流程第 5.4 步：
    5.4 将变异后的精英种群加入原有种群
    """
    return list(population) + list(mutants)


def step_5_5_keep_top_m(population: Population, *, M: int) -> Population:
    """
    主流程第 5.5 步：
    5.5 保留 ρ 最大的 M 个个体，形成新种群
    """
    # ρ 越大越好
    sorted_pop = sorted(population, key=lambda ind: float(ind.rho), reverse=True)
    return sorted_pop[:M]


def step_5_6_record_best_rho(population: Population, *, iteration_k: int, csv_path: str) -> tuple[Population, float]:
    """
    主流程第 5.6 步：
    5.6 记录种群最好的 ρ 值，并把每次进化迭代种群最好的 ρ 值追加写入 CSV 文件。

    返回值：
    - population: 原样返回（用于继续主流程的 population 变量）
    - best_rho: 本次迭代新种群中的最优 ρ（越大越好），用于写入 CSV，也可用于外部监控
    """
    if len(population) == 0:
        raise ValueError("population 不能为空，无法记录 best_rho")

    best_rho = max(float(ind.rho) for ind in population)

    p = Path(csv_path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not p.exists()) or (p.stat().st_size == 0)
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["iteration_k", "best_rho"])
        writer.writerow([iteration_k, best_rho])

    return population, best_rho


def step_record_best_simulation_data_if_improved(
    population: Population,
    *,
    csv_path: str,
    best_rho_so_far: float | None = None,
) -> tuple[Population, float, bool]:
    """
    将“当前全局最优个体”的 simulation_data 覆盖写入额外的 CSV 文件。

    规则：
    - 若当前种群中的最优个体优于 best_rho_so_far，则覆盖写入 csv_path；
    - 若没有更优个体出现，则不改动该文件；
    - 返回更新后的全局最优 rho，以及本次是否发生覆盖写入。
    """
    if len(population) == 0:
        raise ValueError("population 不能为空，无法记录最优 simulation_data")

    best_individual = max(population, key=lambda ind: float(ind.rho))
    best_rho = float(best_individual.rho)

    if best_rho_so_far is not None and best_rho <= float(best_rho_so_far):
        return population, float(best_rho_so_far), False

    simulation_data = getattr(best_individual, "metadata", {}).get("simulation_data")
    if simulation_data is None:
        raise ValueError("最优个体缺少 metadata['simulation_data']，无法写入最优 simulation_data CSV")

    _overwrite_simulation_data_csv(simulation_data, csv_path)
    return population, best_rho, True


def step_6_simulate_best_and_export_routes(
    env: Environment,
    population: Population,
    *,
    output_rou_path: str,
) -> tuple[Population, str | None]:
    """
    主流程第 6 步：
    - 从最终种群中选取 ρ 最高的个体
    - 直接使用最优个体对应的奖励 R + 纯 A* 再仿真一次
    - 记录所有车辆完成路径规划后的路径，并导出一个新的 rou 文件到 outputs

    兼容性说明：
    - 该步骤依赖环境对象提供 `simulate_evaluate_with_route_export(...)`
    - 若当前环境未实现该方法，则返回 `(population, None)`，由 runner 决定如何提示
    """
    if len(population) == 0:
        raise ValueError("population 不能为空，无法执行第 6 步")

    export_fn = getattr(env, "simulate_evaluate_with_route_export", None)
    if not callable(export_fn):
        return population, None

    best_individual = max(population, key=lambda ind: float(ind.rho))
    output_path = str(export_fn(best_individual.rewards, output_rou_path=output_rou_path))
    best_individual.metadata["step_6_exported_rou_path"] = output_path
    return population, output_path


def step_clone_elite_for_mutation(elite: Individual) -> Individual:
    """
    可选辅助：克隆精英个体（便于实现“从精英个体出发做变异”的语义）。
    这里做浅拷贝；具体深拷贝策略由用户数据结构决定。
    """
    return replace(elite)
