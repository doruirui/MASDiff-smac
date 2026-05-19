import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
import numpy as np
import torch

from src.pipeline.types import Individual, Population
from src.evolution.temperature_selection import TemperatureEliteSelector
from src.evolution.evox_adapter import (
    build_evox_algorithm,
    build_evox_problem,
    get_evox_std_workflow_cls,
)
from src.parallel.ray_executor import RayExecutor
from src.utils.import_utils import import_symbol
from src.policy.starcraft_policy import StarcraftPolicy  # 导入神经网络策略


def run_pipeline(config: Dict[str, Any]) -> None:
    print("===== MASDiff-EVOX Pipeline Started =====")

    # 1. 基础配置
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ========== 动态计算策略网络参数量（关键新增部分）==========
    # 导入环境类并临时创建实例，以获取观测维度和动作数
    env_cls = import_symbol(config["environment"]["class_path"])
    env_kwargs = config["environment"].get("kwargs", {}).copy()
    temp_env = env_cls(**env_kwargs)

    # 强制初始化底层 SMAC 环境（不同环境类的初始化方式可能不同）
    if hasattr(temp_env, '_init_env'):
        temp_env._init_env()
    elif hasattr(temp_env, 'reset'):
        temp_env.reset()
    else:
        raise RuntimeError("环境类没有 _init_env 或 reset 方法，无法初始化")

    # 获取观测维度和动作数
    obs_dim = temp_env.env.get_obs_size()
    n_actions = temp_env.env.get_total_actions()
    hidden_dim = config.get("policy_hidden_dim", 64)  # 可在配置中设置，默认64

    # 创建临时策略计算参数量
    temp_policy = StarcraftPolicy(obs_dim, n_actions, hidden_dim=hidden_dim)
    param_dim = temp_policy.param_dim

    # 覆盖配置中的算法参数维度
    config["algorithm"]["N"] = param_dim
    # 如果 q_provider 需要 param_dim，也一并覆盖
    if "q_provider" in config and "kwargs" in config["q_provider"]:
        config["q_provider"]["kwargs"]["param_dim"] = param_dim

    print(f"自动计算得到策略网络参数量: {param_dim} (obs_dim={obs_dim}, n_actions={n_actions}, hidden_dim={hidden_dim})")

    # 释放临时环境（避免资源占用）
    try:
        temp_env.close()
    except AttributeError:
        pass
    del temp_env
    # ====================================================

    M = config["algorithm"]["M"]
    N = config["algorithm"]["N"]
    K = config["algorithm"].get("K", 1)
    pop_size = config["evox"]["algorithm"]["kwargs"]["pop_size"]

    # 2. 初始化种群（使用正确的参数维度 N）
    initial_population = []
    for i in range(pop_size):
        params = np.random.randn(N).astype(np.float32) * 0.1
        ind = Individual(params=params, rewards=np.zeros(1), rho=0.0)
        initial_population.append(ind)

    # 3. 构建环境、Q provider、metric
    env_cls = import_symbol(config["environment"]["class_path"])
    env_kwargs = config["environment"].get("kwargs", {})
    environment = env_cls(**env_kwargs)

    q_cls = import_symbol(config["q_provider"]["class_path"])
    q_kwargs = config["q_provider"].get("kwargs", {})
    q_provider = q_cls(**q_kwargs)

    metric_cls = import_symbol(config["metric"]["class_path"])
    metric_kwargs = config["metric"].get("kwargs", {})
    metric = metric_cls(**metric_kwargs)

    # 4. 将 evox 配置转换为 SimpleNamespace 供 adapter 使用
    evox_dict = config["evox"]
    evox_cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            class_path=evox_dict["algorithm"]["class_path"],
            kwargs=evox_dict["algorithm"].get("kwargs", {})
        ),
        reward_min=evox_dict["reward_min"],
        reward_max=evox_dict["reward_max"],
        repo_path=evox_dict.get("repo_path", ""),
        init_strategy=evox_dict.get("init_strategy", "best"),
        device=evox_dict.get("device", "cpu"),
        allow_shim_fallback=evox_dict.get("allow_shim_fallback", True)
    )

    # 5. 构建 EvoX 算法
    algorithm = build_evox_algorithm(evox_cfg, initial_population)

    # 6. 奖励形状（维度）
    reward_shape = (q_provider.num_actions, 1) if hasattr(q_provider, "num_actions") else (N, 1)

    # 7. 批量评估器
    def batch_evaluator(pop_tensor: torch.Tensor) -> Population:
        pop_np = pop_tensor.detach().cpu().numpy()
        new_population = []
        for i in range(pop_np.shape[0]):
            params = pop_np[i]
            ind = Individual(params=params, rewards=np.zeros(1), rho=0.0)
            evaluated = environment.evaluate(ind)
            # 确保 rho 已填充
            if evaluated.rho is None and evaluated.rewards is not None:
                evaluated.rho = metric(evaluated.rewards)
            new_population.append(evaluated)
        return new_population

    # 8. 构建 Problem
    problem = build_evox_problem(
        evox_cfg=evox_cfg,
        reward_shape=reward_shape,
        batch_evaluator=batch_evaluator,
    )

    # 9. 构建 Workflow
    workflow_cls = get_evox_std_workflow_cls(
        evox_cfg.repo_path,
        allow_shim_fallback=evox_cfg.allow_shim_fallback
    )
    workflow = workflow_cls(
        algorithm=algorithm,
        problem=problem,
        opt_direction="max",
        device=evox_cfg.device
    )

    # 10. 初始化并行执行器（Ray）
    executor_cls = import_symbol(config["parallel_executor"]["class_path"])
    executor_kwargs = config["parallel_executor"].get("kwargs", {})
    executor = executor_cls(**executor_kwargs)

    # 11. 进化开始
    workflow.init_step()
    best_rho_history = []
    best_individual = None

    for iteration in range(M):
        print(f"\n--- Iteration {iteration+1}/{M} ---")
        workflow.step()
        latest_population = getattr(problem, "last_population", [])
        if latest_population:
            current_best = max(latest_population, key=lambda ind: ind.rho)
            if best_individual is None or current_best.rho > best_individual.rho:
                best_individual = current_best
                print(f"New best rho: {best_individual.rho}")
            best_rho_history.append(best_individual.rho)
        else:
            best_rho_history.append(best_individual.rho if best_individual else 0.0)

    # 12. 保存结果
    import csv
    csv_path = config["logging"].get("best_rho_csv_path", "outputs/best_rho.csv")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_rho"])
        for i, rho in enumerate(best_rho_history):
            writer.writerow([i+1, rho])

    if best_individual is not None:
        np.save("outputs/best_params.npy", best_individual.params)
        print(f"\nOptimization finished. Best rho: {best_individual.rho}")
    else:
        print("No valid individual found.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = _project_root / "configs" / "starcraft_smac_es.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_pipeline(config)