from __future__ import annotations

import csv
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config.schema import MasDiffConfig
from src.diffusion.base import DiffusionModel
from src.environments.base import Environment
from src.evolution.evox_adapter import (
    build_evox_algorithm,
    build_evox_problem,
    get_evox_std_workflow_cls,
    import_torch_and_evox,
)
from src.metrics.base import Metric
from src.parallel.base import ParallelExecutor
try:
    # 可选：只有在配置选择 RayExecutor 时才会用到
    from src.parallel.ray_executor import RayExecutor  # type: ignore
except Exception:  # pragma: no cover
    RayExecutor = None  # type: ignore
from src.pipeline.steps import (
    step_1_load_or_create_q,
    step_2_init_diffusion_model,
    step_3_4_try_load_initial_population,
    step_3_4_save_initial_population,
    step_5_6_record_best_rho,
    step_record_best_simulation_data_if_improved,
    step_6_simulate_best_and_export_routes,
)
from src.pipeline.types import Population
from src.q.base import QProvider
from src.utils.import_utils import instantiate


def _maybe_cleanup_cuda_cache() -> None:
    """Best-effort cleanup of unused Python and CUDA cached memory."""
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


def _callable_name(fn: Any) -> str:
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return name
    inner = getattr(fn, "func", None)
    inner_name = getattr(inner, "__name__", None)
    if isinstance(inner_name, str) and inner_name:
        return inner_name
    return type(fn).__name__


def _save_initial_population_in_child(population: Population, population_path: str) -> None:
    """Background child process entrypoint for initial population cache writes."""
    step_3_4_save_initial_population(population, population_path=population_path)


def _start_async_initial_population_save(
    population: Population,
    *,
    population_path: str,
) -> multiprocessing.process.BaseProcess | None:
    """
    Start a background cache write on Linux/Unix using ``fork``.

    Returns ``None`` when async save is unavailable and callers should fall back
    to the existing synchronous save path.
    """
    if os.name != "posix":
        return None

    try:
        start_methods = multiprocessing.get_all_start_methods()
    except Exception:
        return None
    if "fork" not in start_methods:
        return None

    try:
        ctx = multiprocessing.get_context("fork")
        process = ctx.Process(
            target=_save_initial_population_in_child,
            args=(population, population_path),
            daemon=False,
        )
        process.start()
    except Exception:
        return None
    return process


def _submit_ray_refs(
    executor: Any,
    fn: Any,
    items: list[Any],
) -> list[Any]:
    ray = executor._ray  # type: ignore[attr-defined]
    options = executor._options_for(fn)  # type: ignore[attr-defined]
    remote_fn = ray.remote(fn).options(**options) if options else ray.remote(fn)
    return [remote_fn.remote(x) for x in items]


def _wait_ray_progress(executor: Any, fn: Any, refs: list[Any]) -> None:
    if not getattr(executor, "show_progress", False):
        return
    total = len(refs)
    if total == 0:
        return

    ray = executor._ray  # type: ignore[attr-defined]
    max_updates = max(1, int(getattr(executor, "max_updates", 20)))
    step = max(1, total // max_updates)
    print(f"{executor.desc}({ _callable_name(fn) }): 0/{total}")

    pending = list(refs)
    done = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        done += len(ready)
        if done == total or (done % step == 0):
            print(f"{executor.desc}({ _callable_name(fn) }): {done}/{total}")


def _wait_ray_progress_named(executor: Any, name: str, refs: list[Any]) -> None:
    if not getattr(executor, "show_progress", False):
        return
    total = len(refs)
    if total == 0:
        return

    ray = executor._ray  # type: ignore[attr-defined]
    max_updates = max(1, int(getattr(executor, "max_updates", 20)))
    step = max(1, total // max_updates)
    print(f"{executor.desc}({name}): 0/{total}")

    pending = list(refs)
    done = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        done += len(ready)
        if done == total or (done % step == 0):
            print(f"{executor.desc}({name}): {done}/{total}")


def _fetch_population_and_optionally_cache(
    refs: list[Any],
    *,
    population_path: str | None,
) -> Population:
    import ray  # type: ignore

    population = list(ray.get(refs))
    if population_path:
        step_3_4_save_initial_population(population, population_path=population_path)
    return population


def _remote_train_options(executor: Any, *, diffusion_device: str) -> dict[str, Any]:
    opts = dict(getattr(executor, "_default_task_options", {}))
    by_name = getattr(executor, "_task_options_by_name", {}).get("train_diffusion_on_population")
    if by_name:
        opts.update(by_name)

    device = str(diffusion_device).lower()
    if device.startswith("cuda") and "num_gpus" not in opts:
        opts["num_gpus"] = 1
    if "num_cpus" not in opts:
        opts["num_cpus"] = 1
    return opts


def _mutate_actor_options(executor: Any, *, diffusion_device: str) -> dict[str, Any]:
    opts = dict(executor.options_for_name("mutate_one"))
    if "num_cpus" not in opts:
        opts["num_cpus"] = 1

    device = str(diffusion_device).lower()
    if device.startswith("cuda"):
        if "num_gpus" not in opts:
            opts["num_gpus"] = 1
    else:
        opts.pop("num_gpus", None)
    return opts


def _build_mutate_actors(
    executor: Any,
    *,
    q: Any,
    environment_spec: Any,
    diffusion_model_spec: Any,
    metric_spec: Any,
    elite_count: int,
    diffusion_device: str,
) -> list[Any]:
    device = str(diffusion_device).lower()
    if not device.startswith("cuda"):
        return []

    total_gpu = float(executor.cluster_gpu_count())
    actor_options = _mutate_actor_options(executor, diffusion_device=diffusion_device)
    gpu_per_actor = float(actor_options.get("num_gpus", 1.0))
    if gpu_per_actor <= 0:
        gpu_per_actor = 1.0

    actor_count = min(max(1, int(total_gpu / gpu_per_actor)), max(0, int(elite_count)))
    if actor_count <= 0:
        return []

    from src.parallel.ray_tasks import OnlyAstarMutateActor

    q_ref = _strict_ray_put(executor, q, label="q")
    ray = executor._ray  # type: ignore[attr-defined]
    remote_actor = ray.remote(OnlyAstarMutateActor)
    actors: list[Any] = []
    for actor_index in range(actor_count):
        actor = remote_actor.options(**actor_options).remote(
            q_ref=q_ref,
            environment_spec=environment_spec,
            diffusion_model_spec=diffusion_model_spec,
            metric_spec=metric_spec,
            actor_index=actor_index,
        )
        actors.append(actor)
    return actors


def _shutdown_mutate_actors(executor: Any, actors: list[Any]) -> None:
    if not actors:
        return

    ray = executor._ray  # type: ignore[attr-defined]
    close_refs: list[Any] = []
    for actor in actors:
        try:
            close_refs.append(actor.close.remote())
        except Exception:
            continue
    if close_refs:
        try:
            ray.get(close_refs)
        except Exception:
            pass
    for actor in actors:
        try:
            ray.kill(actor)
        except Exception:
            continue


def _safe_ray_put(executor: ParallelExecutor, value: Any, *, label: str) -> Any:
    if (RayExecutor is not None) and isinstance(executor, RayExecutor):
        safe_put = getattr(executor, "safe_put", None)
        if callable(safe_put):
            try:
                return safe_put(value, label=label)
            except Exception as e:
                # q 是高复用对象，put 失败通常意味着 Ray 已不可用；其余临时批次降级为按值传递。
                if str(label) == "q":
                    raise RuntimeError(f"Ray put 失败（label={label}），无法继续提交并行任务。") from e
                print(f"[WARN] Ray put 失败（label={label}），降级为按值传递。")
                return value
    return value


def _require_ray_executor(executor: ParallelExecutor) -> "RayExecutor":
    if RayExecutor is None:
        raise ImportError("RayExecutor is unavailable; pure Ray mode cannot continue.")
    if not isinstance(executor, RayExecutor):
        raise TypeError(
            f"Pure Ray mode requires `src.parallel.ray_executor:RayExecutor`, got {type(executor).__name__}."
        )
    return executor


def _strict_ray_put(executor: ParallelExecutor, value: Any, *, label: str) -> Any:
    ray_executor = _require_ray_executor(executor)
    try:
        return ray_executor.safe_put(value, label=label)
    except Exception as e:
        if str(label) == "q":
            raise RuntimeError(f"Ray put failed for label={label}, cannot continue.") from e
        print(f"[WARN] Ray put failed for label={label}, falling back to by-value transfer.")
        return value


def _ensure_instance(obj: Any, expected_type: type, *, name: str) -> None:
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"模块 `{name}` 需要是 {expected_type.__name__} 的实例（或其子类），"
            f"但得到 {type(obj).__name__}"
        )


def _infer_reward_shape(*, population: Population | None, q: Any, num_agents: int) -> tuple[int, int]:
    if population:
        rewards = getattr(population[0], "rewards", None)
        shape = getattr(rewards, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
        if isinstance(rewards, (list, tuple)) and len(rewards) > 0:
            first_row = rewards[0]
            return int(len(rewards)), int(len(first_row))

    q_shape = getattr(q, "shape", None)
    if q_shape is not None and len(q_shape) >= 1:
        return int(num_agents), int(q_shape[-1])

    if isinstance(q, (list, tuple)) and len(q) > 0:
        last = q[-1]
        if isinstance(last, (list, tuple)) and len(last) > 0:
            return int(num_agents), int(len(last))
        return int(num_agents), int(len(q))

    raise ValueError("无法推断 reward 形状。")


def _evaluate_rewards_batch(
    *,
    rewards_batch: list[Any],
    q: Any,
    environment: Environment,
    metric: Metric,
    executor: ParallelExecutor,
    environment_spec: Any,
    metric_spec: Any,
) -> Population:
    ray_executor = _require_ray_executor(executor)
    q_ref = _strict_ray_put(ray_executor, q, label="q")
    batch_size_env = os.environ.get("MASDIFF_RAY_EVAL_BATCH_SIZE", "2")
    try:
        batch_size = max(1, int(batch_size_env))
    except Exception:
        batch_size = 2

    batched_tasks: list[dict[str, Any]] = []
    for start in range(0, len(rewards_batch), batch_size):
        rewards_chunk = rewards_batch[start : start + batch_size]
        rewards_batch_ref = _strict_ray_put(
            ray_executor,
            rewards_chunk,
            label=f"rewards_batch_{start}",
        )
        batched_tasks.append(
            {
                "rewards_batch_ref": rewards_batch_ref,
                "q_ref": q_ref,
                "environment_spec": environment_spec,
                "metric_spec": metric_spec,
            }
        )

    from src.parallel.ray_tasks import evaluate_reward_batch as ray_evaluate_reward_batch

    batched_population = ray_executor.map(ray_evaluate_reward_batch, batched_tasks)
    flattened: Population = []
    for group in batched_population:
        flattened.extend(list(group))
    return flattened


def run_masdiff(cfg: MasDiffConfig) -> Population:
    """
    执行 MASDiff 主流程。

    说明：
    - 框架只调度主流程；所有具体实现由 cfg 中的 class_path 指向的用户模块提供。
    - 主流程每一步都封装为一个函数，runner 负责串联返回值。
    """

    # ---------- 计时统计（只修改 runner.py，不改其他文件） ----------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_rows: list[dict[str, Any]] = []

    def _add_timing_row(
        *,
        scope: str,
        part: str,
        name: str,
        duration_s: float,
        iteration_k: Optional[int] = None,
        count: Optional[int] = None,
    ) -> None:
        timing_rows.append(
            {
                "run_id": run_id,
                "scope": scope,  # global / iter / iter_mutate_mean / iter_mutate_sum
                "iteration_k": iteration_k,
                "part": part,
                "name": name,
                "duration_s": duration_s,
                # percent 在结束时统一补齐（需要总时长）
                "percent": None,
                "count": count,
            }
        )

    def _finalize_percents() -> None:
        # global: 相对 run_total
        run_total = None
        for r in timing_rows:
            if r.get("scope") == "global" and r.get("part") == "run_total":
                run_total = r.get("duration_s")
                break
        if isinstance(run_total, (int, float)) and run_total > 0:
            for r in timing_rows:
                if r.get("scope") == "global" and r.get("part") != "run_total":
                    r["percent"] = (float(r["duration_s"]) / float(run_total)) * 100.0

        # iter: 相对 5.iter_total（按 iteration_k 分组）
        iter_total_by_k: dict[int, float] = {}
        for r in timing_rows:
            if r.get("scope") == "iter" and r.get("part") == "5.iter_total" and isinstance(r.get("iteration_k"), int):
                iter_total_by_k[int(r["iteration_k"])] = float(r["duration_s"])
        for r in timing_rows:
            if r.get("scope") == "iter" and r.get("part") != "5.iter_total" and isinstance(r.get("iteration_k"), int):
                denom = iter_total_by_k.get(int(r["iteration_k"]))
                if denom and denom > 0:
                    r["percent"] = (float(r["duration_s"]) / denom) * 100.0

        # mutate 聚合：相对 mutate_total_mean / mutate_total_sum（按 iteration_k 分组）
        mutate_total_mean_by_k: dict[int, float] = {}
        mutate_total_sum_by_k: dict[int, float] = {}
        for r in timing_rows:
            if not isinstance(r.get("iteration_k"), int):
                continue
            k = int(r["iteration_k"])
            if r.get("scope") == "iter_mutate_mean" and r.get("part") == "5.3.mutate_total_mean":
                mutate_total_mean_by_k[k] = float(r["duration_s"])
            if r.get("scope") == "iter_mutate_sum" and r.get("part") == "5.3.mutate_total_sum":
                mutate_total_sum_by_k[k] = float(r["duration_s"])
        for r in timing_rows:
            if not isinstance(r.get("iteration_k"), int):
                continue
            k = int(r["iteration_k"])
            if r.get("scope") == "iter_mutate_mean" and r.get("part") != "5.3.mutate_total_mean":
                denom = mutate_total_mean_by_k.get(k)
                if denom and denom > 0:
                    r["percent"] = (float(r["duration_s"]) / denom) * 100.0
            if r.get("scope") == "iter_mutate_sum" and r.get("part") != "5.3.mutate_total_sum":
                denom = mutate_total_sum_by_k.get(k)
                if denom and denom > 0:
                    r["percent"] = (float(r["duration_s"]) / denom) * 100.0

    def _write_timing_csv() -> None:
        if not timing_rows:
            return
        try:
            best_rho_path = Path(cfg.logging.best_rho_csv_path)
            timing_csv_path = best_rho_path.with_name(f"{best_rho_path.stem}_timings.csv")
            if timing_csv_path.parent:
                os.makedirs(timing_csv_path.parent, exist_ok=True)

            fieldnames = ["run_id", "scope", "iteration_k", "part", "name", "duration_s", "percent", "count"]
            write_header = not timing_csv_path.exists() or timing_csv_path.stat().st_size == 0
            with open(timing_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                for r in timing_rows:
                    w.writerow(r)
            print(f"计时统计 CSV 已写入：{timing_csv_path}")
        except Exception as e:
            # 计时落盘失败不应影响主流程
            print(f"[WARN] 写入计时 CSV 失败：{e}")

    # ---------- 按配置加载用户自定义模块 ----------
    print("0. 加载配置并实例化各模块")
    _t_run_start = time.perf_counter()
    _t0 = time.perf_counter()
    q_provider = instantiate(cfg.q_provider)
    environment = instantiate(cfg.environment)
    diffusion_model = instantiate(cfg.diffusion_model)
    metric = instantiate(cfg.metric)
    executor = instantiate(cfg.parallel_executor)
    _add_timing_row(scope="global", part="0", name="加载配置并实例化各模块", duration_s=time.perf_counter() - _t0)

    _ensure_instance(q_provider, QProvider, name="q_provider")
    _ensure_instance(environment, Environment, name="environment")
    _ensure_instance(diffusion_model, DiffusionModel, name="diffusion_model")
    _ensure_instance(metric, Metric, name="metric")
    _ensure_instance(executor, ParallelExecutor, name="parallel_executor")
    ray_executor = _require_ray_executor(executor)

    M = cfg.algorithm.M
    N = cfg.algorithm.N
    K = cfg.algorithm.K
    initial_population_path = cfg.logging.initial_population_path
    best_rho_path = Path(cfg.logging.best_rho_csv_path)
    best_simulation_data_csv_path = best_rho_path.with_name(f"{best_rho_path.stem}_best_simulation_data.csv")
    global_best_rho: float | None = None
    cached_initial_population = step_3_4_try_load_initial_population(initial_population_path)
    initial_population_save_process: multiprocessing.process.BaseProcess | None = None
    initial_population_refs: list[Any] | None = None
    population: Population | None = None

    if cached_initial_population is not None:
        print(f"检测到初始种群缓存文件，将直接读取：{initial_population_path}")
        population = cached_initial_population

    try:
        # =========================
        # 1. 读取或创建 Q
        # =========================
        print("1. 读取或创建 Q")
        _t1 = time.perf_counter()
        q = step_1_load_or_create_q(q_provider, environment)
        _add_timing_row(scope="global", part="1", name="读取或创建 Q", duration_s=time.perf_counter() - _t1)

        # =========================
        # 2. 随机初始化扩散模型
        # =========================
        print("2. 随机初始化扩散模型")
        _t2 = time.perf_counter()
        diffusion_model = step_2_init_diffusion_model(diffusion_model)
        _add_timing_row(scope="global", part="2", name="随机初始化扩散模型", duration_s=time.perf_counter() - _t2)

        # =========================
        # 3~4. 形成初始种群（默认串行，留出并行接口）
        # =========================
        print(f"3-4. 构建初始种群（种群规模 M={M}，智能体数量 N={N}）")
        _t34 = time.perf_counter()
        if cached_initial_population is not None:
            if len(population) != M:
                print(f"[WARN] 初始种群缓存数量为 {len(population)}，当前配置 M={M}。将继续使用缓存种群。")
        else:
            q_ref = _strict_ray_put(ray_executor, q, label="q")
            tasks = []
            for i in range(1, M + 1):
                tasks.append(
                    {
                        "i": i,
                        "N": N,
                        "seed": int(cfg.seed) + int(i),
                        "q_ref": q_ref,
                        "environment_spec": cfg.environment,
                        "metric_spec": cfg.metric,
                        "reward_min": float(cfg.evox.reward_min),
                        "reward_max": float(cfg.evox.reward_max),
                    }
                )
            from src.parallel.ray_tasks import build_initial_individual as ray_build_initial_individual

            population = ray_executor.map(ray_build_initial_individual, tasks)

            if initial_population_path and population is not None:
                _t34_save = time.perf_counter()
                initial_population_save_process = _start_async_initial_population_save(
                    population,
                    population_path=initial_population_path,
                )
                if initial_population_save_process is None:
                    step_3_4_save_initial_population(population, population_path=initial_population_path)
                    _add_timing_row(
                        scope="global",
                        part="3-4.save_initial_population",
                        name="写入初始种群缓存（同步）",
                        duration_s=time.perf_counter() - _t34_save,
                        count=len(population),
                    )
                    print(f"初始种群已保存到：{initial_population_path}")
                else:
                    _add_timing_row(
                        scope="global",
                        part="3-4.save_initial_population_async_start",
                        name="启动初始种群异步缓存写盘",
                        duration_s=time.perf_counter() - _t34_save,
                        count=len(population),
                    )
                    print(f"初始种群缓存后台写入中：{initial_population_path}")
        _add_timing_row(scope="global", part="3-4", name="构建初始种群（3-4）", duration_s=time.perf_counter() - _t34)
        print("4. 初始种群构建完成")

        if population is not None:
            population, global_best_rho, wrote_best_simulation_data = step_record_best_simulation_data_if_improved(
                population,
                csv_path=str(best_simulation_data_csv_path),
                best_rho_so_far=global_best_rho,
            )
            if wrote_best_simulation_data:
                print(f"4. 已初始化最优个体 simulation_data CSV：{best_simulation_data_csv_path}")

        # =========================
        # 5. for k = 1..K 进化迭代
        # =========================
        _t5_total = time.perf_counter()
        reward_shape = _infer_reward_shape(population=population, q=q, num_agents=N)
        evox_algorithm = build_evox_algorithm(cfg.evox, population or [])

        def evaluate_population_from_vectors(pop_vectors: Any) -> Population:
            torch, _ = import_torch_and_evox(
                cfg.evox.repo_path,
                allow_shim_fallback=cfg.evox.allow_shim_fallback,
            )
            pop_cpu = pop_vectors.detach().cpu() if hasattr(pop_vectors, "detach") else pop_vectors
            rewards_batch = [
                pop_cpu[idx].reshape(reward_shape[0], reward_shape[1]).to(dtype=torch.float32)
                for idx in range(int(pop_cpu.shape[0]))
            ]
            return _evaluate_rewards_batch(
                rewards_batch=rewards_batch,
                q=q,
                environment=environment,
                metric=metric,
                executor=executor,
                environment_spec=cfg.environment,
                metric_spec=cfg.metric,
            )

        evox_problem = build_evox_problem(
            evox_cfg=cfg.evox,
            reward_shape=reward_shape,
            batch_evaluator=evaluate_population_from_vectors,
        )
        std_workflow_cls = get_evox_std_workflow_cls(
            cfg.evox.repo_path,
            allow_shim_fallback=cfg.evox.allow_shim_fallback,
        )
        evox_workflow = std_workflow_cls(
            evox_algorithm,
            evox_problem,
            opt_direction="max",
            device=cfg.evox.device,
        )

        for k in range(1, K + 1):
            print(f"5. EvoX 进化迭代（k={k}/{K}）")
            _t_iter = time.perf_counter()
            if k == 1:
                evox_workflow.init_step()
            else:
                evox_workflow.step()
            population = list(getattr(evox_problem, "last_population", []) or [])
            if not population:
                raise ValueError("EvoX 迭代没有返回任何候选个体。")

            _add_timing_row(
                scope="iter",
                iteration_k=k,
                part="5.evox_step",
                name="EvoX 单次进化迭代",
                duration_s=time.perf_counter() - _t_iter,
                count=len(population),
            )

            print("5.6 记录本次 EvoX 迭代种群最优 ρ 到 CSV")
            _t56 = time.perf_counter()
            population, best_rho_k = step_5_6_record_best_rho(
                population,
                iteration_k=k,
                csv_path=cfg.logging.best_rho_csv_path,
            )
            _add_timing_row(scope="iter", iteration_k=k, part="5.6", name="记录最优 ρ 到 CSV", duration_s=time.perf_counter() - _t56)
            print(f"    本次迭代最优 ρ = {best_rho_k}")

            population, global_best_rho, wrote_best_simulation_data = step_record_best_simulation_data_if_improved(
                population,
                csv_path=str(best_simulation_data_csv_path),
                best_rho_so_far=global_best_rho,
            )
            if wrote_best_simulation_data:
                print(f"    检测到新的全局最优个体，已覆盖写入：{best_simulation_data_csv_path}")

            _add_timing_row(
                scope="iter",
                iteration_k=k,
                part="5.iter_total",
                name="单次进化迭代总耗时（wall time）",
                duration_s=time.perf_counter() - _t_iter,
            )

        best_evox_individual = getattr(evox_problem, "best_individual", None)
        if best_evox_individual is not None:
            population = [best_evox_individual]

        _add_timing_row(scope="global", part="5", name="进化迭代总耗时（5）", duration_s=time.perf_counter() - _t5_total)

        # =========================
        # 6. 用最终最优个体的奖励 R + 纯 A* 再仿真一次，导出所有车辆规划路径为新的 rou 文件
        # =========================
        print("6. 用最终最优个体的奖励 R 导出规划后的 rou 文件")
        _t6 = time.perf_counter()
        best_rho_path = Path(cfg.logging.best_rho_csv_path)
        output_rou_path = best_rho_path.with_name(f"{best_rho_path.stem}_planned_routes_{run_id}.rou.xml")
        population, exported_rou_path = step_6_simulate_best_and_export_routes(
            environment,
            population,
            output_rou_path=str(output_rou_path),
        )
        _add_timing_row(scope="global", part="6", name="导出最优个体规划后的 rou 文件", duration_s=time.perf_counter() - _t6)
        if exported_rou_path:
            print(f"6. 新的 rou 文件已导出：{exported_rou_path}")
        else:
            print("6. 当前环境未实现 rou 导出能力，已跳过该步骤")
        return population
    finally:
        if initial_population_save_process is not None:
            _t_async_wait = time.perf_counter()
            initial_population_save_process.join()
            _add_timing_row(
                scope="global",
                part="finalize.save_initial_population_async_wait",
                name="等待初始种群异步缓存写盘完成",
                duration_s=time.perf_counter() - _t_async_wait,
            )
            if initial_population_save_process.exitcode != 0 and initial_population_path:
                print("[WARN] 初始种群异步缓存写盘失败，正在回退为同步写入")
                _t_async_recover = time.perf_counter()
                step_3_4_save_initial_population(population, population_path=initial_population_path)
                _add_timing_row(
                    scope="global",
                    part="finalize.save_initial_population_recovery",
                    name="异步缓存写盘失败后的同步补写",
                    duration_s=time.perf_counter() - _t_async_recover,
                )
                print(f"初始种群已保存到：{initial_population_path}")
        _add_timing_row(scope="global", part="run_total", name="主流程总耗时（wall time）", duration_s=time.perf_counter() - _t_run_start)
        _finalize_percents()
        _write_timing_csv()
        executor.close()
