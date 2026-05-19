from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

from src.parallel.base import ParallelExecutor

T = TypeVar("T")
R = TypeVar("R")


_RAY_TRANSPORT_ERROR_MARKERS = (
    "broken pipe",
    "connection refused",
    "connection reset by peer",
    "failed to connect to all addresses",
    "grpcunavailable",
    "local raylet died",
    "raylet died",
    "raylet exited",
    "raylet has died",
    "fate shares with the agent",
    "dashboard_agent",
    "runtime_env_agent",
)

_RAY_STARTUP_TIMEOUT_ERROR_MARKERS = (
    "timed out during startup",
    "failed to get node info",
    "node registration may not be complete yet",
    "raylet failed to startup",
    "gcs cannot find the node",
    "deadline exceeded",
)

_RAY_MIN_OBJECT_STORE_MEMORY = 80 * 1024 * 1024

_RAY_UNIX_SOCKET_PATH_SOFT_LIMIT = 100
_RAY_PLASMA_SOCKET_SUFFIX_SAMPLE = os.path.join(
    "session_2000-01-01_00-00-00_000000_000000",
    "sockets",
    "plasma_store",
)


def _callable_name(fn: Any) -> str:
    """
    尽量给出一个稳定的“任务名”，用于按任务名匹配 Ray 资源配置。
    - 普通函数：fn.__name__
    - functools.partial：partial.func.__name__
    - 其它：type(fn).__name__
    """
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return name
    inner = getattr(fn, "func", None)  # e.g. functools.partial
    inner_name = getattr(inner, "__name__", None)
    if isinstance(inner_name, str) and inner_name:
        return inner_name
    return type(fn).__name__


def _looks_like_ray_transport_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    exc_type_name = type(exc).__name__.lower()
    if "localrayletdiederror" in exc_type_name:
        return True
    return any(marker in text for marker in _RAY_TRANSPORT_ERROR_MARKERS)


def _looks_like_ray_startup_timeout(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RAY_STARTUP_TIMEOUT_ERROR_MARKERS)


def _would_exceed_unix_socket_path_limit(temp_dir: str) -> bool:
    if os.name != "posix":
        return False
    candidate = os.path.join(temp_dir, _RAY_PLASMA_SOCKET_SUFFIX_SAMPLE)
    return len(candidate) > _RAY_UNIX_SOCKET_PATH_SOFT_LIMIT


def _get_system_memory_bytes() -> int | None:
    if os.name != "posix":
        return None
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages <= 0 or page_size <= 0:
            return None
        return pages * page_size
    except Exception:
        return None


def _get_dev_shm_bytes() -> int | None:
    if os.name != "posix":
        return None
    try:
        st = os.statvfs("/dev/shm")
        return int(st.f_frsize) * int(st.f_blocks)
    except Exception:
        return None


def _pick_usable_temp_dir(preferred: str) -> str:
    candidates: list[str] = [preferred]
    if os.name == "posix":
        candidates.extend(["/tmp/ray", "/tmp/r"])

    seen: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = os.path.abspath(raw)
        if path in seen:
            continue
        seen.add(path)

        if _would_exceed_unix_socket_path_limit(path):
            continue

        try:
            os.makedirs(path, exist_ok=True)
            return path
        except Exception:
            continue

    return os.path.abspath(preferred)


def _normalize_ray_init_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize Ray init kwargs for cross-environment compatibility."""
    normalized = dict(kwargs)

    if "include_dashboard" not in normalized:
        # 在无头环境中默认关闭 dashboard，减少 Ray 组件启动负担。
        normalized["include_dashboard"] = False

    # Ray expects absolute path for temp_dir/_temp_dir.
    for key in ("_temp_dir", "temp_dir"):
        raw = normalized.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        normalized[key] = _pick_usable_temp_dir(raw)

    raw_store_mem = normalized.get("object_store_memory")
    if isinstance(raw_store_mem, (int, float)):
        store_mem = int(raw_store_mem)
        if store_mem <= 0:
            normalized.pop("object_store_memory", None)
        else:
            shm_bytes = _get_dev_shm_bytes()
            total_mem = _get_system_memory_bytes()

            # /dev/shm 太小的容器中，显式 object_store_memory 往往导致 raylet 启动失败。
            if isinstance(shm_bytes, int) and shm_bytes > 0 and shm_bytes < _RAY_MIN_OBJECT_STORE_MEMORY:
                normalized.pop("object_store_memory", None)
            else:
                caps: list[int] = []
                if isinstance(shm_bytes, int) and shm_bytes > 0:
                    caps.append(int(shm_bytes * 0.8))
                if isinstance(total_mem, int) and total_mem > 0:
                    caps.append(int(total_mem * 0.25))
                if caps:
                    cap = max(_RAY_MIN_OBJECT_STORE_MEMORY, min(caps))
                    store_mem = min(store_mem, cap)
                normalized["object_store_memory"] = store_mem

    return normalized


def _is_unix_socket_path_too_long_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "af_unix path length" in text or "path length cannot exceed" in text


def _ensure_raylet_start_wait_env(wait_time_s: int) -> int:
    """Ensure Ray startup wait time env var is at least wait_time_s."""
    desired = max(30, int(wait_time_s))
    existing_raw = os.environ.get("RAY_raylet_start_wait_time_s")
    existing: int | None = None
    if isinstance(existing_raw, str) and existing_raw.strip():
        try:
            existing = int(existing_raw)
        except Exception:
            existing = None

    effective = desired if existing is None else max(desired, existing)
    os.environ["RAY_raylet_start_wait_time_s"] = str(effective)
    return effective


def _disable_ray_usage_stats() -> None:
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")


def _build_retry_temp_dir(base_dir: str, attempt: int) -> str:
    root = _pick_usable_temp_dir(_strip_retry_suffix(base_dir))
    candidate = os.path.join(root, f"retry_{os.getpid()}_{attempt}")
    return _pick_usable_temp_dir(candidate)


def _strip_retry_suffix(path: str) -> str:
    if not isinstance(path, str) or not path:
        return "/tmp/r"
    marker = f"{os.sep}retry_"
    idx = path.find(marker)
    if idx <= 0:
        return path
    return path[:idx]


def _cleanup_stale_ray_sessions(temp_dir: str) -> None:
    """Best-effort cleanup for stale session pointers under temp_dir."""
    if not isinstance(temp_dir, str) or not temp_dir.strip():
        return
    root = os.path.abspath(temp_dir)
    try:
        entries = os.listdir(root)
    except Exception:
        return

    for name in entries:
        if name == "session_latest" or name.startswith("session_"):
            target = os.path.join(root, name)
            try:
                if os.path.islink(target) or os.path.isfile(target):
                    os.unlink(target)
                elif os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
            except Exception:
                continue


def _force_stop_ray_processes() -> None:
    """Best-effort hard stop for stale ray processes between retries."""
    try:
        subprocess.run(
            ["ray", "stop", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def _ensure_local_node_ip(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Force a stable local node ip in containerized single-node execution."""
    updated = dict(kwargs)
    if "_node_ip_address" not in updated and "node_ip_address" not in updated:
        updated["_node_ip_address"] = os.environ.get("MASDIFF_RAY_NODE_IP", "127.0.0.1")
    return updated


def _format_ray_init_kwargs(kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(kwargs):
        parts.append(f"{key}={kwargs[key]!r}")
    return ", ".join(parts)


def _build_ray_startup_failure(exc: BaseException, *, attempts: int, kwargs: dict[str, Any]) -> RuntimeError:
    temp_dir = kwargs.get("_temp_dir") or kwargs.get("temp_dir") or "/tmp/ray"
    return RuntimeError(
        "Ray startup failed after "
        f"{attempts} attempt(s). "
        f"Effective ray.init kwargs: {_format_ray_init_kwargs(kwargs)}. "
        f"Check `{temp_dir}/session_latest/logs/` for raylet/gcs logs, "
        f"and clean `{temp_dir}` if you need to remove stale Ray state."
    )


class RayExecutor(ParallelExecutor):
    """
    基于 Ray 的并行执行器。

    设计目标：
    - runner.py 只依赖 ParallelExecutor.map，因此这里实现 map(fn, items) 语义。
    - 支持按“任务名”（fn.__name__）配置资源，例如：
      - build_initial_individual（构建初始种群个体）
      - mutate_one（精英个体变异）
    - 通过 Ray 的 task options 控制 CPU/GPU 等资源占用。

    配置示例（YAML 中 parallel_executor.kwargs）：
      ray_init_kwargs:
        num_cpus: 16
        num_gpus: 1
        object_store_memory: 8000000000   # bytes
      default_task_options:
        num_cpus: 1
      task_options_by_name:
        build_initial_individual:
          num_cpus: 4
          num_gpus: 1
        mutate_one:
          num_cpus: 4
          num_gpus: 1
    """

    def __init__(
        self,
        *,
        ray_init_kwargs: dict[str, Any] | None = None,
        default_task_options: dict[str, Any] | None = None,
        task_options_by_name: dict[str, dict[str, Any]] | None = None,
        show_progress: bool = True,
        desc: str = "RayExecutor",
        max_updates: int = 20,
        max_inflight_tasks: int | None = None,
        ray_startup_retries: int = 3,
        ray_startup_retry_delay_s: float = 3.0,
        raylet_start_wait_time_s: int = 120,
        enable_eval_gpu: bool = False,
        eval_gpu_per_task: float = 0.1,
        fallback_max_workers: int = 25,
        shutdown_on_close: bool = True,
    ) -> None:
        self.show_progress = bool(show_progress)
        self.desc = str(desc)
        self.max_updates = int(max_updates)
        if max_inflight_tasks is None:
            env_value = os.environ.get("MASDIFF_RAY_MAX_INFLIGHT", "8")
            try:
                parsed = int(env_value)
            except Exception:
                parsed = 8
        else:
            parsed = int(max_inflight_tasks)
        self.max_inflight_tasks = max(0, parsed)
        self.ray_startup_retries = max(1, int(ray_startup_retries))
        self.ray_startup_retry_delay_s = max(0.0, float(ray_startup_retry_delay_s))
        self.raylet_start_wait_time_s = max(30, int(raylet_start_wait_time_s))
        self.enable_eval_gpu = bool(enable_eval_gpu)
        self.eval_gpu_per_task = max(0.0, float(eval_gpu_per_task))
        self.fallback_max_workers = max(1, int(fallback_max_workers))
        self.shutdown_on_close = bool(shutdown_on_close)
        self._ray_init_kwargs = _normalize_ray_init_kwargs(dict(ray_init_kwargs or {}))
        self._shared_refs: dict[str, Any] = {}
        self._shared_values: dict[str, Any] = {}
        self._ref_payloads: dict[str, Any] = {}
        self._thread_pool: ThreadPoolExecutor | None = None
        self._ray_fallback_enabled = False
        self._ray_startup_error: BaseException | None = None
        self._warned_gpu_downgrade_tasks: set[str] = set()

        self._default_task_options = dict(default_task_options or {})
        self._task_options_by_name = {str(k): dict(v) for k, v in (task_options_by_name or {}).items()}
        self._ray: Any | None = None
        self._owns_ray = False

        # 延迟导入：避免用户未安装 ray 时，import 直接失败
        try:
            import ray  # type: ignore
        except Exception as e:  # pragma: no cover
            self._ray = None
            self._ray_startup_error = e
            self._enable_thread_fallback(
                reason=(
                    "ray import failed; "
                    f"{type(e).__name__}: {e}"
                )
            )
            return

        self._ray = ray
        self._owns_ray = False

        # Ray 官方错误提示中给出了该环境变量；确保使用不小于配置值的等待时间。
        _disable_ray_usage_stats()
        self.raylet_start_wait_time_s = _ensure_raylet_start_wait_env(self.raylet_start_wait_time_s)

        if not ray.is_initialized():
            try:
                self._start_ray_session()
            except Exception as e:
                self._ray_startup_error = e
                self._enable_thread_fallback(
                    reason=(
                        "ray init failed after retries; "
                        f"{type(e).__name__}: {e}"
                    )
                )

    def _enable_thread_fallback(self, *, reason: str) -> None:
        if self._ray_fallback_enabled:
            return
        self._ray_fallback_enabled = True
        self._owns_ray = False
        print(
            "[RayExecutor] Ray unavailable; "
            f"fallback to ThreadPoolExecutor({self.fallback_max_workers}). "
            f"reason={reason}"
        )

    def _ensure_thread_pool(self) -> ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.fallback_max_workers,
                thread_name_prefix="masdiff-fallback",
            )
        return self._thread_pool

    def _thread_map_once(self, fn: Callable[[T], R], seq: list[T]) -> list[R]:
        total = len(seq)
        if total == 0:
            return []

        pool = self._ensure_thread_pool()
        results: list[Any] = [None] * total
        future_to_idx = {pool.submit(fn, item): idx for idx, item in enumerate(seq)}

        if self.show_progress:
            max_updates = max(1, self.max_updates)
            step = max(1, total // max_updates)
            print(f"{self.desc}[thread]({ _callable_name(fn) }): 0/{total}")
        else:
            step = 0

        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if self.show_progress and (done == total or (done % step == 0)):
                print(f"{self.desc}[thread]({ _callable_name(fn) }): {done}/{total}")

        return list(results)

    def _start_ray_session(self) -> bool:
        if self._ray is None:
            raise RuntimeError("ray module is unavailable")

        last_error: Exception | None = None
        startup_kwargs = _ensure_local_node_ip(self._ray_init_kwargs)

        for attempt in range(1, self.ray_startup_retries + 1):
            try:
                print(
                    "[RayExecutor] ray.init attempt "
                    f"{attempt}/{self.ray_startup_retries} with "
                    f"RAY_raylet_start_wait_time_s={os.environ.get('RAY_raylet_start_wait_time_s', 'unset')}"
                )
                self._ray.init(**startup_kwargs)
                self._ray_init_kwargs = dict(startup_kwargs)
                self._owns_ray = True
                return True
            except Exception as e:
                last_error = e
                try:
                    self._ray.shutdown()
                except Exception:
                    pass

                _force_stop_ray_processes()

                retry_kwargs: dict[str, Any] | None = None
                if _is_unix_socket_path_too_long_error(e):
                    retry_kwargs = dict(startup_kwargs)
                    retry_kwargs.pop("temp_dir", None)
                    retry_kwargs["_temp_dir"] = _build_retry_temp_dir("/tmp/r", attempt)
                elif _looks_like_ray_startup_timeout(e):
                    retry_kwargs = dict(startup_kwargs)
                    temp_root = str(retry_kwargs.get("_temp_dir") or retry_kwargs.get("temp_dir") or "/tmp/r")
                    retry_kwargs.pop("temp_dir", None)
                    retry_kwargs["_temp_dir"] = _build_retry_temp_dir(temp_root, attempt)
                    raw_mem = retry_kwargs.get("object_store_memory")
                    if isinstance(raw_mem, (int, float)) and int(raw_mem) > _RAY_MIN_OBJECT_STORE_MEMORY:
                        retry_kwargs["object_store_memory"] = max(
                            _RAY_MIN_OBJECT_STORE_MEMORY,
                            int(int(raw_mem) * 0.5),
                        )
                    raw_cpus = retry_kwargs.get("num_cpus")
                    if isinstance(raw_cpus, (int, float)) and float(raw_cpus) > 8:
                        retry_kwargs["num_cpus"] = max(4, int(float(raw_cpus) * 0.5))
                    # 实测部分环境显式 num_gpus 会放大 raylet 启动失败概率，超时后改为自动探测。
                    retry_kwargs.pop("num_gpus", None)

                can_retry = retry_kwargs is not None and attempt < self.ray_startup_retries
                if not can_retry:
                    raise

                startup_kwargs = _ensure_local_node_ip(_normalize_ray_init_kwargs(retry_kwargs))
                retry_temp_dir = str(startup_kwargs.get("_temp_dir") or startup_kwargs.get("temp_dir") or "")
                _cleanup_stale_ray_sessions(retry_temp_dir)
                print(
                    "[RayExecutor] ray.init failed and will retry. "
                    f"reason={type(e).__name__}: {e}. "
                    f"next kwargs: {_format_ray_init_kwargs(startup_kwargs)}"
                )
                if self.ray_startup_retry_delay_s > 0:
                    time.sleep(self.ray_startup_retry_delay_s)

        if last_error is not None:
            if _looks_like_ray_startup_timeout(last_error) or _is_unix_socket_path_too_long_error(last_error):
                raise _build_ray_startup_failure(
                    last_error,
                    attempts=self.ray_startup_retries,
                    kwargs=startup_kwargs,
                ) from last_error
            raise last_error
        raise RuntimeError("Ray init failed with unknown error")

    def is_ray_active(self) -> bool:
        if self._ray_fallback_enabled or self._ray is None:
            return False
        try:
            return bool(self._ray.is_initialized())
        except Exception:
            return False

    def safe_put(self, value: Any, *, label: str = "object") -> Any:
        if label == "q":
            self._shared_values[label] = value

        if self._ray_fallback_enabled:
            # 回退到线程池时无需 ray.put，统一按值传递。
            return value

        if self._ray is None:
            raise RuntimeError("ray module is unavailable")

        if label != "q":
            # 临时批次对象不做显式 put，避免在 agent 不稳定时放大 put 失败面。
            return value

        # 保留原始值，用于 Ray 会话重启后的参数重建。
        self._shared_values[label] = value

        if not self.is_ray_active():
            restarted = self._restart_ray()
            if (not restarted) and self._ray_fallback_enabled:
                # Ray 已不可用，直接按值传递，交给线程池回退路径处理。
                return value

        # 仅缓存可复用的大对象（例如 q），避免重复 put。
        cached = self._shared_refs.get(label)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                ref = self._ray.put(value)
                # 仅对明确可复用对象开启缓存。
                if label == "q":
                    self._shared_refs[label] = ref
                    self._ref_payloads[str(ref)] = value
                return ref
            except Exception as exc:
                last_error = exc
                if attempt == 0 and _looks_like_ray_transport_error(exc):
                    restarted = self._restart_ray()
                    if restarted:
                        continue
                    if self._ray_fallback_enabled:
                        return value
                break

        if last_error is not None:
            # q put 失败时降级按值传递，由上游决定是否继续。
            raise last_error
        return value

    def _restore_known_object_refs(self, value: Any) -> Any:
        object_ref_type = getattr(self._ray, "ObjectRef", None)
        if object_ref_type is not None and isinstance(value, object_ref_type):
            payload = self._ref_payloads.get(str(value))
            if payload is not None:
                return payload
            return value

        if isinstance(value, dict):
            return {k: self._restore_known_object_refs(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._restore_known_object_refs(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._restore_known_object_refs(v) for v in value)
        return value

    def _prepare_seq_for_retry(self, seq: list[T]) -> list[T]:
        restored = [self._restore_known_object_refs(item) for item in seq]
        return list(restored)

    def _options_for(self, fn: Callable[[T], R]) -> dict[str, Any]:
        name = _callable_name(fn)
        # 任务级 options：默认 + 按名字覆盖（同名 key 覆盖默认）
        opts = dict(self._default_task_options)
        by_name = self._task_options_by_name.get(name)
        if by_name:
            opts.update(by_name)
        if self.enable_eval_gpu and name in {"evaluate_reward_candidate", "evaluate_reward_batch"}:
            opts.setdefault("num_gpus", self.eval_gpu_per_task)

        raw_num_gpus = opts.get("num_gpus")
        if isinstance(raw_num_gpus, (int, float)):
            requested = float(raw_num_gpus)
            if requested <= 0:
                opts.pop("num_gpus", None)
            elif self._ray is not None:
                available = self.cluster_gpu_count()
                if available <= 0:
                    opts.pop("num_gpus", None)
                    if name not in self._warned_gpu_downgrade_tasks:
                        print(
                            "[RayExecutor] no GPU resources detected in Ray cluster; "
                            f"task `{name}` falls back to CPU scheduling."
                        )
                        self._warned_gpu_downgrade_tasks.add(name)
        return opts

    def options_for_name(self, name: str) -> dict[str, Any]:
        opts = dict(self._default_task_options)
        by_name = self._task_options_by_name.get(str(name))
        if by_name:
            opts.update(by_name)
        return opts

    def cluster_gpu_count(self) -> float:
        if self._ray_fallback_enabled:
            return 0.0
        if self._ray is None:
            return 0.0
        try:
            resources = self._ray.cluster_resources()
        except Exception:
            return 0.0
        return float(resources.get("GPU", 0.0))

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        seq = list(items)
        if self._ray_fallback_enabled or self._ray is None:
            return self._thread_map_once(fn, seq)

        if self._ray is None:
            raise RuntimeError("ray module is unavailable")
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                if not self.is_ray_active():
                    restarted = self._restart_ray()
                    if not restarted:
                        if self._ray_fallback_enabled:
                            seq = self._prepare_seq_for_retry(seq)
                            return self._thread_map_once(fn, seq)
                        raise RuntimeError("Ray restart failed before task submission")
                return self._map_once(fn, seq)
            except Exception as exc:
                last_error = exc
                if attempt == 0 and _looks_like_ray_transport_error(exc):
                    # Ray 重启后，旧会话 ObjectRef 会出现 owner unknown；重试前先替换成已知原值。
                    seq = self._prepare_seq_for_retry(seq)
                    restarted = self._restart_ray()
                    if restarted:
                        continue
                    if self._ray_fallback_enabled:
                        return self._thread_map_once(fn, seq)
                if _looks_like_ray_transport_error(exc):
                    seq = self._prepare_seq_for_retry(seq)
                    self._enable_thread_fallback(
                        reason=(
                            "ray transport failure during map; "
                            f"{type(exc).__name__}: {exc}"
                        )
                    )
                    return self._thread_map_once(fn, seq)
                raise

        if last_error is not None:
            raise last_error
        return []

    def _restart_ray(self) -> bool:
        if self._ray_fallback_enabled:
            return False
        if self._ray is None:
            return False
        try:
            self._ray.shutdown()
        except Exception:
            pass
        self._shared_refs.clear()
        self._ref_payloads.clear()
        try:
            return bool(self._start_ray_session())
        except Exception as e:
            self._ray_startup_error = e
            self._enable_thread_fallback(
                reason=(
                    "ray restart failed; "
                    f"{type(e).__name__}: {e}"
                )
            )
            return False

    def _map_once(self, fn: Callable[[T], R], seq: list[T]) -> list[R]:
        if self._ray is None:
            raise RuntimeError("ray module is unavailable")
        ray = self._ray
        total = len(seq)
        if total == 0:
            return []

        options = self._options_for(fn)
        remote_fn = ray.remote(fn).options(**options) if options else ray.remote(fn)
        max_inflight = int(getattr(self, "max_inflight_tasks", 0))

        if max_inflight <= 0 or max_inflight >= total:
            refs = [remote_fn.remote(x) for x in seq]

            if self.show_progress:
                max_updates = max(1, self.max_updates)
                step = max(1, total // max_updates)
                print(f"{self.desc}({ _callable_name(fn) }): 0/{total}")

                pending = list(refs)
                done = 0
                while pending:
                    ready, pending = ray.wait(pending, num_returns=1, timeout=None)
                    done += len(ready)
                    if done == total or (done % step == 0):
                        print(f"{self.desc}({ _callable_name(fn) }): {done}/{total}")

            return list(ray.get(refs))

        if self.show_progress:
            max_updates = max(1, self.max_updates)
            step = max(1, total // max_updates)
            print(f"{self.desc}({ _callable_name(fn) }): 0/{total}")
        else:
            step = 0

        results: list[Any] = [None] * total
        pending: list[Any] = []
        ref_to_idx: dict[Any, int] = {}

        next_idx = 0
        initial = min(total, max_inflight)
        for _ in range(initial):
            ref = remote_fn.remote(seq[next_idx])
            pending.append(ref)
            ref_to_idx[ref] = next_idx
            next_idx += 1

        done = 0
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=None)
            ref = ready[0]
            idx = ref_to_idx.pop(ref)
            results[idx] = ray.get(ref)
            done += 1

            if self.show_progress and (done == total or (done % step == 0)):
                print(f"{self.desc}({ _callable_name(fn) }): {done}/{total}")

            if next_idx < total:
                new_ref = remote_fn.remote(seq[next_idx])
                pending.append(new_ref)
                ref_to_idx[new_ref] = next_idx
                next_idx += 1

        return list(results)

    def close(self) -> None:
        self._shared_refs.clear()
        self._shared_values.clear()
        self._ref_payloads.clear()

        if self._thread_pool is not None:
            try:
                self._thread_pool.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            self._thread_pool = None

        if not self.shutdown_on_close:
            return None
        # 仅在由本执行器初始化 Ray 时 shutdown，避免影响外部 Ray 会话
        if not getattr(self, "_owns_ray", False):
            return None
        if self._ray is None:
            return None
        try:
            self._ray.shutdown()
        except Exception:
            return None
        return None

