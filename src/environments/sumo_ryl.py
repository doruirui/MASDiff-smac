from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.environments.base import Environment


class SumoRylEnvironment(Environment):
    """
    一个基于 SUMO/libsumo 的环境实现（供 MASDiff 框架接入）。

    关键点（按你的要求）：
    - policies 是 DQN 模型列表（长度 = num_car），一辆车对应一个 policy（索引一致）
    - 仿真不再使用外部“权重矩阵”；路径代价使用 SUMO 路网长度：
      - 使用 libsumo.edge.getLength(edge_id) 获取路段长度（作为 A* 的边代价）
    - Tau 是张量：
      - tau.shape == [num_car, num_road, 2]
      - 2个特征：[当前路距离终点的距离, 当前路的排队长度]
      - 车辆在“首次出现”时填充其对应的 tau 行
    - simulation_data 也是张量：
      - 仿真 end_tick=600，sample_interval=6 => 100 个时间点
      - simulation_data.shape == [100, num_road]（每行是一次采样时刻的全网排队长度向量）
    - 路径规划：
      - 车辆在出现时进行路径规划
      - 若传入的是奖励矩阵 R，则直接使用“R + 纯 A*”
      - 若传入的是可用 policy，则沿路径逐边使用“DQN 选下一条边 + A* 启发式/回退”
      - 若既没有奖励也没有可用 policy（例如 policy=None），则只用 A*（按长度）
    """

    def __init__(
        self,
        *,
        sumo_config: str,
        net_file: str | None = None,
        route_file: str | None = None,
        sumo_use_gui: bool = False,
        sumo_binary: str | None = None,
        end_tick: int = 600,
        update_interval: int = 6,
        sample_interval: int = 6,
        # 设备策略：训练/存储在 CPU，但“使用 DQN 做决策”可在 GPU 上（用完再搬回 CPU）
        move_policies_to_device: bool = True,
        policy_device: str = "auto",  # auto/cpu/cuda/cuda:0...
        reward_astar_weight: float = 1.0,
        controlled_vehicle_ids: list[str] | None = None,
        traci_label: str | None = None,
        extra_sumo_args: list[str] | None = None,
        sumo_start_retries: int = 2,
        sumo_start_retry_delay: float = 0.5,
        traci_auto_unique_label: bool = True,
    ) -> None:
        self.sumo_config = self._resolve_path_for_workers(str(sumo_config))
        self.net_file = self._resolve_path_for_workers(str(net_file)) if net_file else None
        self.route_file = self._resolve_path_for_workers(str(route_file)) if route_file else None
        self.sumo_use_gui = bool(sumo_use_gui)
        self.sumo_binary = str(sumo_binary) if sumo_binary else None
        self.end_tick = int(end_tick)
        self.update_interval = int(update_interval)
        self.sample_interval = int(sample_interval)
        self.move_policies_to_device = bool(move_policies_to_device)
        self.policy_device = str(policy_device)
        self.reward_astar_weight = float(reward_astar_weight)
        self.controlled_vehicle_ids = list(controlled_vehicle_ids) if controlled_vehicle_ids else None
        self.traci_label = str(traci_label) if traci_label else None
        self.extra_sumo_args = list(extra_sumo_args) if extra_sumo_args else []
        self.sumo_start_retries = int(sumo_start_retries)
        self.sumo_start_retry_delay = float(sumo_start_retry_delay)
        self.traci_auto_unique_label = bool(traci_auto_unique_label)

        if self.end_tick <= 0:
            raise ValueError("end_tick 必须 > 0")
        if self.update_interval <= 0:
            raise ValueError("update_interval 必须 > 0")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval 必须 > 0")
        if self.sumo_start_retries < 0:
            raise ValueError("sumo_start_retries 必须 >= 0")
        if self.sumo_start_retry_delay < 0:
            raise ValueError("sumo_start_retry_delay 必须 >= 0")

    def _make_traci_label(self, attempt: int) -> str | None:
        if self.traci_label:
            return self.traci_label
        if not self.traci_auto_unique_label:
            return None
        return f"sumo-{os.getpid()}-{attempt}-{uuid4().hex[:8]}"

    @staticmethod
    def _cleanup_traci_connection_by_label(traci_mod: Any, label: str | None) -> None:
        if not label:
            return None
        try:
            conns = getattr(traci_mod, "_connections", None)
            if isinstance(conns, dict):
                conn = conns.get(label)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                conns.pop(label, None)
        except Exception:
            return None

    @staticmethod
    def _resolve_path_for_workers(path_str: str) -> str:
        """Resolve relative paths robustly across Ray worker cwd differences."""
        p = Path(path_str).expanduser()
        if p.is_absolute():
            return str(p)

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [Path.cwd() / p, repo_root / p]
        for c in candidates:
            if c.exists():
                return str(c.resolve())

        # Keep a deterministic absolute fallback even when file does not exist yet.
        return str((repo_root / p).resolve())

    def _resolve_policy_device(self, torch_mod) -> str:
        """
        返回用于“策略推理”的 device 字符串：
        - policy_device=auto：有 CUDA 则 cuda，否则 cpu
        - policy_device=cuda/cuda:0：若无 CUDA 则回退 cpu
        """
        dev = (self.policy_device or "auto").lower()
        if dev == "auto":
            return "cuda" if torch_mod.cuda.is_available() else "cpu"
        if dev.startswith("cuda") and (not torch_mod.cuda.is_available()):
            return "cpu"
        return self.policy_device

    def _maybe_move_policies(self, policies: list[Any], device: str, torch_mod) -> None:
        """将 policies 中的 torch.nn.Module 移到指定 device。"""
        try:
            module_cls = torch_mod.nn.Module
        except Exception:
            return None
        for p in policies:
            if p is None:
                continue
            if isinstance(p, module_cls):
                p.to(device)
                p.eval()
        return None

    def _looks_like_reward_matrix(self, planner_input: Any) -> bool:
        shape = getattr(planner_input, "shape", None)
        if shape is not None:
            try:
                return len(shape) == 2
            except Exception:
                pass

        if isinstance(planner_input, (list, tuple)) and len(planner_input) > 0:
            first = planner_input[0]
            first_shape = getattr(first, "shape", None)
            if first_shape is not None:
                try:
                    return len(first_shape) == 1
                except Exception:
                    pass
            return isinstance(first, (list, tuple))

        return False

    def _split_planner_input(self, planner_input: Any) -> tuple[list[Any], Any | None]:
        """
        兼容两类输入：
        - policies: list[Any]
        - rewards: [num_car, num_road]
        """
        if self._looks_like_reward_matrix(planner_input):
            try:
                num_car = int(len(planner_input))
            except Exception as e:
                raise TypeError("奖励矩阵无法解析 num_car。") from e
            return [None] * num_car, planner_input

        if isinstance(planner_input, list):
            return planner_input, None
        if isinstance(planner_input, tuple):
            return list(planner_input), None

        raise TypeError("planner_input 必须是策略列表，或 shape=[num_car,num_road] 的奖励矩阵。")

    def _get_reward_row(self, rewards: Any | None, car_idx: int) -> Any | None:
        if rewards is None:
            return None
        try:
            return rewards[car_idx]
        except Exception:
            return None

    def _reward_value_for_edge(self, reward_row: Any | None, edge_index: int) -> float:
        if reward_row is None:
            return 0.0
        try:
            value = reward_row[edge_index]
        except Exception:
            return 0.0

        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass

        try:
            return max(0.0, float(value))
        except Exception:
            return 0.0

    # -------------------------
    # Environment 接口实现
    # -------------------------
    def simulate_collect(self, planner_input: Any) -> tuple[list[list[Any]], Any]:
        """
        用给定策略仿真一次，收集经验库与 Tau（奖励先留空）。

        返回：
        - experience_buffers: list[list[experience]]，长度为 N（N = num_car）
          experience = [s, a, r]
          s = [当前road_id, [特征1, 特征2]]，其中特征为 [距离到终点, 当前路排队长度]
          a = 下一条路（这里使用下一条路的 index，便于 DQN 以离散动作建模）
          r = None（留空，后续由扩散模型填充）
        - tau: torch.Tensor，shape=[num_car, num_road, 2]
        """
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("SumoRylEnvironment 需要 torch 以返回张量 tau。请先安装 torch。") from e

        traci_conn = self._start_sumo()
        try:
            policies, rewards = self._split_planner_input(planner_input)
            # “使用 DQN 做决策”阶段：临时搬到 GPU（结束再搬回 CPU，满足“平常存储在 CPU 上”）
            if self.move_policies_to_device:
                infer_dev = self._resolve_policy_device(torch)
                self._maybe_move_policies(policies, infer_dev, torch)

            edge_ids, outgoing_map = self._load_network_and_edges(traci_conn)
            num_road = len(edge_ids)
            edge_id_to_index = {e: i for i, e in enumerate(edge_ids)}

            # 路段长度是静态的：启动后取一次即可（traci 获取长度）
            edge_length_map = self._collect_edge_length_map(traci_conn, edge_ids)

            # 反向邻接：用于“从终点反推”一次性算出所有路到终点的最短距离
            incoming_map = self._build_incoming_map(outgoing_map)

            num_car = len(policies)
            experience_buffers: list[list[Any]] = [[] for _ in range(num_car)]

            # tau: [num_car, num_road, 2]
            tau = torch.zeros((num_car, num_road, 2), dtype=torch.float32)
            tau[:, :, 0] = float("inf")  # 距离默认不可达

            # vehicle_id -> policy index
            vehicle_id_to_index, dynamic_next_index = self._build_vehicle_id_to_index_map(policies, traci_conn)

            # 预构建“路网连接经验模板”：所有 (from_edge, to_edge) 对
            edge_connections: list[tuple[str, str]] = []
            for u, outs in outgoing_map.items():
                if u not in edge_id_to_index:
                    continue
                for v in outs:
                    if v in edge_id_to_index:
                        edge_connections.append((u, v))

            # 缓存：同一终点的距离向量可复用
            dist_cache: dict[str, list[float]] = {}
            previous_vehicles: set[str] = set()
            filled_indices: set[int] = set()
            planned_vehicle_ids: set[str] = set()

            tick = 0
            while tick < self.end_tick:
                tick += 1
                traci_conn.simulationStep()

                # 采集当前“路段排队长度”（这里用 halting number 近似 queue length）
                queue_len_map = self._collect_queue_length_map(traci_conn, edge_ids)
                current_vehicles = set(traci_conn.vehicle.getIDList())
                new_vehicles = current_vehicles - previous_vehicles

                # 车辆“首次出现”时：
                # 1) 生成该车辆对应的 tau 行（[num_road,2]）
                # 2) 生成该车辆对应的经验库（奖励留空）
                # 3) 进行一次路径规划：有 DQN 用 DQN + A*，无 DQN 只用 A*
                for veh_id in new_vehicles:
                    if veh_id in planned_vehicle_ids:
                        continue

                    # 动态绑定（仅当没有 route_file / controlled_vehicle_ids 时使用）
                    if veh_id not in vehicle_id_to_index and dynamic_next_index < num_car:
                        vehicle_id_to_index[veh_id] = dynamic_next_index
                        dynamic_next_index += 1

                    if veh_id not in vehicle_id_to_index:
                        continue
                    planned_vehicle_ids.add(veh_id)

                    car_idx = int(vehicle_id_to_index[veh_id])
                    if not (0 <= car_idx < num_car):
                        continue
                    if car_idx in filled_indices:
                        continue

                    # 终点路段（用于距离特征）
                    dest_edge = self._get_vehicle_destination_edge(traci_conn, veh_id)
                    if not dest_edge or dest_edge not in edge_id_to_index:
                        continue

                    if dest_edge not in dist_cache:
                        dist_cache[dest_edge] = self._compute_all_distances_to_goal(
                            goal_edge=dest_edge,
                            edge_ids=edge_ids,
                            incoming_map=incoming_map,
                            edge_length_map=edge_length_map,
                        )
                    dists = dist_cache[dest_edge]  # len=num_road

                    # 当前时刻的“全网排队长度向量”
                    queue_vec = [float(queue_len_map.get(eid, 0.0)) for eid in edge_ids]

                    # 写入 tau[car_idx, :, :]
                    tau[car_idx, :, 0] = torch.tensor(dists, dtype=torch.float32)
                    tau[car_idx, :, 1] = torch.tensor(queue_vec, dtype=torch.float32)
                    filled_indices.add(car_idx)

                    # 生成经验库：遍历路网连接关系，形成 (s,a,r=None)
                    for u, v in edge_connections:
                        u_i = edge_id_to_index[u]
                        v_i = edge_id_to_index[v]
                        s = [u, [float(dists[u_i]), float(queue_vec[u_i])]]
                        a = int(v_i)
                        experience_buffers[car_idx].append([s, a, None])

                    # 车辆出现时进行一次路径规划
                    start_edge = traci_conn.vehicle.getRoadID(veh_id)
                    if start_edge and (start_edge in edge_id_to_index) and start_edge != dest_edge:
                        planned_route = self._plan_route_on_appearance(
                            policy=policies[car_idx] if car_idx < len(policies) else None,
                            reward_row=self._get_reward_row(rewards, car_idx),
                            traci_conn=traci_conn,
                            vehicle_id=veh_id,
                            start_edge=start_edge,
                            dest_edge=dest_edge,
                            outgoing_map=outgoing_map,
                            edge_length_map=edge_length_map,
                            edge_id_to_index=edge_id_to_index,
                            dists_to_dest=dists,
                            queue_len_map=queue_len_map,
                        )
                        if planned_route:
                            self._try_set_vehicle_route(traci_conn, veh_id, planned_route)

                previous_vehicles = current_vehicles

            return experience_buffers, tau
        finally:
            # 结束后把 policies 放回 CPU（避免长期占用显存；也符合“平常存储在 CPU 上”）
            if self.move_policies_to_device:
                try:
                    self._maybe_move_policies(policies, "cpu", torch)
                except Exception:
                    pass
            self._close_sumo(traci_conn)

    def simulate_evaluate(self, planner_input: Any) -> Any:
        """
        用给定策略仿真一次，返回用于评估的 simulation_data 张量。

        返回：
        - simulation_data: torch.Tensor，shape=[end_tick/sample_interval, num_road]
        """
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("SumoRylEnvironment 需要 torch 以返回张量 simulation_data。请先安装 torch。") from e

        traci_conn = self._start_sumo()
        try:
            policies, rewards = self._split_planner_input(planner_input)
            if self.move_policies_to_device:
                infer_dev = self._resolve_policy_device(torch)
                self._maybe_move_policies(policies, infer_dev, torch)

            edge_ids, outgoing_map = self._load_network_and_edges(traci_conn)
            num_road = len(edge_ids)
            edge_id_set = set(edge_ids)
            edge_id_to_index = {e: i for i, e in enumerate(edge_ids)}
            edge_length_map = self._collect_edge_length_map(traci_conn, edge_ids)
            incoming_map = self._build_incoming_map(outgoing_map)

            num_car = len(policies)
            vehicle_id_to_index, dynamic_next_index = self._build_vehicle_id_to_index_map(policies, traci_conn)
            dist_cache: dict[str, list[float]] = {}

            num_points = self.end_tick // self.sample_interval
            simulation_data = torch.zeros((num_points, num_road), dtype=torch.float32)
            point_idx = 0

            previous_vehicles: set[str] = set()
            planned_vehicle_ids: set[str] = set()

            tick = 0
            while tick < self.end_tick:
                tick += 1
                traci_conn.simulationStep()

                queue_len_map = self._collect_queue_length_map(traci_conn, edge_ids)
                current_vehicles = set(traci_conn.vehicle.getIDList())
                new_vehicles = current_vehicles - previous_vehicles

                # 车辆出现时进行一次路径规划（评估也遵循同样规则）
                for veh_id in new_vehicles:
                    if veh_id in planned_vehicle_ids:
                        continue
                    if veh_id not in vehicle_id_to_index and dynamic_next_index < num_car:
                        vehicle_id_to_index[veh_id] = dynamic_next_index
                        dynamic_next_index += 1
                    if veh_id not in vehicle_id_to_index:
                        continue
                    planned_vehicle_ids.add(veh_id)

                    car_idx = int(vehicle_id_to_index[veh_id])
                    if not (0 <= car_idx < num_car):
                        continue

                    dest_edge = self._get_vehicle_destination_edge(traci_conn, veh_id)
                    if not dest_edge:
                        continue

                    # 计算 dists（若目的地不在 edge_ids，规划会自然失败/跳过）
                    if dest_edge not in dist_cache and dest_edge in edge_id_set:
                        dist_cache[dest_edge] = self._compute_all_distances_to_goal(
                            goal_edge=dest_edge,
                            edge_ids=edge_ids,
                            incoming_map=incoming_map,
                            edge_length_map=edge_length_map,
                        )

                    start_edge = traci_conn.vehicle.getRoadID(veh_id)
                    if not start_edge or start_edge == dest_edge:
                        continue

                    if dest_edge in dist_cache:
                        planned_route = self._plan_route_on_appearance(
                            policy=policies[car_idx] if car_idx < len(policies) else None,
                            reward_row=self._get_reward_row(rewards, car_idx),
                            traci_conn=traci_conn,
                            vehicle_id=veh_id,
                            start_edge=start_edge,
                            dest_edge=dest_edge,
                            outgoing_map=outgoing_map,
                            edge_length_map=edge_length_map,
                            edge_id_to_index=edge_id_to_index,
                            dists_to_dest=dist_cache[dest_edge],
                            queue_len_map=queue_len_map,
                        )
                        if planned_route:
                            self._try_set_vehicle_route(traci_conn, veh_id, planned_route)

                previous_vehicles = current_vehicles

                # 按采样间隔记录全网排队长度向量，形成 [100,114] 张量
                if tick % self.sample_interval == 0 and point_idx < simulation_data.shape[0]:
                    queue_vec = [float(queue_len_map.get(eid, 0.0)) for eid in edge_ids]
                    simulation_data[point_idx, :] = torch.tensor(queue_vec, dtype=torch.float32)
                    point_idx += 1

            return simulation_data
        finally:
            if self.move_policies_to_device:
                try:
                    self._maybe_move_policies(policies, "cpu", torch)
                except Exception:
                    pass
            self._close_sumo(traci_conn)

    # -------------------------
    # SUMO/libsumo & 路网工具
    # -------------------------
    def _start_sumo(self):
        backend = "libsumo"
        try:
            import libsumo as traci  # type: ignore
            if not hasattr(traci, "start"):
                raise AttributeError("libsumo 缺少 start()，需要回退到 traci")
        except Exception as e_libsumo:  # pragma: no cover
            # 部分环境虽然能导入 libsumo，但缺少启动接口；此时自动回退到 traci。
            try:
                import traci  # type: ignore

                backend = "traci"
            except Exception as e_traci:  # pragma: no cover
                raise ImportError(
                    "无法导入 libsumo 或 traci。请确认已安装 SUMO Python 工具链，"
                    "并把 SUMO 的 tools 目录加入 PYTHONPATH。"
                ) from e_traci

        # 尽量不依赖外部 config.py：全部由 kwargs 传入
        sumo_cfg = Path(self.sumo_config)
        if not sumo_cfg.exists():
            raise FileNotFoundError(f"sumo_config 不存在: {sumo_cfg}")

        sumo_args: list[str] = []
        if self.sumo_binary:
            sumo_args.append(self.sumo_binary)
        else:
            try:
                from sumolib import checkBinary  # type: ignore

                sumo_args.append(checkBinary("sumo-gui" if self.sumo_use_gui else "sumo"))
            except Exception as e:  # pragma: no cover
                raise ImportError("无法导入 sumolib.checkBinary。请确认 SUMO_HOME/tools 已加入 PYTHONPATH。") from e

        # 用户可追加 SUMO 参数
        extra = list(self.extra_sumo_args or [])

        def has_opt(opt: str) -> bool:
            return opt in extra

        # 基础参数（注意：如果 extra 里已经设置了同名 option，则不重复设置）
        sumo_args += ["-c", str(sumo_cfg)]
        if not has_opt("--no-step-log"):
            sumo_args += ["--no-step-log", "true"]
        if not has_opt("--no-warnings"):
            sumo_args += ["--no-warnings", "true"]

        sumo_args += extra

        # 最后做一次去重（SUMO 对重复 option 会报错并退出）
        # 规则：同一个 option（以 '-' 开头的 token）只保留第一次出现及其紧随的 value（若有）。
        deduped = [sumo_args[0]]
        seen: set[str] = set()
        i = 1
        while i < len(sumo_args):
            tok = sumo_args[i]
            if tok.startswith("-"):
                if tok in seen:
                    # 跳过该 option 以及它可能携带的 value
                    i += 1
                    if i < len(sumo_args) and (not sumo_args[i].startswith("-")):
                        i += 1
                    continue
                seen.add(tok)
                deduped.append(tok)
                i += 1
                if i < len(sumo_args) and (not sumo_args[i].startswith("-")):
                    deduped.append(sumo_args[i])
                    i += 1
                continue
            deduped.append(tok)
            i += 1
        sumo_args = deduped

        # libsumo 不支持 traci 的多连接标签模式；当前项目并行依赖多进程（如 Ray worker）隔离实例
        if self.traci_label and backend == "libsumo":
            raise ValueError(
                "当前 SUMO 环境已切换为 libsumo，`traci_label` 不再可用。"
                "如需并行，请使用多进程/多 worker 隔离实例，而不是在单进程内使用 label 多连接。"
            )

        if backend == "traci":
            retries = int(self.sumo_start_retries)
            delay = float(self.sumo_start_retry_delay)
            last_error: Exception | None = None

            for attempt in range(retries + 1):
                label = self._make_traci_label(attempt)
                try:
                    if label:
                        traci.start(sumo_args, label=label)
                        return traci.getConnection(label)

                    traci.start(sumo_args)
                    return traci
                except Exception as e:
                    last_error = e
                    self._cleanup_traci_connection_by_label(traci, label)
                    if attempt >= retries:
                        break
                    if delay > 0.0:
                        time.sleep(delay * float(attempt + 1))

            cmd_preview = " ".join(str(x) for x in sumo_args)
            raise RuntimeError(
                "SUMO 启动失败（traci）。"
                f" 已重试 {retries} 次；最后错误: {last_error}; 命令: {cmd_preview}"
            ) from last_error

        traci.start(sumo_args)
        return traci

    def _close_sumo(self, traci_conn) -> None:
        label = getattr(traci_conn, "_label", None)
        try:
            if hasattr(traci_conn, "close"):
                traci_conn.close()
            else:
                # libsumo / traci 模块级 close
                traci_conn.close()  # type: ignore[attr-defined]
        except Exception:
            # 避免因关闭异常影响上层流程
            pass

        # traci 连接异常关闭时，可能遗留 _connections 项，后续复用同 label 会触发隐性失败。
        try:
            import traci as traci_mod  # type: ignore

            self._cleanup_traci_connection_by_label(traci_mod, label)
        except Exception:
            return None

    def _load_network_and_edges(self, traci_conn) -> tuple[list[str], dict[str, list[str]]]:
        """
        返回：
        - edge_ids: 全部边 ID（用于构造 queue_vector 等）
        - outgoing_map: edge_id -> [outgoing_edge_id, ...]
        """
        # 以 net_file 的 edge 顺序作为“路的顺序”（更稳定，且通常不包含 internal edge）
        edge_ids: list[str] = []
        outgoing_map: dict[str, list[str]] = {}

        if self.net_file:
            try:
                import sumolib  # type: ignore

                net = sumolib.net.readNet(self.net_file)
                edge_ids = [e.getID() for e in net.getEdges() if not str(e.getID()).startswith(":")]
                outgoing_map = {eid: [] for eid in edge_ids}
                for e in net.getEdges():
                    eid = e.getID()
                    if eid not in outgoing_map:
                        continue
                    outs = [oe.getID() for oe in e.getOutgoing()]
                    outgoing_map[eid] = [x for x in outs if x in outgoing_map]
                return edge_ids, outgoing_map
            except Exception:
                pass

        # 退化：使用 libsumo 的 edge 列表（过滤 internal edge）
        edge_ids = [eid for eid in list(traci_conn.edge.getIDList()) if not str(eid).startswith(":")]
        outgoing_map = {eid: [] for eid in edge_ids}
        return edge_ids, outgoing_map

    def _collect_queue_length_map(self, traci_conn, edge_ids: list[str]) -> dict[str, float]:
        # 参考 lane_data_collector：静止车辆数量可作为排队长度 proxy
        m: dict[str, float] = {}
        for eid in edge_ids:
            try:
                m[eid] = float(traci_conn.edge.getLastStepHaltingNumber(eid))
            except Exception:
                m[eid] = 0.0
        return m

    def _collect_edge_length_map(self, traci_conn, edge_ids: list[str]) -> dict[str, float]:
        # 核心：用 libsumo 获取路段长度
        m: dict[str, float] = {}
        for eid in edge_ids:
            try:
                m[eid] = float(traci_conn.edge.getLength(eid))
            except Exception:
                m[eid] = 0.0
        return m

    # -------------------------
    # vehicle_id -> policy_index 映射
    # -------------------------
    def _build_vehicle_id_to_index_map(self, policies: list[Any], traci_conn) -> tuple[dict[str, int], int]:
        """
        返回：
        - vehicle_id_to_index: {vehicle_id: idx}，idx 与 policies 的索引一致
        - dynamic_next_index: 动态分配的下一个 idx（仅当没有 route_file/controlled_vehicle_ids 时使用）

        优先级（与配置注释一致）：
        1) controlled_vehicle_ids：显式指定 vehicle_id 的顺序
        2) route_file：按 rou.xml 中 vehicle 的出现顺序分配 idx
        3) 动态：车辆出现时按出现顺序依次分配 idx
        """
        n = len(policies)
        if n == 0:
            return {}, 0

        if self.controlled_vehicle_ids:
            ids = list(self.controlled_vehicle_ids)[:n]
            mapping = {vid: i for i, vid in enumerate(ids)}
            return mapping, len(mapping)

        if self.route_file and Path(self.route_file).exists():
            ids = self._parse_vehicle_ids_from_route_file(self.route_file)
            # 不强制报错：允许 policies 数量与 route 内车辆数不一致的实验
            ids = ids[:n]
            mapping = {vid: i for i, vid in enumerate(ids)}
            return mapping, len(mapping)

        # 动态：开始时空映射，出现时再分配
        return {}, 0

    def _parse_vehicle_ids_from_route_file(self, route_file: str) -> list[str]:
        tree = ET.parse(route_file)
        root = tree.getroot()
        ids: list[str] = []
        for elem in root.iter():
            if elem.tag == "vehicle":
                vid = elem.get("id")
                if vid:
                    ids.append(vid)
        return ids

    def _vehicle_exists(self, traci_conn, vehicle_id: str) -> bool:
        try:
            return vehicle_id in set(traci_conn.vehicle.getIDList())
        except Exception:
            return False

    def _get_vehicle_destination_edge(self, traci_conn, vehicle_id: str) -> str | None:
        try:
            route = traci_conn.vehicle.getRoute(vehicle_id)
            if not route:
                return None
            return str(route[-1])
        except Exception:
            return None

    def _build_incoming_map(self, outgoing_map: dict[str, list[str]]) -> dict[str, list[str]]:
        """由 outgoing_map 构造反向邻接表：v -> [u...]。"""
        incoming: dict[str, list[str]] = {eid: [] for eid in outgoing_map.keys()}
        for u, outs in outgoing_map.items():
            for v in outs:
                if v in incoming:
                    incoming[v].append(u)
        return incoming

    def _compute_all_distances_to_goal(
        self,
        *,
        goal_edge: str,
        edge_ids: list[str],
        incoming_map: dict[str, list[str]],
        edge_length_map: dict[str, float],
    ) -> list[float]:
        """
        计算“所有路段到 goal_edge 的最短距离向量”（按路段长度）。

        返回：
        - dists: len(edge_ids) 的 list[float]，与 edge_ids 顺序一一对应

        说明：
        - dist[goal_edge] = 0
        - 采用反向 Dijkstra：从 goal 出发沿 incoming 反推
        - 转移代价使用“进入 v 的长度”：dist[u] = min(dist[v] + length(v))
        """
        import heapq

        INF = float("inf")
        dist: dict[str, float] = {goal_edge: 0.0}
        heap: list[tuple[float, str]] = [(0.0, goal_edge)]

        while heap:
            d, v = heapq.heappop(heap)
            if d != dist.get(v, INF):
                continue
            for u in incoming_map.get(v, []):
                nd = d + float(edge_length_map.get(v, 0.0))
                if nd < dist.get(u, INF):
                    dist[u] = nd
                    heapq.heappush(heap, (nd, u))

        return [float(dist.get(eid, INF)) for eid in edge_ids]

    def _plan_route_on_appearance(
        self,
        *,
        policy: Any,
        reward_row: Any | None,
        traci_conn: Any,
        vehicle_id: str,
        start_edge: str,
        dest_edge: str,
        outgoing_map: dict[str, list[str]],
        edge_length_map: dict[str, float],
        edge_id_to_index: dict[str, int],
        dists_to_dest: list[float],
        queue_len_map: dict[str, float],
    ) -> list[str]:
        """
        车辆出现时的路径规划：
        - 有 reward_row：直接走“R + 纯 A*”
        - 无 reward_row 且无 policy：直接 A*（按长度）规划 start->dest
        - 有 policy：不是只在 start 决策一次，而是沿路径逐边执行“DQN 选下一条边 + A* 启发式/回退”
        """
        # 只用 A*（按长度）
        def astar_only() -> list[str]:
            return self._shortest_path_edges_by_edge_length(
                outgoing_map=outgoing_map,
                edge_length_map=edge_length_map,
                start_edge=start_edge,
                goal_edge=dest_edge,
            )

        def reward_astar_only() -> list[str]:
            return self._shortest_path_edges_by_reward_astar(
                outgoing_map=outgoing_map,
                edge_length_map=edge_length_map,
                edge_id_to_index=edge_id_to_index,
                dists_to_dest=dists_to_dest,
                reward_row=reward_row,
                start_edge=start_edge,
                goal_edge=dest_edge,
            )

        candidates = outgoing_map.get(start_edge, [])
        if not candidates:
            return reward_astar_only() if reward_row is not None else astar_only()

        if reward_row is not None:
            return reward_astar_only()

        # 没有可用 DQN：只用 A*
        if policy is None:
            return astar_only()

        # 逐边规划：每一步都重新基于当前 edge 做一次 DQN 选择。
        # A* 不再只在起点后一次性补全整段路径，而是作为启发式距离/失败回退存在。
        route = [start_edge]
        current_edge = start_edge
        max_hops = max(1, len(edge_id_to_index) * 2)
        visited_edges: set[str] = {start_edge}

        for _ in range(max_hops):
            if current_edge == dest_edge:
                return route

            current_candidates = list(outgoing_map.get(current_edge, []))
            if not current_candidates:
                break

            current_i = edge_id_to_index.get(current_edge, -1)
            if current_i < 0:
                break

            state = [current_edge, [float(dists_to_dest[current_i]), float(queue_len_map.get(current_edge, 0.0))]]
            candidate_dists = {
                c: float(dists_to_dest[edge_id_to_index[c]])
                for c in current_candidates
                if c in edge_id_to_index
            }
            candidate_indices = [int(edge_id_to_index[c]) for c in current_candidates if c in edge_id_to_index]

            chosen_next = self._select_action_next_edge(
                policy=policy,
                state=state,
                vehicle_id=vehicle_id,
                current_edge=current_edge,
                destination_edge=dest_edge,
                candidates=current_candidates,
                candidate_dists=candidate_dists,
                candidate_indices=candidate_indices,
            )

            if chosen_next not in current_candidates:
                chosen_next = min(current_candidates, key=lambda e: candidate_dists.get(e, float("inf")))

            # 尽量避免 DQN 落入环；若出现回环，则用启发式最优且未访问的边回退。
            if chosen_next in visited_edges:
                non_cycle_candidates = [c for c in current_candidates if c not in visited_edges]
                if non_cycle_candidates:
                    chosen_next = min(non_cycle_candidates, key=lambda e: candidate_dists.get(e, float("inf")))

            route.append(chosen_next)
            current_edge = chosen_next
            visited_edges.add(chosen_next)

        if current_edge == dest_edge:
            return route

        # 若逐边决策中途失败，则仅从当前边做一次 A* 收尾，避免整条路线直接失效。
        tail = self._shortest_path_edges_by_edge_length(
            outgoing_map=outgoing_map,
            edge_length_map=edge_length_map,
            start_edge=current_edge,
            goal_edge=dest_edge,
        )
        if tail:
            return route[:-1] + tail
        return astar_only()

    def _select_action_next_edge(
        self,
        *,
        policy: Any,
        state: Any,
        vehicle_id: str,
        current_edge: str,
        destination_edge: str,
        candidates: list[str],
        candidate_dists: dict[str, float],
        candidate_indices: list[int] | None = None,
    ) -> str | None:
        """
        兼容不同 policy 形态：
        - policy.select_action(state, candidates=..., info=...) -> action
        - policy.act(state) -> action
        - callable(policy)(state, candidates=..., info=...) -> action

        action 支持：
        - 返回 edge_id（str）
        - 返回候选索引（int）
        """
        info = {
            "vehicle_id": vehicle_id,
            "current_edge": current_edge,
            "destination_edge": destination_edge,
            "candidates": candidates,
            "candidate_dists": candidate_dists,
            # 与 candidates 对齐的“动作索引”（便于 DQN 在全局动作空间上选动作）
            "candidate_indices": candidate_indices,
        }

        action: Any = None
        if hasattr(policy, "select_action") and callable(getattr(policy, "select_action")):
            action = policy.select_action(state, candidates=candidates, info=info)
        elif hasattr(policy, "act") and callable(getattr(policy, "act")):
            action = policy.act(state)
        elif callable(policy):
            action = policy(state, candidates=candidates, info=info)
        else:
            return None

        if isinstance(action, str):
            return action
        if isinstance(action, int):
            if 0 <= action < len(candidates):
                return candidates[action]
        return None

    def _shortest_path_edges_by_reward_astar(
        self,
        *,
        outgoing_map: dict[str, list[str]],
        edge_length_map: dict[str, float],
        edge_id_to_index: dict[str, int],
        dists_to_dest: list[float],
        reward_row: Any | None,
        start_edge: str,
        goal_edge: str,
    ) -> list[str]:
        """
        使用奖励 R 对边代价进行重加权后，执行纯 A* 路径规划。

        代价设计：
        - 先将路段长度按“全网平均边长”归一化，避免绝对长度过大压制奖励；
        - 再叠加一个由奖励决定的非负惩罚项：reward 越大，惩罚越小；
        - 整体代价始终为正，且 heuristic 仍可安全使用“纯长度下界”。
        """
        if start_edge == goal_edge:
            return [start_edge]

        import heapq

        weight = max(0.0, float(self.reward_astar_weight))
        if weight <= 0.0:
            return self._shortest_path_edges_by_edge_length(
                outgoing_map=outgoing_map,
                edge_length_map=edge_length_map,
                start_edge=start_edge,
                goal_edge=goal_edge,
            )

        num_road = len(edge_id_to_index)
        reward_values = [self._reward_value_for_edge(reward_row, edge_index) for edge_index in range(num_road)]
        max_reward = max(reward_values) if reward_values else 0.0
        min_reward = min(reward_values) if reward_values else 0.0
        reward_span = max_reward - min_reward

        positive_lengths = [float(v) for v in edge_length_map.values() if float(v) > 0.0]
        mean_edge_length = (
            sum(positive_lengths) / float(len(positive_lengths))
            if positive_lengths
            else 1.0
        )
        mean_edge_length = max(mean_edge_length, 1.0e-6)

        def edge_cost(edge_id: str) -> float:
            base = float(edge_length_map.get(edge_id, 0.0))
            if base <= 0.0:
                base = 1.0e-6
            edge_idx = edge_id_to_index.get(edge_id)
            if edge_idx is None:
                return max(base / mean_edge_length, 1.0e-6)
            reward_value = self._reward_value_for_edge(reward_row, edge_idx)
            if reward_span > 1.0e-12:
                reward_ratio = (reward_value - min_reward) / reward_span
            else:
                reward_ratio = (reward_value / max_reward) if max_reward > 1.0e-12 else 0.0

            normalized_length = base / mean_edge_length
            reward_penalty = weight * (1.0 - reward_ratio)
            return max(normalized_length + reward_penalty, 1.0e-6)

        def heuristic(edge_id: str) -> float:
            edge_idx = edge_id_to_index.get(edge_id)
            if edge_idx is None:
                return 0.0
            try:
                base_dist = float(dists_to_dest[edge_idx])
            except Exception:
                return 0.0
            if base_dist == float("inf"):
                return 0.0
            return max(base_dist / mean_edge_length, 0.0)

        INF = float("inf")
        g_score: dict[str, float] = {start_edge: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, float, str]] = [(heuristic(start_edge), 0.0, start_edge)]

        while heap:
            _, g_cur, u = heapq.heappop(heap)
            if g_cur != g_score.get(u, INF):
                continue
            if u == goal_edge:
                break
            for v in outgoing_map.get(u, []):
                nd = g_cur + edge_cost(v)
                if nd < g_score.get(v, INF):
                    g_score[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd + heuristic(v), nd, v))

        if goal_edge not in g_score:
            return []

        path_rev = [goal_edge]
        cur = goal_edge
        while cur in prev:
            cur = prev[cur]
            path_rev.append(cur)
        path = list(reversed(path_rev))
        if path and path[0] == start_edge:
            return path
        return []

    def _try_set_vehicle_route(self, traci_conn, vehicle_id: str, route_edges: list[str]) -> None:
        try:
            cur_route = list(traci_conn.vehicle.getRoute(vehicle_id))
            if cur_route == route_edges:
                return None
            if len(route_edges) >= 1:
                traci_conn.vehicle.setRoute(vehicle_id, route_edges)
        except Exception:
            return None

    # -------------------------
    # 最短路（边为节点的图）：按 edge length
    # -------------------------
    def _shortest_path_distance_by_edge_length(
        self,
        *,
        outgoing_map: dict[str, list[str]],
        edge_length_map: dict[str, float],
        start_edge: str,
        goal_edge: str,
    ) -> float:
        """
        返回从 start_edge 到 goal_edge 的最短距离（按 edge_length_map）。
        不可达返回 inf。
        """
        if start_edge == goal_edge:
            return 0.0

        import heapq

        INF = float("inf")
        dist: dict[str, float] = {start_edge: 0.0}
        heap: list[tuple[float, str]] = [(0.0, start_edge)]

        while heap:
            d, u = heapq.heappop(heap)
            if d != dist.get(u, INF):
                continue
            if u == goal_edge:
                return d
            for v in outgoing_map.get(u, []):
                w = float(edge_length_map.get(v, 0.0))
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return INF

    def _shortest_path_edges_by_edge_length(
        self,
        *,
        outgoing_map: dict[str, list[str]],
        edge_length_map: dict[str, float],
        start_edge: str,
        goal_edge: str,
    ) -> list[str]:
        """
        返回从 start_edge 到 goal_edge 的最短路径（edge_id 列表，包含 start_edge 作为首元素）。
        不可达返回空列表。
        """
        if start_edge == goal_edge:
            return [start_edge]

        import heapq

        INF = float("inf")
        dist: dict[str, float] = {start_edge: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, start_edge)]

        while heap:
            d, u = heapq.heappop(heap)
            if d != dist.get(u, INF):
                continue
            if u == goal_edge:
                break
            for v in outgoing_map.get(u, []):
                w = float(edge_length_map.get(v, 0.0))
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if goal_edge not in dist:
            return []

        # 回溯
        path_rev = [goal_edge]
        cur = goal_edge
        while cur in prev:
            cur = prev[cur]
            path_rev.append(cur)
        path = list(reversed(path_rev))
        if path and path[0] == start_edge:
            return path
        return []

