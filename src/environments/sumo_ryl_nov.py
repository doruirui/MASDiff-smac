from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.environments.sumo_ryl import SumoRylEnvironment


class SumoRylNovEnvironment(SumoRylEnvironment):
    """
    基于 `SumoRylEnvironment` 的“道路过车数”版本。

    设计目标：
    - 严禁改动已有 `SumoRylEnvironment`；
    - 复用其基于 libsumo 的 SUMO 启停、路网解析、A* 路径规划、vehicle->policy 映射等逻辑；
    - 仅把原实现中的“道路排队长度”特征/评估量替换为“道路累计过车数”。

    这里的“道路过车数”定义为：
    - 仿真过程中，车辆每次**进入某条非 internal edge** 时，
      对该 edge 的计数 +1；
    - 因此该值是“累计进入该道路的车辆数”。

    对应输出变化：
    - tau.shape == [num_car, num_road, 2]
      - [:, :, 0] = 当前路到终点的最短距离
      - [:, :, 1] = 当前时刻各道路累计过车数
    - simulation_data.shape == [end_tick / sample_interval, num_road]
      - 每一行是采样时刻的全网累计过车数向量
    """

    def _update_edge_pass_count_map(
        self,
        sim_conn: Any,
        *,
        edge_id_set: set[str],
        edge_pass_count_map: dict[str, float],
        last_counted_edge_by_vehicle: dict[str, str],
    ) -> set[str]:
        """
        更新“道路累计过车数”并返回当前车辆集合。

        规则：
        - 若车辆当前位于非 internal edge，且与上次已计数的 edge 不同，
          则说明它新进入了该 edge，对该 edge 计数 +1。
        - internal edge（例如 `:xxx`）不计入统计。
        """
        try:
            current_vehicle_ids = list(sim_conn.vehicle.getIDList())
        except Exception:
            return set()

        current_vehicles = set(current_vehicle_ids)

        for veh_id in current_vehicle_ids:
            try:
                edge_id = str(sim_conn.vehicle.getRoadID(veh_id))
            except Exception:
                continue

            if (not edge_id) or edge_id.startswith(":") or (edge_id not in edge_id_set):
                continue

            prev_edge = last_counted_edge_by_vehicle.get(veh_id)
            if prev_edge != edge_id:
                edge_pass_count_map[edge_id] = float(edge_pass_count_map.get(edge_id, 0.0)) + 1.0
                last_counted_edge_by_vehicle[veh_id] = edge_id

        # 清理已离开仿真的车辆，避免状态字典无限增长
        for veh_id in list(last_counted_edge_by_vehicle.keys()):
            if veh_id not in current_vehicles:
                last_counted_edge_by_vehicle.pop(veh_id, None)

        return current_vehicles

    def _normalize_route_edges(self, route_edges: list[Any] | tuple[Any, ...] | None) -> list[str]:
        if not route_edges:
            return []
        return [str(x) for x in route_edges if str(x)]

    def _write_planned_routes_to_rou_file(
        self,
        *,
        output_rou_path: str,
        planned_routes_by_vehicle: dict[str, list[str]],
    ) -> str:
        if not self.route_file:
            raise ValueError("当前环境未配置 route_file，无法导出新的 rou 文件")

        src_route_file = Path(self.route_file)
        if not src_route_file.exists():
            raise FileNotFoundError(f"route_file 不存在：{src_route_file}")

        tree = ET.parse(str(src_route_file))
        root = tree.getroot()

        route_defs_by_id: dict[str, ET.Element] = {}
        for elem in root.findall("route"):
            rid = elem.get("id")
            if rid:
                route_defs_by_id[str(rid)] = elem

        for vehicle_elem in root.iter("vehicle"):
            veh_id = vehicle_elem.get("id")
            if not veh_id:
                continue

            planned_route = planned_routes_by_vehicle.get(str(veh_id))
            if not planned_route:
                continue

            route_child = vehicle_elem.find("route")
            if route_child is not None:
                route_child.set("edges", " ".join(planned_route))
                continue

            route_ref = vehicle_elem.get("route")
            if route_ref and route_ref in route_defs_by_id:
                vehicle_elem.attrib.pop("route", None)
                new_route_elem = ET.Element("route")
                new_route_elem.set("edges", " ".join(planned_route))
                vehicle_elem.append(new_route_elem)
                continue

            new_route_elem = ET.Element("route")
            new_route_elem.set("edges", " ".join(planned_route))
            vehicle_elem.append(new_route_elem)

        out_path = Path(output_rou_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
        return str(out_path)

    def simulate_collect(self, planner_input: Any) -> tuple[list[list[Any]], Any]:
        """
        用给定策略仿真一次，收集经验库与 Tau（奖励先留空）。

        与父类区别：
        - 原实现使用“当前路排队长度”作为第二维特征；
        - 本实现改为“当前时刻各道路累计过车数”。
        """
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("SumoRylNovEnvironment 需要 torch 以返回张量 tau。请先安装 torch。") from e

        sim_conn = self._start_sumo()
        try:
            policies, rewards = self._split_planner_input(planner_input)
            if self.move_policies_to_device:
                infer_dev = self._resolve_policy_device(torch)
                self._maybe_move_policies(policies, infer_dev, torch)

            edge_ids, outgoing_map = self._load_network_and_edges(sim_conn)
            num_road = len(edge_ids)
            edge_id_set = set(edge_ids)
            edge_id_to_index = {e: i for i, e in enumerate(edge_ids)}
            edge_length_map = self._collect_edge_length_map(sim_conn, edge_ids)
            incoming_map = self._build_incoming_map(outgoing_map)

            num_car = len(policies)
            experience_buffers: list[list[Any]] = [[] for _ in range(num_car)]

            tau = torch.zeros((num_car, num_road, 2), dtype=torch.float32)
            tau[:, :, 0] = float("inf")

            vehicle_id_to_index, dynamic_next_index = self._build_vehicle_id_to_index_map(policies, sim_conn)

            edge_connections: list[tuple[str, str]] = []
            for u, outs in outgoing_map.items():
                if u not in edge_id_to_index:
                    continue
                for v in outs:
                    if v in edge_id_to_index:
                        edge_connections.append((u, v))

            dist_cache: dict[str, list[float]] = {}
            previous_vehicles: set[str] = set()
            filled_indices: set[int] = set()
            planned_vehicle_ids: set[str] = set()

            edge_pass_count_map: dict[str, float] = {eid: 0.0 for eid in edge_ids}
            last_counted_edge_by_vehicle: dict[str, str] = {}

            tick = 0
            while tick < self.end_tick:
                tick += 1
                sim_conn.simulationStep()

                current_vehicles = self._update_edge_pass_count_map(
                    sim_conn,
                    edge_id_set=edge_id_set,
                    edge_pass_count_map=edge_pass_count_map,
                    last_counted_edge_by_vehicle=last_counted_edge_by_vehicle,
                )
                new_vehicles = current_vehicles - previous_vehicles

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
                    if car_idx in filled_indices:
                        continue

                    dest_edge = self._get_vehicle_destination_edge(sim_conn, veh_id)
                    if not dest_edge or dest_edge not in edge_id_to_index:
                        continue

                    if dest_edge not in dist_cache:
                        dist_cache[dest_edge] = self._compute_all_distances_to_goal(
                            goal_edge=dest_edge,
                            edge_ids=edge_ids,
                            incoming_map=incoming_map,
                            edge_length_map=edge_length_map,
                        )
                    dists = dist_cache[dest_edge]

                    pass_count_vec = [float(edge_pass_count_map.get(eid, 0.0)) for eid in edge_ids]

                    tau[car_idx, :, 0] = torch.tensor(dists, dtype=torch.float32)
                    tau[car_idx, :, 1] = torch.tensor(pass_count_vec, dtype=torch.float32)
                    filled_indices.add(car_idx)

                    for u, v in edge_connections:
                        u_i = edge_id_to_index[u]
                        v_i = edge_id_to_index[v]
                        s = [u, [float(dists[u_i]), float(pass_count_vec[u_i])]]
                        a = int(v_i)
                        experience_buffers[car_idx].append([s, a, None])

                    start_edge = sim_conn.vehicle.getRoadID(veh_id)
                    if start_edge and (start_edge in edge_id_to_index) and start_edge != dest_edge:
                        planned_route = self._plan_route_on_appearance(
                            policy=policies[car_idx] if car_idx < len(policies) else None,
                            reward_row=self._get_reward_row(rewards, car_idx),
                            traci_conn=sim_conn,
                            vehicle_id=veh_id,
                            start_edge=start_edge,
                            dest_edge=dest_edge,
                            outgoing_map=outgoing_map,
                            edge_length_map=edge_length_map,
                            edge_id_to_index=edge_id_to_index,
                            dists_to_dest=dists,
                            # 复用父类接口；这里传入的是“累计过车数 map”
                            queue_len_map=edge_pass_count_map,
                        )
                        if planned_route:
                            self._try_set_vehicle_route(sim_conn, veh_id, planned_route)

                previous_vehicles = current_vehicles

            return experience_buffers, tau
        finally:
            if self.move_policies_to_device:
                try:
                    self._maybe_move_policies(policies, "cpu", torch)
                except Exception:
                    pass
            self._close_sumo(sim_conn)

    def simulate_evaluate(self, planner_input: Any) -> Any:
        """
        用给定策略仿真一次，返回用于评估的 simulation_data 张量。

        与父类区别：
        - 原实现记录采样时刻的“全网排队长度向量”；
        - 本实现记录采样时刻的“全网累计过车数向量”。
        """
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("SumoRylNovEnvironment 需要 torch 以返回张量 simulation_data。请先安装 torch。") from e

        sim_conn = self._start_sumo()
        try:
            policies, rewards = self._split_planner_input(planner_input)
            if self.move_policies_to_device:
                infer_dev = self._resolve_policy_device(torch)
                self._maybe_move_policies(policies, infer_dev, torch)

            edge_ids, outgoing_map = self._load_network_and_edges(sim_conn)
            num_road = len(edge_ids)
            edge_id_set = set(edge_ids)
            edge_id_to_index = {e: i for i, e in enumerate(edge_ids)}
            edge_length_map = self._collect_edge_length_map(sim_conn, edge_ids)
            incoming_map = self._build_incoming_map(outgoing_map)

            num_car = len(policies)
            vehicle_id_to_index, dynamic_next_index = self._build_vehicle_id_to_index_map(policies, sim_conn)
            dist_cache: dict[str, list[float]] = {}

            num_points = self.end_tick // self.sample_interval
            simulation_data = torch.zeros((num_points, num_road), dtype=torch.float32)
            point_idx = 0

            previous_vehicles: set[str] = set()
            edge_pass_count_map: dict[str, float] = {eid: 0.0 for eid in edge_ids}
            last_counted_edge_by_vehicle: dict[str, str] = {}
            planned_vehicle_ids: set[str] = set()

            tick = 0
            while tick < self.end_tick:
                tick += 1
                sim_conn.simulationStep()

                current_vehicles = self._update_edge_pass_count_map(
                    sim_conn,
                    edge_id_set=edge_id_set,
                    edge_pass_count_map=edge_pass_count_map,
                    last_counted_edge_by_vehicle=last_counted_edge_by_vehicle,
                )
                new_vehicles = current_vehicles - previous_vehicles

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

                    dest_edge = self._get_vehicle_destination_edge(sim_conn, veh_id)
                    if not dest_edge:
                        continue

                    if dest_edge not in dist_cache and dest_edge in edge_id_set:
                        dist_cache[dest_edge] = self._compute_all_distances_to_goal(
                            goal_edge=dest_edge,
                            edge_ids=edge_ids,
                            incoming_map=incoming_map,
                            edge_length_map=edge_length_map,
                        )

                    start_edge = sim_conn.vehicle.getRoadID(veh_id)
                    if not start_edge or start_edge == dest_edge:
                        continue

                    if dest_edge in dist_cache:
                        planned_route = self._plan_route_on_appearance(
                            policy=policies[car_idx] if car_idx < len(policies) else None,
                            reward_row=self._get_reward_row(rewards, car_idx),
                            traci_conn=sim_conn,
                            vehicle_id=veh_id,
                            start_edge=start_edge,
                            dest_edge=dest_edge,
                            outgoing_map=outgoing_map,
                            edge_length_map=edge_length_map,
                            edge_id_to_index=edge_id_to_index,
                            dists_to_dest=dist_cache[dest_edge],
                            # 复用父类接口；这里传入的是“累计过车数 map”
                            queue_len_map=edge_pass_count_map,
                        )
                        if planned_route:
                            self._try_set_vehicle_route(sim_conn, veh_id, planned_route)

                previous_vehicles = current_vehicles

                if tick % self.sample_interval == 0 and point_idx < simulation_data.shape[0]:
                    pass_count_vec = [float(edge_pass_count_map.get(eid, 0.0)) for eid in edge_ids]
                    simulation_data[point_idx, :] = torch.tensor(pass_count_vec, dtype=torch.float32)
                    point_idx += 1

            return simulation_data
        finally:
            if self.move_policies_to_device:
                try:
                    self._maybe_move_policies(policies, "cpu", torch)
                except Exception:
                    pass
            self._close_sumo(sim_conn)

    def simulate_evaluate_with_route_export(self, planner_input: Any, *, output_rou_path: str) -> str:
        """
        使用给定策略再仿真一次，并把所有车辆“路径规划后的最终路径”导出为新的 rou 文件。

        返回：
        - 写出的 rou 文件路径
        """
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("SumoRylNovEnvironment 需要 torch。请先安装 torch。") from e

        sim_conn = self._start_sumo()
        try:
            policies, rewards = self._split_planner_input(planner_input)
            if self.move_policies_to_device:
                infer_dev = self._resolve_policy_device(torch)
                self._maybe_move_policies(policies, infer_dev, torch)

            edge_ids, outgoing_map = self._load_network_and_edges(sim_conn)
            edge_id_set = set(edge_ids)
            edge_id_to_index = {e: i for i, e in enumerate(edge_ids)}
            edge_length_map = self._collect_edge_length_map(sim_conn, edge_ids)
            incoming_map = self._build_incoming_map(outgoing_map)

            num_car = len(policies)
            vehicle_id_to_index, dynamic_next_index = self._build_vehicle_id_to_index_map(policies, sim_conn)
            dist_cache: dict[str, list[float]] = {}

            previous_vehicles: set[str] = set()
            edge_pass_count_map: dict[str, float] = {eid: 0.0 for eid in edge_ids}
            last_counted_edge_by_vehicle: dict[str, str] = {}
            planned_routes_by_vehicle: dict[str, list[str]] = {}
            planned_vehicle_ids: set[str] = set()

            tick = 0
            while tick < self.end_tick:
                tick += 1
                sim_conn.simulationStep()

                current_vehicles = self._update_edge_pass_count_map(
                    sim_conn,
                    edge_id_set=edge_id_set,
                    edge_pass_count_map=edge_pass_count_map,
                    last_counted_edge_by_vehicle=last_counted_edge_by_vehicle,
                )
                new_vehicles = current_vehicles - previous_vehicles

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

                    try:
                        current_route = self._normalize_route_edges(list(sim_conn.vehicle.getRoute(veh_id)))
                    except Exception:
                        current_route = []

                    dest_edge = self._get_vehicle_destination_edge(sim_conn, veh_id)
                    start_edge = sim_conn.vehicle.getRoadID(veh_id)

                    if not start_edge:
                        if current_route:
                            planned_routes_by_vehicle[str(veh_id)] = current_route
                        continue

                    if not dest_edge or start_edge == dest_edge:
                        fallback_route = current_route if current_route else [str(start_edge)]
                        planned_routes_by_vehicle[str(veh_id)] = self._normalize_route_edges(fallback_route)
                        continue

                    if dest_edge not in dist_cache and dest_edge in edge_id_set:
                        dist_cache[dest_edge] = self._compute_all_distances_to_goal(
                            goal_edge=dest_edge,
                            edge_ids=edge_ids,
                            incoming_map=incoming_map,
                            edge_length_map=edge_length_map,
                        )

                    if dest_edge in dist_cache and start_edge in edge_id_to_index:
                        planned_route = self._plan_route_on_appearance(
                            policy=policies[car_idx] if car_idx < len(policies) else None,
                            reward_row=self._get_reward_row(rewards, car_idx),
                            traci_conn=sim_conn,
                            vehicle_id=veh_id,
                            start_edge=start_edge,
                            dest_edge=dest_edge,
                            outgoing_map=outgoing_map,
                            edge_length_map=edge_length_map,
                            edge_id_to_index=edge_id_to_index,
                            dists_to_dest=dist_cache[dest_edge],
                            queue_len_map=edge_pass_count_map,
                        )
                        normalized_route = self._normalize_route_edges(planned_route)
                        if normalized_route:
                            planned_routes_by_vehicle[str(veh_id)] = normalized_route
                            self._try_set_vehicle_route(sim_conn, veh_id, normalized_route)
                        else:
                            fallback_route = current_route if current_route else [str(start_edge)]
                            planned_routes_by_vehicle[str(veh_id)] = self._normalize_route_edges(fallback_route)
                    else:
                        fallback_route = current_route if current_route else [str(start_edge)]
                        planned_routes_by_vehicle[str(veh_id)] = self._normalize_route_edges(fallback_route)

                previous_vehicles = current_vehicles

            return self._write_planned_routes_to_rou_file(
                output_rou_path=output_rou_path,
                planned_routes_by_vehicle=planned_routes_by_vehicle,
            )
        finally:
            if self.move_policies_to_device:
                try:
                    self._maybe_move_policies(policies, "cpu", torch)
                except Exception:
                    pass
            self._close_sumo(sim_conn)
