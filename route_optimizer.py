import csv
import heapq
import math
from typing import Callable, Dict, List, Optional, Tuple

Coord = Tuple[int, int]

WATER_BIOMES = {
    "ocean", "deep_ocean", "warm_ocean", "lukewarm_ocean", "cold_ocean",
    "deep_lukewarm_ocean", "deep_cold_ocean", "deep_frozen_ocean", "river"
}
WATER_COST_SEC_PER_CHUNK = 7.30
UNKNOWN_OVERWORLD_FALLBACK = 4.20
UNKNOWN_NETHER_FALLBACK = 4.50


def load_overworld_cost_model(csv_path: str) -> Tuple[Dict[str, float], float]:
    cost_map: Dict[str, float] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cost_map[row["biome"]] = float(row["risk_cost_robust"])
    for wb in WATER_BIOMES:
        cost_map[wb] = WATER_COST_SEC_PER_CHUNK
    return cost_map, min(cost_map.values())


def load_nether_cost_model(csv_path: str) -> Tuple[Dict[str, float], float]:
    cost_map: Dict[str, float] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cost_map[row["biome"]] = float(row["risk_cost_robust"])
    # Brakujący w pomiarach crimson_forest ma tę samą geometrię terenu co warped_forest
    if "crimson_forest" not in cost_map and "warped_forest" in cost_map:
        cost_map["crimson_forest"] = cost_map["warped_forest"]
    return cost_map, min(cost_map.values())


class RouteOptimizer:
    def __init__(
        self,
        cost_map: Dict[str, float],
        tau_min: float,
        biome_sampler: Callable[[int, int], str],
        step: int = 16,
        fallback_cost: float = UNKNOWN_OVERWORLD_FALLBACK
    ):
        self.cost_map = cost_map
        self.tau_min = tau_min
        self.biome_sampler = biome_sampler
        self.step = step
        self.fallback_cost = fallback_cost
        self.biome_cache: Dict[Tuple[int, int], float] = {}
        self.segment_cache: Dict[Tuple[Coord, Coord], Tuple[float, List[Coord]]] = {}

    def _get_node_cost(self, gx: int, gz: int) -> float:
        if (gx, gz) not in self.biome_cache:
            wx = gx * self.step + (self.step // 2)
            wz = gz * self.step + (self.step // 2)
            b_name = self.biome_sampler(wx, wz)
            self.biome_cache[(gx, gz)] = self.cost_map.get(b_name, self.fallback_cost)
        return self.biome_cache[(gx, gz)]

    def _to_grid(self, pos: Coord) -> Tuple[int, int]:
        return (pos[0] // self.step, pos[1] // self.step)

    def _to_world(self, gpos: Tuple[int, int]) -> Coord:
        return (gpos[0] * self.step + self.step // 2, gpos[1] * self.step + self.step // 2)

    def find_segment_astar(self, start: Coord, goal: Coord) -> Tuple[float, List[Coord]]:
        if (start, goal) in self.segment_cache:
            return self.segment_cache[(start, goal)]

        s_node = self._to_grid(start)
        g_node = self._to_grid(goal)
        sec_per_block_min = self.tau_min / 16.0

        open_set = [(0.0, s_node)]
        came_from = {}
        g_score = {s_node: 0.0}
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == g_node:
                path = [goal]
                curr = current
                while curr in came_from:
                    path.append(self._to_world(curr))
                    curr = came_from[curr]
                path.append(start)
                path.reverse()
                res = (g_score[current], path)
                self.segment_cache[(start, goal)] = res
                return res

            c_cost = self._get_node_cost(*current) / 16.0
            for dx, dz in neighbors:
                nxt = (current[0] + dx, current[1] + dz)
                dist = math.hypot(dx * self.step, dz * self.step)
                n_cost = self._get_node_cost(*nxt) / 16.0
                tentative_g = g_score[current] + dist * 0.5 * (c_cost + n_cost)

                if tentative_g < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    h = math.hypot((g_node[0] - nxt[0]) * self.step, (g_node[1] - nxt[1]) * self.step) * sec_per_block_min
                    heapq.heappush(open_set, (tentative_g + h, nxt))

        return float("inf"), []


def evaluate_full_scenario(
    nodes_by_type: Dict[str, List[Dict]],
    scenario: List[str],
    ow_opt: RouteOptimizer,
    net_opt: RouteOptimizer
) -> Optional[Dict]:
    """
    Ocenia pojedynczy 6-etapowy scenariusz z valid_path.py:
    [Spawn, Early_Struct, Lava] (OW) -> Portal 1:8 -> [Net_1, Net_2] (Nether) -> Portal 8:1 -> [Stronghold] (OW)
    """
    for req_type in scenario:
        if req_type not in nodes_by_type or len(nodes_by_type[req_type]) == 0:
            return None

    # Budowa warstw z obsługą przejść między wymiarami
    layers = [nodes_by_type[t] for t in scenario]
    dp = [{i: float("inf") for i in range(len(layer))} for layer in layers]
    parent = [{} for _ in layers]
    paths = [{} for _ in layers]
    dp[0][0] = 0.0

    for l in range(1, len(layers)):
        prev_layer = layers[l - 1]
        curr_layer = layers[l]

        for u_idx, u_node in enumerate(prev_layer):
            if dp[l - 1][u_idx] == float("inf"):
                continue

            u_dim = u_node["dimension"]
            ux, uz = int(u_node["x"]), int(u_node["z"])

            for v_idx, v_node in enumerate(curr_layer):
                v_dim = v_node["dimension"]
                vx, vz = int(v_node["x"]), int(v_node["z"])

                # Wybór odpowiedniego optymalizatora i przeliczenie współrzędnych portalu
                if u_dim == "Overworld" and v_dim == "Overworld":
                    seg_time, seg_path = ow_opt.find_segment_astar((ux, uz), (vx, vz))
                elif u_dim == "Overworld" and v_dim == "Nether":
                    # Wejście do Netheru przez portal na (x // 8, z // 8)
                    portal_net = (ux // 8, uz // 8)
                    seg_time, seg_path = net_opt.find_segment_astar(portal_net, (vx, vz))
                elif u_dim == "Nether" and v_dim == "Nether":
                    seg_time, seg_path = net_opt.find_segment_astar((ux, uz), (vx, vz))
                elif u_dim == "Nether" and v_dim == "Overworld":
                    # Powrót do Overworldu przez portal na (x * 8, z * 8)
                    portal_ow = (ux * 8, uz * 8)
                    seg_time, seg_path = ow_opt.find_segment_astar(portal_ow, (vx, vz))
                else:
                    continue

                total_time = dp[l - 1][u_idx] + seg_time
                if total_time < dp[l][v_idx]:
                    dp[l][v_idx] = total_time
                    parent[l][v_idx] = u_idx
                    paths[l][v_idx] = seg_path

    last_idx = len(layers) - 1
    best_end_idx = min(dp[last_idx], key=dp[last_idx].get)
    best_time = dp[last_idx][best_end_idx]
    if best_time == float("inf"):
        return None

    chosen_nodes = []
    ow_waypoints: List[Coord] = []
    net_waypoints: List[Coord] = []
    curr_idx = best_end_idx

    for l in range(last_idx, 0, -1):
        node = layers[l][curr_idx]
        chosen_nodes.append(node)
        seg = paths[l][curr_idx]
        if node["dimension"] == "Nether":
            net_waypoints = seg + net_waypoints
        else:
            ow_waypoints = seg + ow_waypoints
        curr_idx = parent[l][curr_idx]

    chosen_nodes.append(layers[0][0])
    chosen_nodes.reverse()

    return {
        "scenario": scenario,
        "total_time_sec": round(best_time, 2),
        "chosen_sequence": chosen_nodes,
        "overworld_waypoints": ow_waypoints,
        "nether_waypoints": net_waypoints
    }
