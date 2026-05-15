"""Builds a heuristic transport network for the Selenia City CodinGame puzzle."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import isqrt
import sys

MAX_MONTHS = 20
MONTH_DAYS = 20
MAX_TUBES_PER_BUILDING = 5
MAX_PODS = 500
POD_COST = 1000
TELEPORT_COST = 5000
DEBUG_LIST_LIMIT = 18


@dataclass(slots=True)
class Building:
    """Stores one building with its identifier, type, coordinates, and monthly landing-pad demand."""
    id: int
    kind: int
    x: int
    y: int
    demand: Counter[int] = field(default_factory=Counter)


@dataclass(slots=True)
class Pod:
    """Stores one transport pod identifier and ordered itinerary."""
    id: int
    path: list[int]


@dataclass(slots=True)
class Candidate:
    """Stores one possible infrastructure action bundle with its cost, score, route work, and estimated deliveries."""
    score: int
    cost: int
    pad_id: int
    module_id: int
    astronaut_type: int
    path: list[int] = field(default_factory=list)
    tubes: list[tuple[int, int]] = field(default_factory=list)
    upgrades: list[tuple[int, int]] = field(default_factory=list)
    teleport: tuple[int, int] | None = None
    delivered: int = 0

    @property
    def efficiency(self) -> float:
        """Returns the score produced per resource spent by this bundle."""
        return self.score / max(1, self.cost)


class Planner:
    """Maintains city memory and chooses monthly construction actions from buildings, routes, pods, and resources."""
    buildings: dict[int, Building]
    month: int
    resources: int
    tubes: dict[tuple[int, int], int]
    teleports: dict[int, int]
    pods: dict[int, Pod]

    def __init__(self):
        """Initializes empty persistent city state before the first input month."""
        self.buildings = {}
        self.month = 0
        self.resources = 0
        self.tubes = {}
        self.teleports = {}
        self.pods = {}

    def play(self):
        """Runs the interactive month loop and prints one action line per month."""
        while True:
            try:
                self.read_month()
            except EOFError:
                return
            print(";".join(self.choose_actions()) or "WAIT")
            self.month += 1

    def read_month(self):
        """Reads one monthly city snapshot and updates persistent building memory."""
        self.resources = int(input())
        self.tubes = {}
        self.teleports = {}
        route_count = int(input())
        for _ in range(route_count):
            a, b, capacity = map(int, input().split())
            if capacity == 0:
                self.teleports[a] = b
            else:
                self.tubes[route_key(a, b)] = capacity

        self.pods = {}
        pod_count = int(input())
        for _ in range(pod_count):
            parts = list(map(int, input().split()))
            self.pods[parts[0]] = Pod(parts[0], parts[2:])

        new_buildings = []
        for _ in range(int(input())):
            parts = list(map(int, input().split()))
            if parts[0] == 0:
                building = Building(parts[1], 0, parts[2], parts[3], Counter(parts[5:]))
            else:
                building = Building(parts[1], parts[0], parts[2], parts[3])
            self.buildings[building.id] = building
            new_buildings.append(building)
        self.debug_month_input(route_count, pod_count, new_buildings)

    def choose_actions(self) -> list[str]:
        """Chooses and returns semicolon-separated action fragments for the current month."""
        actions = []
        serviced = self.get_serviced_pairs()
        module_load = self.get_module_load()
        degrees = self.get_tube_degrees()
        teleport_used = self.get_teleport_used_buildings()
        teleported_pairs = self.get_teleported_pairs()
        pod_ids = set(self.pods)
        tubes = dict(self.tubes)
        direct_pod_counts = self.get_direct_service_pod_counts()
        budget = self.resources
        modules_by_type = self.get_modules_by_type()
        unserved_demands = self.get_unserved_demands(serviced)
        min_efficiency = self.min_efficiency()
        self.debug_plan_start(serviced, module_load, degrees, direct_pod_counts, unserved_demands, min_efficiency)

        for demand_ind, (pad, astronaut_type, count) in enumerate(unserved_demands, 1):
            candidate = self.best_service_candidate(pad, astronaut_type, count, modules_by_type, module_load, degrees, teleport_used, tubes, budget, pod_ids)
            prefix = f"[M{self.month + 1:02d}] demand {demand_ind}/{len(unserved_demands)} {describe_demand(pad, astronaut_type, count)}"
            if candidate is None:
                print(f"{prefix} no_candidate budget={budget}", file=sys.stderr)
            elif candidate.efficiency < min_efficiency:
                print(f"{prefix} skip best={describe_candidate(candidate)} min_eff={min_efficiency:.2f}", file=sys.stderr)
            else:
                print(f"{prefix} choose best={describe_candidate(candidate)}", file=sys.stderr)
                budget = self.apply_candidate("service", candidate, actions, serviced, module_load, degrees, teleport_used, tubes, direct_pod_counts,
                                              budget, pod_ids)
                if candidate.teleport is not None:
                    teleported_pairs.add((candidate.pad_id, candidate.astronaut_type))

        speed_round = 0
        while True:
            speed_budget = self.speed_spend_budget(budget)
            speed_pick = self.best_speed_candidate(serviced, tubes, direct_pod_counts, teleport_used, teleported_pairs, speed_budget, pod_ids)
            if speed_budget <= 0:
                print(f"[M{self.month + 1:02d}] speed stop reserve budget={budget} reserve={self.reserve_floor()} rounds={speed_round}", file=sys.stderr)
                break
            if speed_pick is None:
                print(f"[M{self.month + 1:02d}] speed stop no_candidate budget={budget} spendable={speed_budget} rounds={speed_round}", file=sys.stderr)
                break
            reason, candidate = speed_pick
            if candidate.score <= 0:
                print(f"[M{self.month + 1:02d}] speed stop non_positive best={describe_candidate(candidate)}", file=sys.stderr)
                break
            min_speed_efficiency = self.min_speed_efficiency()
            if candidate.efficiency < min_speed_efficiency:
                print(
                    f"[M{self.month + 1:02d}] speed stop low_eff best={describe_candidate(candidate)} min_speed_eff={min_speed_efficiency:.2f}",
                    file=sys.stderr,
                )
                break
            speed_round += 1
            print(f"[M{self.month + 1:02d}] speed choose reason={reason} round={speed_round} best={describe_candidate(candidate)}", file=sys.stderr)
            budget = self.apply_candidate(reason, candidate, actions, serviced, module_load, degrees, teleport_used, tubes, direct_pod_counts, budget, pod_ids)
            if candidate.teleport is not None:
                teleported_pairs.add((candidate.pad_id, candidate.astronaut_type))

        action_line = ";".join(actions) or "WAIT"
        print(f"[M{self.month + 1:02d}] output resources={self.resources} spent={self.resources - budget} remaining={budget}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] output actions={len(actions)} line={action_line}", file=sys.stderr)
        return actions

    def debug_month_input(self, route_count: int, pod_count: int, new_buildings: list[Building]):
        """Prints the parsed monthly input snapshot to the debug log."""
        module_counts = Counter(building.kind for building in self.buildings.values() if building.kind > 0)
        pads = self.get_landing_pads()
        total_demand = sum(sum(pad.demand.values()) for pad in pads)
        print(
            f"[M{self.month + 1:02d}] input resources={self.resources} route_lines={route_count} tubes={len(self.tubes)} "
            f"teleports={len(self.teleports)} pod_lines={pod_count} pods={len(self.pods)} new_buildings={len(new_buildings)} "
            f"total_buildings={len(self.buildings)} pads={len(pads)} modules={sum(module_counts.values())} total_monthly_demand={total_demand}",
            file=sys.stderr,
        )
        print(f"[M{self.month + 1:02d}] input modules_by_type={format_counter(module_counts)}", file=sys.stderr)
        if self.tubes:
            print(f"[M{self.month + 1:02d}] input tubes={format_tubes(self.tubes)}", file=sys.stderr)
        if self.teleports:
            print(f"[M{self.month + 1:02d}] input teleports={format_teleports(self.teleports)}", file=sys.stderr)
        if self.pods:
            print(f"[M{self.month + 1:02d}] input pods={format_pods(self.pods)}", file=sys.stderr)
        for building in new_buildings:
            print(f"[M{self.month + 1:02d}] input new {describe_building(building)}", file=sys.stderr)

    def debug_plan_start(self, serviced: set[tuple[int, int]], module_load: Counter[int], degrees: Counter[int], direct_pod_counts: Counter[tuple[int, int]],
                         demands: list[tuple[Building, int, int]], min_efficiency: float):
        """Prints the planning state before candidate selection starts."""
        print(
            f"[M{self.month + 1:02d}] plan start months_left={self.months_left()} min_eff={min_efficiency:.2f} "
            f"min_speed_eff={self.min_speed_efficiency():.2f} reserve={self.reserve_floor()} teleport_threshold={self.teleport_threshold()} "
            f"serviced_pairs={len(serviced)} unserved_demands={len(demands)}",
            file=sys.stderr,
        )
        print(f"[M{self.month + 1:02d}] plan module_load={format_counter(module_load)}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan tube_degrees={format_counter(degrees)}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan direct_pods={format_pair_counter(direct_pod_counts)}", file=sys.stderr)

    def get_serviced_pairs(self) -> set[tuple[int, int]]:
        """Gets landing-pad and astronaut-type pairs already served by teleporters or pods."""
        serviced = set()
        for entrance, exit_id in self.teleports.items():
            if entrance in self.buildings and exit_id in self.buildings and self.buildings[entrance].kind == 0 and self.buildings[exit_id].kind > 0:
                serviced.add((entrance, self.buildings[exit_id].kind))

        for pod in self.pods.values():
            if not pod.path or pod.path[0] not in self.buildings or self.buildings[pod.path[0]].kind != 0:
                continue
            pad_id = pod.path[0]
            for building_id in pod.path[1:]:
                if building_id in self.buildings and self.buildings[building_id].kind > 0:
                    serviced.add((pad_id, self.buildings[building_id].kind))
        return serviced

    def get_module_load(self) -> Counter[int]:
        """Estimates monthly passenger counts assigned to each module."""
        loads = Counter()
        counted_pairs = set()
        for entrance, exit_id in self.teleports.items():
            if entrance not in self.buildings or exit_id not in self.buildings:
                continue
            pad = self.buildings[entrance]
            module = self.buildings[exit_id]
            if pad.kind == 0 and module.kind > 0:
                loads[exit_id] += pad.demand[module.kind]
                counted_pairs.add((pad.id, module.kind))

        for pod in self.pods.values():
            if not pod.path or pod.path[0] not in self.buildings or self.buildings[pod.path[0]].kind != 0:
                continue
            pad = self.buildings[pod.path[0]]
            for building_id in pod.path[1:]:
                if building_id not in self.buildings or self.buildings[building_id].kind <= 0 or (pad.id, self.buildings[building_id].kind) in counted_pairs:
                    continue
                counted_pairs.add((pad.id, self.buildings[building_id].kind))
                loads[building_id] += pad.demand[self.buildings[building_id].kind]
        return loads

    def get_tube_degrees(self) -> Counter[int]:
        """Counts magnetic tube endpoints per building."""
        degrees = Counter()
        for a, b in self.tubes:
            degrees[a] += 1
            degrees[b] += 1
        return degrees

    def get_teleport_used_buildings(self) -> set[int]:
        """Gets buildings that already host a teleporter entrance or exit."""
        used = set(self.teleports)
        used.update(self.teleports.values())
        return used

    def get_teleported_pairs(self) -> set[tuple[int, int]]:
        """Gets landing-pad and astronaut-type pairs already served by direct teleporters."""
        pairs = set()
        for entrance, exit_id in self.teleports.items():
            if entrance in self.buildings and exit_id in self.buildings and self.buildings[entrance].kind == 0 and self.buildings[exit_id].kind > 0:
                pairs.add((entrance, self.buildings[exit_id].kind))
        return pairs

    def get_direct_service_pod_counts(self) -> Counter[tuple[int, int]]:
        """Counts pods that directly shuttle between landing pads and modules."""
        counts = Counter()
        for pod in self.pods.values():
            if len(pod.path) == 3 and pod.path[0] == pod.path[2] and pod.path[0] in self.buildings and pod.path[1] in self.buildings:
                if self.buildings[pod.path[0]].kind == 0 and self.buildings[pod.path[1]].kind > 0:
                    counts[(pod.path[0], pod.path[1])] += 1
        return counts

    def get_unserved_demands(self, serviced: set[tuple[int, int]]) -> list[tuple[Building, int, int]]:
        """Gets unserved monthly demands in priority order from the already served pad and astronaut-type pairs."""
        demands = []
        for pad in self.get_landing_pads():
            for astronaut_type, count in pad.demand.items():
                if (pad.id, astronaut_type) not in serviced:
                    demands.append((pad, astronaut_type, count))
        return sorted(demands, key=lambda demand: (-demand[2], demand[0].id, demand[1]))[:260]

    def best_service_candidate(self, pad: Building, astronaut_type: int, count: int, modules_by_type: dict[int, list[Building]], module_load: Counter[int],
                               degrees: Counter[int], teleport_used: set[int], tubes: dict[tuple[int, int], int], budget: int,
                               pod_ids: set[int]) -> Candidate | None:
        """Finds the strongest affordable service candidate for one pad demand, using current planned routes and remaining budget."""
        best = None
        for module in self.best_modules(modules_by_type[astronaut_type], pad, module_load):
            candidates = self.service_candidates(pad, module, astronaut_type, count, module_load[module.id], degrees, teleport_used, tubes, budget, pod_ids)
            for candidate in candidates:
                if best is None or candidate.efficiency > best.efficiency or candidate.efficiency == best.efficiency and candidate.score > best.score:
                    best = candidate
        return best

    def get_modules_by_type(self) -> dict[int, list[Building]]:
        """Groups known lunar modules by astronaut type."""
        modules_by_type = {}
        for building in self.buildings.values():
            if building.kind > 0:
                modules_by_type.setdefault(building.kind, []).append(building)
        return modules_by_type

    def get_landing_pads(self) -> list[Building]:
        """Gets all known landing pads with monthly demands."""
        return [building for building in self.buildings.values() if building.kind == 0 and building.demand]

    def best_modules(self, modules: list[Building], pad: Building, module_load: Counter[int]) -> list[Building]:
        """Orders modules for one landing pad by load-adjusted distance and returns the most promising subset."""
        return sorted(modules, key=lambda module: (module_load[module.id] // 20, tube_cost(pad, module)))[:4]

    def service_candidates(self, pad: Building, module: Building, astronaut_type: int, count: int, current_load: int, degrees: Counter[int],
                           teleport_used: set[int], tubes: dict[tuple[int, int], int], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds affordable tube, pod, and teleporter bundles for one landing-pad demand."""
        candidates = []
        if len(pod_ids) < MAX_PODS:
            has_tube_candidate = False
            existing_path = self.shortest_tube_path(pad.id, module.id, tubes)
            if existing_path is not None and len(existing_path) <= 7 and POD_COST <= budget:
                delivered = monthly_pod_deliveries(count, len(existing_path) - 1, 1)
                score = monthly_score(delivered, len(existing_path) - 1, current_load) * self.months_left()
                candidates.append(Candidate(score, POD_COST, pad.id, module.id, astronaut_type, loop_path(existing_path), delivered=delivered))
                has_tube_candidate = True

            direct_tubes = [] if route_key(pad.id, module.id) in tubes else [(pad.id, module.id)]
            direct_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in direct_tubes) + POD_COST
            if direct_cost <= budget and self.can_reserve_tubes(direct_tubes, degrees, tubes) and len(pod_ids) < MAX_PODS:
                delivered = monthly_pod_deliveries(count, 1, 1)
                score = monthly_score(delivered, 1, current_load) * self.months_left()
                path = [pad.id, module.id, pad.id]
                candidates.append(Candidate(score, direct_cost, pad.id, module.id, astronaut_type, path, direct_tubes, delivered=delivered))
                has_tube_candidate = True

            if not has_tube_candidate and count >= 8:
                for via in self.two_hop_buildings(pad, module):
                    if via.id in (pad.id, module.id):
                        continue
                    path = [pad.id, via.id, module.id]
                    two_hop_tubes = [edge for edge in zip(path, path[1:]) if route_key(edge[0], edge[1]) not in tubes]
                    two_hop_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in two_hop_tubes) + POD_COST
                    if two_hop_cost <= budget and self.can_reserve_tubes(two_hop_tubes, degrees, tubes):
                        delivered = monthly_pod_deliveries(count, 2, 1)
                        score = monthly_score(delivered, 2, current_load) * self.months_left()
                        candidate_path = loop_path(path)
                        candidates.append(Candidate(score, two_hop_cost, pad.id, module.id, astronaut_type, candidate_path, two_hop_tubes, delivered=delivered))
                        break

        if count >= self.teleport_threshold() and pad.id not in teleport_used and module.id not in teleport_used and TELEPORT_COST <= budget:
            score = monthly_teleport_score(count, current_load) * self.months_left()
            candidates.append(Candidate(score, TELEPORT_COST, pad.id, module.id, astronaut_type, teleport=(pad.id, module.id), delivered=count))
        return candidates

    def shortest_tube_path(self, start_id: int, finish_id: int, tubes: dict[tuple[int, int], int]) -> list[int] | None:
        """Gets the shortest existing magnetic-tube building path between two building ids, or no path if disconnected."""
        if start_id == finish_id:
            return [start_id]
        graph = {}
        for a, b in tubes:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)
        queue = deque([start_id])
        parent = {start_id: start_id}
        while queue:
            building_id = queue.popleft()
            for neighbor_id in graph.get(building_id, []):
                if neighbor_id in parent:
                    continue
                parent[neighbor_id] = building_id
                if neighbor_id == finish_id:
                    return unwind_path(parent, start_id, finish_id)
                queue.append(neighbor_id)
        return None

    def can_reserve_tubes(self, new_tubes: list[tuple[int, int]], degrees: Counter[int], tubes: dict[tuple[int, int], int]) -> bool:
        """Checks whether all new tube endpoint pairs can be added to the current planned geometry."""
        extra = []
        extra_keys = set()
        extra_degrees = Counter()
        for a, b in new_tubes:
            key = route_key(a, b)
            if key in tubes or key in extra_keys:
                continue
            if degrees[a] + extra_degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] + extra_degrees[b] >= MAX_TUBES_PER_BUILDING:
                return False
            if not self.can_build_tube(a, b, tubes, extra):
                return False
            extra.append((a, b))
            extra_keys.add(key)
            extra_degrees[a] += 1
            extra_degrees[b] += 1
        return True

    def can_build_tube(self, a: int, b: int, tubes: dict[tuple[int, int], int], extra_tubes: list[tuple[int, int]]) -> bool:
        """Checks whether one tube segment can be built without crossing tubes or passing through buildings."""
        if a == b or route_key(a, b) in tubes:
            return False
        first = self.buildings[a]
        second = self.buildings[b]
        for building in self.buildings.values():
            if building.id not in (a, b) and point_on_segment(building, first, second):
                return False
        for c, d in list(tubes) + [route_key(x, y) for x, y in extra_tubes]:
            if len({a, b, c, d}) < 4:
                continue
            if segments_intersect(first, second, self.buildings[c], self.buildings[d]):
                return False
        return True

    def two_hop_buildings(self, pad: Building, module: Building) -> list[Building]:
        """Orders possible intermediate buildings for a two-hop connection between a landing pad and module."""
        return sorted(self.buildings.values(), key=lambda building: tube_cost(pad, building) + tube_cost(building, module))[:20]

    def best_capacity_candidate(self, serviced: set[tuple[int, int]], tubes: dict[tuple[int, int], int], direct_counts: Counter[tuple[int, int]],
                                teleported_pairs: set[tuple[int, int]], budget: int, pod_ids: set[int]) -> Candidate | None:
        """Finds the best affordable extra pod or tube upgrade for already served direct routes."""
        if len(pod_ids) >= MAX_PODS:
            return None
        best = None
        for pad in self.get_landing_pads():
            for astronaut_type, count in pad.demand.items():
                if (pad.id, astronaut_type) not in serviced or (pad.id, astronaut_type) in teleported_pairs:
                    continue
                modules = [building for building in self.buildings.values() if building.kind == astronaut_type and route_key(pad.id, building.id) in tubes]
                for module in modules:
                    edge = route_key(pad.id, module.id)
                    pods_on_edge = direct_counts[(pad.id, module.id)]
                    if pods_on_edge <= 0 or pods_on_edge >= 4:
                        continue
                    current_capacity = tubes[edge]
                    upgrade_cost = 0 if pods_on_edge < current_capacity else tube_cost(pad, module) * (current_capacity + 1)
                    cost = POD_COST + upgrade_cost
                    if cost > budget:
                        continue
                    old_score = monthly_score(monthly_pod_deliveries(count, 1, pods_on_edge), 1, 0, pods_on_edge)
                    new_score = monthly_score(monthly_pod_deliveries(count, 1, pods_on_edge + 1), 1, 0, pods_on_edge + 1)
                    candidate = Candidate((new_score - old_score) * self.months_left(), cost, pad.id, module.id, astronaut_type, [pad.id, module.id, pad.id],
                                          upgrades=[] if upgrade_cost == 0 else [edge], delivered=count)
                    if best is None or candidate.efficiency > best.efficiency:
                        best = candidate
        return best

    def best_speed_candidate(self, serviced: set[tuple[int, int]], tubes: dict[tuple[int, int], int], direct_counts: Counter[tuple[int, int]],
                             teleport_used: set[int], teleported_pairs: set[tuple[int, int]], budget: int,
                             pod_ids: set[int]) -> tuple[str, Candidate] | None:
        """Finds the best currently spendable speed improvement candidate."""
        candidates = []
        capacity_candidate = self.best_capacity_candidate(serviced, tubes, direct_counts, teleported_pairs, budget, pod_ids)
        if capacity_candidate is not None:
            candidates.append(("capacity", capacity_candidate))
        teleport_candidate = self.best_teleport_speed_candidate(serviced, direct_counts, teleport_used, budget)
        if teleport_candidate is not None:
            candidates.append(("speed_teleport", teleport_candidate))
        if not candidates:
            return None
        if self.months_left() <= 2:
            return max(candidates, key=lambda item: (item[1].score, item[1].efficiency))
        return max(candidates, key=lambda item: (item[1].efficiency, item[1].score))

    def best_teleport_speed_candidate(self, serviced: set[tuple[int, int]], direct_counts: Counter[tuple[int, int]], teleport_used: set[int],
                                      budget: int) -> Candidate | None:
        """Finds the best teleporter that speeds up an already served direct route."""
        if budget < TELEPORT_COST:
            return None
        best = None
        for pad in self.get_landing_pads():
            if pad.id in teleport_used:
                continue
            for astronaut_type, count in pad.demand.items():
                if (pad.id, astronaut_type) not in serviced:
                    continue
                for module in self.get_modules_by_type()[astronaut_type]:
                    pod_count = direct_counts[(pad.id, module.id)]
                    if pod_count <= 0 or module.id in teleport_used:
                        continue
                    old_score = monthly_score(count, 1, 0, pod_count)
                    new_score = monthly_teleport_score(count, 0)
                    candidate = Candidate((new_score - old_score) * self.months_left(), TELEPORT_COST, pad.id, module.id, astronaut_type,
                                          teleport=(pad.id, module.id), delivered=count)
                    if best is None or candidate.score > best.score:
                        best = candidate
        return best

    def apply_candidate(self, reason: str, candidate: Candidate, actions: list[str], serviced: set[tuple[int, int]], module_load: Counter[int],
                        degrees: Counter[int], teleport_used: set[int], tubes: dict[tuple[int, int], int], direct_pod_counts: Counter[tuple[int, int]],
                        budget: int, pod_ids: set[int]) -> int:
        """Appends a chosen candidate to the action list, updates planned state, and returns the remaining budget."""
        action_start = len(actions)
        for a, b in candidate.tubes:
            if route_key(a, b) in tubes:
                continue
            actions.append(f"TUBE {a} {b}")
            tubes[route_key(a, b)] = 1
            degrees[a] += 1
            degrees[b] += 1
        for a, b in candidate.upgrades:
            actions.append(f"UPGRADE {a} {b}")
            tubes[route_key(a, b)] += 1
        if candidate.teleport is not None:
            a, b = candidate.teleport
            actions.append(f"TELEPORT {a} {b}")
            teleport_used.add(a)
            teleport_used.add(b)
        if candidate.path:
            pod_id = next_pod_id(pod_ids)
            pod_ids.add(pod_id)
            actions.append("POD {} {}".format(pod_id, " ".join(map(str, candidate.path))))
            if len(candidate.path) == 3 and candidate.path[0] == candidate.path[2]:
                direct_pod_counts[(candidate.path[0], candidate.path[1])] += 1
        if reason == "service":
            serviced.add((candidate.pad_id, candidate.astronaut_type))
            module_load[candidate.module_id] += candidate.delivered
        new_budget = budget - candidate.cost
        print(
            f"[M{self.month + 1:02d}] apply {reason} budget={budget}->{new_budget} "
            f"actions={format_items(actions[action_start:], 10)} candidate={describe_candidate(candidate)}",
            file=sys.stderr,
        )
        return new_budget

    def months_left(self) -> int:
        """Returns the number of lunar months that can still benefit from new construction."""
        return MAX_MONTHS - self.month

    def min_efficiency(self) -> float:
        """Returns the minimum estimated score per resource worth spending this month."""
        return 0.55 if self.month < 14 else 0.8

    def speed_spend_budget(self, budget: int) -> int:
        """Returns the resources currently allowed for speed-only improvements after preserving reserve."""
        return max(0, budget - self.reserve_floor())

    def reserve_floor(self) -> int:
        """Returns the resource reserve preserved for compounding and future unknown construction."""
        months_left = self.months_left()
        if months_left <= 2:
            return 0
        if months_left <= 5:
            return 1000
        if months_left <= 8:
            return 2500
        if months_left <= 12:
            return 3500
        if months_left <= 16:
            return 4500
        return 6000

    def min_speed_efficiency(self) -> float:
        """Returns the minimum score per resource for speed-only spending at this game stage."""
        months_left = self.months_left()
        if months_left > 12:
            return 0.3
        if months_left > 8:
            return 0.15
        if months_left > 5:
            return 5e-2
        if months_left > 2:
            return 1e-2
        return 0

    def teleport_threshold(self) -> int:
        """Returns the minimum monthly demand for considering a direct teleporter."""
        if self.month <= 3:
            return 55
        if self.month <= 10:
            return 70
        return 90


def describe_building(building: Building) -> str:
    """Formats one building for the debug log."""
    if building.kind == 0:
        return f"pad id={building.id} xy=({building.x},{building.y}) demand={format_counter(building.demand)} total={sum(building.demand.values())}"
    return f"module id={building.id} type={building.kind} xy=({building.x},{building.y})"


def describe_demand(pad: Building, astronaut_type: int, count: int) -> str:
    """Formats one landing-pad demand for the debug log."""
    return f"pad={pad.id} xy=({pad.x},{pad.y}) type={astronaut_type} count={count}"


def describe_candidate(candidate: Candidate) -> str:
    """Formats one candidate with its cost, score, and route details for the debug log."""
    mode = "teleport" if candidate.teleport is not None else "pod"
    parts = [
        f"mode={mode}",
        f"pad={candidate.pad_id}",
        f"type={candidate.astronaut_type}",
        f"module={candidate.module_id}",
        f"cost={candidate.cost}",
        f"score={candidate.score}",
        f"eff={candidate.efficiency:.2f}",
        f"delivered={candidate.delivered}",
    ]
    if candidate.teleport is not None:
        parts.append(f"teleport={candidate.teleport[0]}->{candidate.teleport[1]}")
    if candidate.tubes:
        parts.append(f"tubes={format_edges(candidate.tubes)}")
    if candidate.upgrades:
        parts.append(f"upgrades={format_edges(candidate.upgrades)}")
    if candidate.path:
        parts.append(f"path={format_path(candidate.path)}")
    return " ".join(parts)


def format_counter(counter: Counter[int]) -> str:
    """Formats an integer counter as compact key-value text."""
    items = [f"{key}:{counter[key]}" for key in sorted(counter) if counter[key] > 0]
    return format_items(items)


def format_pair_counter(counter: Counter[tuple[int, int]]) -> str:
    """Formats a pair-keyed counter as compact key-value text."""
    items = [f"{a}->{b}:{counter[(a, b)]}" for a, b in sorted(counter) if counter[(a, b)] > 0]
    return format_items(items)


def format_tubes(tubes: dict[tuple[int, int], int]) -> str:
    """Formats tube capacities as compact route text."""
    return format_items([f"{a}-{b}:c{tubes[(a, b)]}" for a, b in sorted(tubes)])


def format_teleports(teleports: dict[int, int]) -> str:
    """Formats teleporters as compact directed route text."""
    return format_items([f"{entrance}->{teleports[entrance]}" for entrance in sorted(teleports)])


def format_pods(pods: dict[int, Pod]) -> str:
    """Formats pod paths as compact itinerary text."""
    return format_items([f"{pod_id}:{format_path(pods[pod_id].path)}" for pod_id in sorted(pods)])


def format_edges(edges: list[tuple[int, int]]) -> str:
    """Formats endpoint pairs as compact undirected edge text."""
    return format_items([f"{a}-{b}" for a, b in edges])


def format_path(path: list[int]) -> str:
    """Formats a pod path, shortening very long itineraries."""
    if len(path) <= 9:
        return "-".join(map(str, path))
    return "{}-...-{}".format("-".join(map(str, path[:5])), "-".join(map(str, path[-3:])))


def format_items(items: list[str], limit: int = DEBUG_LIST_LIMIT) -> str:
    """Formats a bounded comma-separated item list for debug output."""
    if not items:
        return "none"
    suffix = "" if len(items) <= limit else f",...(+{len(items) - limit})"
    return ",".join(items[:limit]) + suffix


def route_key(a: int, b: int) -> tuple[int, int]:
    """Returns a stable undirected route key for two building ids."""
    return (a, b) if a < b else (b, a)


def tube_cost(a: Building, b: Building) -> int:
    """Calculates magnetic tube construction cost between two buildings."""
    return isqrt(100 * ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)))


def point_on_segment(point: Building, a: Building, b: Building) -> bool:
    """Checks whether a building lies strictly on a segment between two other buildings."""
    return orientation(a, b, point) == 0 and min(a.x, b.x) <= point.x <= max(a.x, b.x) and min(a.y, b.y) <= point.y <= max(a.y, b.y)


def segments_intersect(a: Building, b: Building, c: Building, d: Building) -> bool:
    """Checks whether two closed building-to-building segments intersect."""
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 == 0 and point_on_segment(c, a, b) or o2 == 0 and point_on_segment(d, a, b):
        return True
    if o3 == 0 and point_on_segment(a, c, d) or o4 == 0 and point_on_segment(b, c, d):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def orientation(a: Building, b: Building, c: Building) -> int:
    """Returns the signed orientation of three building coordinates."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def loop_path(path: list[int]) -> list[int]:
    """Returns a round-trip loop path from a one-way building path."""
    return path + path[-2::-1]


def unwind_path(parent: dict[int, int], start_id: int, finish_id: int) -> list[int]:
    """Reconstructs an ordered BFS path from parent links between the start and finish building ids."""
    path = [finish_id]
    while path[-1] != start_id:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def monthly_pod_deliveries(count: int, distance: int, pod_count: int) -> int:
    """Estimates how many passengers a shuttling pod group delivers in one month."""
    delivered = 0
    day = distance
    while day <= MONTH_DAYS and delivered < count:
        delivered += min(count - delivered, 10 * pod_count)
        day += 2 * distance
    return delivered


def monthly_score(count: int, distance: int, current_load: int, pod_count: int = 1) -> int:
    """Estimates one month of score from pod deliveries, travel distance, module load, and synchronized pod count."""
    score = 0
    delivered = 0
    day = distance
    while day <= MONTH_DAYS and delivered < count:
        batch = min(count - delivered, 10 * pod_count)
        for _ in range(batch):
            score += max(0, 50 - day) + max(0, 50 - current_load - delivered)
            delivered += 1
        day += 2 * distance
    return score


def monthly_teleport_score(count: int, current_load: int) -> int:
    """Estimates one month of score for passengers using a direct teleporter into a module with an existing load."""
    return sum(50 + max(0, 50 - current_load - passenger_ind) for passenger_ind in range(count))


def next_pod_id(used_ids: set[int]) -> int:
    """Returns the smallest available pod identifier between 1 and 500 from existing or planned identifiers."""
    for pod_id in range(1, MAX_PODS + 1):
        if pod_id not in used_ids:
            return pod_id
    raise RuntimeError("No pod identifiers remain")


Planner().play()
