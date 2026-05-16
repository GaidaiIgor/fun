"""Builds a heuristic transport network for the Selenia City CodinGame puzzle."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import permutations
from math import isqrt
import sys

MAX_MONTHS = 20
MONTH_DAYS = 20
MAX_TUBES_PER_BUILDING = 5
MAX_PODS = 500
POD_COST = 1000
POD_REFUND = 750
REROUTE_COST = POD_COST - POD_REFUND
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
    services: list[tuple[int, int, int, int]] = field(default_factory=list)
    reroute_pod_id: int | None = None
    lost_score: int = 0
    pressure_gain: int = 0
    max_pressure_gain: int = 0

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
        service_counts = self.get_service_counts()
        module_load = self.get_module_load()
        degrees = self.get_tube_degrees()
        teleport_used = self.get_teleport_used_buildings()
        teleported_pairs = self.get_teleported_pairs()
        pod_ids = set(self.pods)
        tubes = dict(self.tubes)
        direct_pod_counts = self.get_direct_service_pod_counts()
        edge_schedule = self.get_edge_schedule()
        rerouted_pod_ids = set()
        budget = self.resources
        modules_by_type = self.get_modules_by_type()
        unserved_demands = self.get_unserved_demands(serviced)
        min_efficiency = self.min_efficiency()
        self.debug_plan_start(serviced, service_counts, module_load, degrees, direct_pod_counts, edge_schedule, unserved_demands, min_efficiency)

        for demand_ind, (pad, astronaut_type, count) in enumerate(unserved_demands, 1):
            if (pad.id, astronaut_type) in serviced:
                message = f"[M{self.month + 1:02d}] demand {demand_ind}/{len(unserved_demands)} skip already_serviced "
                message += describe_demand(pad, astronaut_type, count)
                print(message, file=sys.stderr)
                continue
            candidate = self.best_service_candidate(pad, astronaut_type, count, modules_by_type, module_load, degrees, teleport_used, tubes, edge_schedule,
                                                    service_counts, rerouted_pod_ids, budget, pod_ids)
            prefix = f"[M{self.month + 1:02d}] demand {demand_ind}/{len(unserved_demands)} {describe_demand(pad, astronaut_type, count)}"
            if candidate is None:
                print(f"{prefix} no_candidate budget={budget}", file=sys.stderr)
            elif candidate.efficiency < min_efficiency:
                print(f"{prefix} skip best={describe_candidate(candidate)} min_eff={min_efficiency:.2f}", file=sys.stderr)
            else:
                print(f"{prefix} choose best={describe_candidate(candidate)}", file=sys.stderr)
                budget = self.apply_candidate("service", candidate, actions, serviced, service_counts, module_load, degrees, teleport_used, tubes,
                                              direct_pod_counts, edge_schedule, budget, pod_ids)
                if candidate.reroute_pod_id is not None:
                    rerouted_pod_ids.add(candidate.reroute_pod_id)
                if candidate.teleport is not None:
                    teleported_pairs.add((candidate.pad_id, candidate.astronaut_type))

        speed_round = 0
        while True:
            speed_budget = self.speed_spend_budget(budget)
            speed_pick = self.best_speed_candidate(serviced, tubes, direct_pod_counts, edge_schedule, rerouted_pod_ids, teleport_used, teleported_pairs,
                                                   speed_budget, pod_ids)
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
                message = f"[M{self.month + 1:02d}] speed stop low_eff best={describe_candidate(candidate)} min_speed_eff={min_speed_efficiency:.2f}"
                print(message, file=sys.stderr)
                break
            speed_round += 1
            print(f"[M{self.month + 1:02d}] speed choose reason={reason} round={speed_round} best={describe_candidate(candidate)}", file=sys.stderr)
            budget = self.apply_candidate(reason, candidate, actions, serviced, service_counts, module_load, degrees, teleport_used, tubes,
                                          direct_pod_counts, edge_schedule, budget, pod_ids)
            if candidate.reroute_pod_id is not None:
                rerouted_pod_ids.add(candidate.reroute_pod_id)
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
        message = f"[M{self.month + 1:02d}] input resources={self.resources} route_lines={route_count} tubes={len(self.tubes)} "
        message += f"teleports={len(self.teleports)} pod_lines={pod_count} pods={len(self.pods)} new_buildings={len(new_buildings)} "
        message += f"total_buildings={len(self.buildings)} pads={len(pads)} modules={sum(module_counts.values())} total_monthly_demand={total_demand}"
        print(message, file=sys.stderr)
        print(f"[M{self.month + 1:02d}] input modules_by_type={format_counter(module_counts)}", file=sys.stderr)
        if self.tubes:
            print(f"[M{self.month + 1:02d}] input tubes={format_tubes(self.tubes)}", file=sys.stderr)
        self.debug_pair_costs()
        if self.teleports:
            print(f"[M{self.month + 1:02d}] input teleports={format_teleports(self.teleports)}", file=sys.stderr)
        if self.pods:
            print(f"[M{self.month + 1:02d}] input pods={format_pods(self.pods)}", file=sys.stderr)
        for building in new_buildings:
            print(f"[M{self.month + 1:02d}] input new {describe_building(building)}", file=sys.stderr)

    def debug_pair_costs(self):
        """Prints construction or upgrade costs for every unordered building pair."""
        degrees = self.get_tube_degrees()
        building_ids = sorted(self.buildings)
        for index, a in enumerate(building_ids):
            for b in building_ids[index + 1:]:
                key = route_key(a, b)
                if key in self.tubes:
                    capacity = self.tubes[key]
                    cost = tube_cost(self.buildings[a], self.buildings[b]) * (capacity + 1)
                    print(f"[M{self.month + 1:02d}] input pair_cost ({a}, {b}) -> {cost}", file=sys.stderr)
                elif degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] >= MAX_TUBES_PER_BUILDING or not self.can_build_tube(a, b, self.tubes, []):
                    print(f"[M{self.month + 1:02d}] input pair_cost ({a}, {b}) -> impossible", file=sys.stderr)
                else:
                    print(f"[M{self.month + 1:02d}] input pair_cost ({a}, {b}) -> {tube_cost(self.buildings[a], self.buildings[b])}", file=sys.stderr)

    def debug_plan_start(self, serviced: set[tuple[int, int]], service_counts: Counter[tuple[int, int]], module_load: Counter[int], degrees: Counter[int],
                         direct_pod_counts: Counter[tuple[int, int]], edge_schedule: Counter[tuple[tuple[int, int], int]],
                         demands: list[tuple[Building, int, int]], min_efficiency: float):
        """Prints the planning state before candidate selection starts."""
        message = f"[M{self.month + 1:02d}] plan start months_left={self.months_left()} min_eff={min_efficiency:.2f} "
        message += f"min_speed_eff={self.min_speed_efficiency():.2f} reserve={self.reserve_floor()} teleport_threshold={self.teleport_threshold()} "
        message += f"serviced_pairs={len(serviced)} unserved_demands={len(demands)}"
        print(message, file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan service_counts={format_pair_counter(service_counts)}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan module_load={format_counter(module_load)}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan tube_degrees={format_counter(degrees)}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan direct_pods={format_pair_counter(direct_pod_counts)}", file=sys.stderr)
        print(f"[M{self.month + 1:02d}] plan edge_schedule={format_edge_schedule(edge_schedule)}", file=sys.stderr)

    def get_serviced_pairs(self) -> set[tuple[int, int]]:
        """Gets landing-pad and astronaut-type pairs already served by teleporters or pods."""
        return set(self.get_service_counts())

    def get_service_counts(self) -> Counter[tuple[int, int]]:
        """Counts how many teleporters or pods currently serve each landing-pad and astronaut-type pair."""
        counts = Counter()
        for entrance, exit_id in self.teleports.items():
            if entrance in self.buildings and exit_id in self.buildings and self.buildings[entrance].kind == 0 and self.buildings[exit_id].kind > 0:
                counts[(entrance, self.buildings[exit_id].kind)] += 1

        for pod in self.pods.values():
            for pad_id, astronaut_type, _, _ in self.pod_services(pod):
                counts[(pad_id, astronaut_type)] += 1
        for pair in self.get_reachable_services():
            counts[pair] = max(1, counts[pair])
        return counts

    def pod_services(self, pod: Pod) -> list[tuple[int, int, int, int]]:
        """Lists pad, astronaut type, module, and one-way distance pairs served by a pod itinerary."""
        if not pod.path or pod.path[0] not in self.buildings or self.buildings[pod.path[0]].kind != 0:
            return []
        services = []
        seen_types = set()
        for distance, building_id in enumerate(pod.path[1:], 1):
            if building_id not in self.buildings or self.buildings[building_id].kind <= 0 or self.buildings[building_id].kind in seen_types:
                continue
            seen_types.add(self.buildings[building_id].kind)
            services.append((pod.path[0], self.buildings[building_id].kind, building_id, distance))
        return services

    def get_reachable_services(self, skip_pod_id: int | None = None) -> dict[tuple[int, int], tuple[int, int]]:
        """Maps each pod-reachable landing-pad demand to the nearest reachable module and travel distance."""
        adjacency = self.get_pod_adjacency(skip_pod_id)
        services = {}
        for pad in self.get_landing_pads():
            queue = deque([pad.id])
            distances = {pad.id: 0}
            while queue:
                building_id = queue.popleft()
                building = self.buildings[building_id]
                if building.kind > 0 and building.kind in pad.demand and (pad.id, building.kind) not in services:
                    services[(pad.id, building.kind)] = (building_id, distances[building_id])
                for neighbor_id in adjacency.get(building_id, []):
                    if neighbor_id in distances:
                        continue
                    distances[neighbor_id] = distances[building_id] + 1
                    queue.append(neighbor_id)
        return services

    def get_pod_adjacency(self, skip_pod_id: int | None = None) -> dict[int, list[int]]:
        """Builds directed pod and teleporter reachability between buildings, optionally omitting one pod."""
        adjacency = {}
        for a, b in self.get_pod_directed_edges(skip_pod_id):
            adjacency.setdefault(a, []).append(b)
        for a, b in self.teleports.items():
            adjacency.setdefault(a, []).append(b)
        return adjacency

    def get_pod_directed_edges(self, skip_pod_id: int | None = None) -> set[tuple[int, int]]:
        """Gets directed building hops that at least one current pod can eventually traverse."""
        edges = set()
        for pod in self.pods.values():
            if pod.id == skip_pod_id:
                continue
            edges.update(zip(pod.path, pod.path[1:]))
        return edges

    def get_module_load(self) -> Counter[int]:
        """Estimates monthly passenger counts assigned to each module."""
        loads = Counter()
        for (pad_id, astronaut_type), (module_id, _) in self.get_reachable_services().items():
            loads[module_id] += self.buildings[pad_id].demand[astronaut_type]
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

    def get_edge_schedule(self) -> Counter[tuple[tuple[int, int], int]]:
        """Counts pods using each magnetic tube edge on each day of the lunar month."""
        schedule = Counter()
        for pod in self.pods.values():
            for edge, day in path_edge_days(pod.path):
                schedule[(edge, day)] += 1
        return schedule

    def get_directed_edge_schedule(self) -> Counter[tuple[tuple[int, int], int]]:
        """Counts pods departing through each directed tube edge on each day of the lunar month."""
        schedule = Counter()
        for pod in self.pods.values():
            for edge, day in directed_path_edge_days(pod.path):
                schedule[(edge, day)] += 1
        return schedule

    def get_unserved_demands(self, serviced: set[tuple[int, int]]) -> list[tuple[Building, int, int]]:
        """Gets unserved monthly demands in priority order from the already served pad and astronaut-type pairs."""
        demands = []
        for pad in self.get_landing_pads():
            for astronaut_type, count in pad.demand.items():
                if (pad.id, astronaut_type) not in serviced:
                    demands.append((pad, astronaut_type, count))
        return sorted(demands, key=lambda demand: (-demand[2], demand[0].id, demand[1]))[:260]

    def best_service_candidate(self, pad: Building, astronaut_type: int, count: int, modules_by_type: dict[int, list[Building]], module_load: Counter[int],
                               degrees: Counter[int], teleport_used: set[int], tubes: dict[tuple[int, int], int],
                               edge_schedule: Counter[tuple[tuple[int, int], int]], service_counts: Counter[tuple[int, int]], rerouted_pod_ids: set[int],
                               budget: int, pod_ids: set[int]) -> Candidate | None:
        """Finds the strongest affordable service candidate for one pad demand, using current planned routes and remaining budget."""
        best = None
        for module in self.best_modules(modules_by_type[astronaut_type], pad, module_load):
            candidates = self.service_candidates(pad, module, astronaut_type, count, module_load[module.id], degrees, teleport_used, tubes, edge_schedule,
                                                 service_counts, rerouted_pod_ids, budget, pod_ids)
            for candidate in candidates:
                if best is None or candidate.efficiency > best.efficiency or candidate.efficiency == best.efficiency and candidate.score > best.score:
                    best = candidate
        for candidate in self.multi_service_candidates(pad, astronaut_type, modules_by_type, module_load, degrees, tubes, edge_schedule, service_counts,
                                                       budget, pod_ids):
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

    def multi_service_candidates(self, pad: Building, required_type: int, modules_by_type: dict[int, list[Building]], module_load: Counter[int],
                                 degrees: Counter[int], tubes: dict[tuple[int, int], int], edge_schedule: Counter[tuple[tuple[int, int], int]],
                                 service_counts: Counter[tuple[int, int]], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds one-pod candidates that visit several modules from the same landing pad."""
        if len(pod_ids) >= MAX_PODS:
            return []
        demands = []
        for astronaut_type, count in pad.demand.items():
            if (pad.id, astronaut_type) not in service_counts and astronaut_type in modules_by_type:
                module = self.best_modules(modules_by_type[astronaut_type], pad, module_load)[0]
                demands.append((astronaut_type, count, module))
        if len(demands) < 2 or all(astronaut_type != required_type for astronaut_type, _, _ in demands):
            return []

        candidates = []
        for size in range(2, min(4, len(demands)) + 1):
            for ordered in permutations(demands, size):
                if all(astronaut_type != required_type for astronaut_type, _, _ in ordered):
                    continue
                path = [pad.id]
                for _, _, module in ordered:
                    path.extend([module.id, pad.id])
                direct_tubes = [(pad.id, module.id) for _, _, module in ordered if route_key(pad.id, module.id) not in tubes]
                if not self.can_reserve_tubes(direct_tubes, degrees, tubes):
                    continue
                upgrade_cost, upgrades = self.path_upgrade_plan(path, direct_tubes, tubes, edge_schedule)
                cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in direct_tubes) + POD_COST + upgrade_cost
                if cost > budget:
                    continue
                score = 0
                delivered_total = 0
                services = []
                period = len(path) - 1
                for stop_index, (astronaut_type, count, module) in enumerate(ordered, 1):
                    first_day = 2 * stop_index - 1
                    delivered = monthly_pod_deliveries(count, first_day, 1, period)
                    score += monthly_score(delivered, first_day, module_load[module.id], period=period) * self.months_left()
                    delivered_total += delivered
                    services.append((pad.id, astronaut_type, module.id, delivered))
                first_type, _, first_module = ordered[0]
                candidate = Candidate(score, cost, pad.id, first_module.id, first_type, path, direct_tubes, upgrades, delivered=delivered_total,
                                      services=services)
                candidates.append(candidate)
        return candidates

    def service_candidates(self, pad: Building, module: Building, astronaut_type: int, count: int, current_load: int, degrees: Counter[int],
                           teleport_used: set[int], tubes: dict[tuple[int, int], int], edge_schedule: Counter[tuple[tuple[int, int], int]],
                           service_counts: Counter[tuple[int, int]], rerouted_pod_ids: set[int], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds affordable tube, pod, and teleporter bundles for one landing-pad demand."""
        candidates = []
        route_options = []
        has_tube_candidate = False
        existing_path = self.shortest_tube_path(pad.id, module.id, tubes)
        if existing_path is not None and len(existing_path) <= 7:
            candidate_path = loop_path(existing_path)
            route_options.append((candidate_path, [], len(existing_path) - 1, 2 * (len(existing_path) - 1)))
            if len(pod_ids) < MAX_PODS:
                upgrade_cost, upgrades = self.path_upgrade_plan(candidate_path, [], tubes, edge_schedule)
                candidate_cost = POD_COST + upgrade_cost
                delivered = monthly_pod_deliveries(count, len(existing_path) - 1, 1)
                score = monthly_score(delivered, len(existing_path) - 1, current_load) * self.months_left()
                if candidate_cost <= budget:
                    candidate = Candidate(score, candidate_cost, pad.id, module.id, astronaut_type, candidate_path, upgrades=upgrades, delivered=delivered)
                    candidates.append(candidate)
                    has_tube_candidate = True
            if len(existing_path) > 2 and self.path_has_pod_coverage(existing_path[:-1]):
                transfer_path = [existing_path[-2], module.id, existing_path[-2]]
                first_day = 2 * (len(existing_path) - 2) + 1
                route_options.append((transfer_path, [], first_day, 2))
                if len(pod_ids) < MAX_PODS:
                    upgrade_cost, upgrades = self.path_upgrade_plan(transfer_path, [], tubes, edge_schedule)
                    transfer_cost = POD_COST + upgrade_cost
                    if transfer_cost <= budget:
                        delivered = monthly_pod_deliveries(count, first_day, 1, 2)
                        score = monthly_score(delivered, first_day, current_load, period=2) * self.months_left()
                        candidate = Candidate(score, transfer_cost, pad.id, module.id, astronaut_type, transfer_path, upgrades=upgrades,
                                              delivered=delivered)
                        candidates.append(candidate)
                        has_tube_candidate = True

        direct_tubes = [] if route_key(pad.id, module.id) in tubes else [(pad.id, module.id)]
        path = [pad.id, module.id, pad.id]
        if self.can_reserve_tubes(direct_tubes, degrees, tubes):
            route_options.append((path, direct_tubes, 1, 2))
            if len(pod_ids) < MAX_PODS:
                upgrade_cost, upgrades = self.path_upgrade_plan(path, direct_tubes, tubes, edge_schedule)
                direct_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in direct_tubes) + POD_COST + upgrade_cost
                if direct_cost <= budget:
                    delivered = monthly_pod_deliveries(count, 1, 1)
                    score = monthly_score(delivered, 1, current_load) * self.months_left()
                    candidates.append(Candidate(score, direct_cost, pad.id, module.id, astronaut_type, path, direct_tubes, upgrades, delivered=delivered))
                    has_tube_candidate = True

        if not has_tube_candidate and count >= 8:
            two_hop_options = 0
            for via in self.two_hop_buildings(pad, module):
                if via.id in (pad.id, module.id):
                    continue
                path = [pad.id, via.id, module.id]
                two_hop_tubes = [edge for edge in zip(path, path[1:]) if route_key(edge[0], edge[1]) not in tubes]
                candidate_path = loop_path(path)
                if not self.can_reserve_tubes(two_hop_tubes, degrees, tubes):
                    continue
                route_options.append((candidate_path, two_hop_tubes, 2, 4))
                two_hop_options += 1
                added_two_hop = False
                if len(pod_ids) < MAX_PODS:
                    upgrade_cost, upgrades = self.path_upgrade_plan(candidate_path, two_hop_tubes, tubes, edge_schedule)
                    two_hop_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in two_hop_tubes) + POD_COST + upgrade_cost
                    if two_hop_cost <= budget:
                        delivered = monthly_pod_deliveries(count, 2, 1)
                        score = monthly_score(delivered, 2, current_load) * self.months_left()
                        candidate = Candidate(score, two_hop_cost, pad.id, module.id, astronaut_type, candidate_path, two_hop_tubes, upgrades,
                                              delivered=delivered)
                        candidates.append(candidate)
                        added_two_hop = True
                if self.path_has_pod_coverage([pad.id, via.id]):
                    transfer_path = [via.id, module.id, via.id]
                    transfer_tubes = [] if route_key(via.id, module.id) in tubes else [(via.id, module.id)]
                    route_options.append((transfer_path, transfer_tubes, 3, 2))
                    if len(pod_ids) < MAX_PODS:
                        upgrade_cost, upgrades = self.path_upgrade_plan(transfer_path, transfer_tubes, tubes, edge_schedule)
                        transfer_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in transfer_tubes) + POD_COST + upgrade_cost
                        if transfer_cost <= budget:
                            delivered = monthly_pod_deliveries(count, 3, 1, 2)
                            score = monthly_score(delivered, 3, current_load, period=2) * self.months_left()
                            candidate = Candidate(score, transfer_cost, pad.id, module.id, astronaut_type, transfer_path, transfer_tubes, upgrades,
                                                  delivered=delivered)
                            candidates.append(candidate)
                            added_two_hop = True
                if added_two_hop:
                    break
                if two_hop_options >= 6:
                    break

        if route_options:
            candidates.extend(self.reroute_candidates(pad, module, astronaut_type, count, current_load, tubes, edge_schedule, rerouted_pod_ids,
                                                      route_options, budget))

        if count >= self.teleport_threshold() and pad.id not in teleport_used and module.id not in teleport_used and TELEPORT_COST <= budget:
            score = monthly_teleport_score(count, current_load) * self.months_left()
            candidates.append(Candidate(score, TELEPORT_COST, pad.id, module.id, astronaut_type, teleport=(pad.id, module.id), delivered=count))
        return candidates

    def path_has_pod_coverage(self, path: list[int]) -> bool:
        """Checks whether current pods can move passengers along every directed hop of a path."""
        edges = self.get_pod_directed_edges()
        return all((a, b) in edges for a, b in zip(path, path[1:]))

    def reroute_candidates(self, pad: Building, module: Building, astronaut_type: int, count: int, current_load: int, tubes: dict[tuple[int, int], int],
                           edge_schedule: Counter[tuple[tuple[int, int], int]], rerouted_pod_ids: set[int],
                           route_options: list[tuple[list[int], list[tuple[int, int]], int, int]], budget: int) -> list[Candidate]:
        """Builds pod destroy-and-recreate candidates for the same route options as new service pods."""
        candidates = []
        for old_pod in self.reroutable_pods(rerouted_pod_ids):
            lost_score = self.reroute_loss(old_pod)
            removed_schedule = self.schedule_without_pod(edge_schedule, old_pod)
            for path, new_tubes, first_day, period in route_options:
                upgrade_cost, upgrades = self.path_upgrade_plan(path, new_tubes, tubes, removed_schedule)
                cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in new_tubes) + REROUTE_COST + upgrade_cost
                if cost > budget:
                    continue
                delivered = monthly_pod_deliveries(count, first_day, 1, period)
                score = monthly_score(delivered, first_day, current_load, period=period) * self.months_left() - lost_score
                candidate = Candidate(score, cost, pad.id, module.id, astronaut_type, path, new_tubes, upgrades, delivered=delivered,
                                      reroute_pod_id=old_pod.id, lost_score=lost_score)
                candidates.append(candidate)
        return candidates

    def reroutable_pods(self, rerouted_pod_ids: set[int]) -> list[Pod]:
        """Orders existing pods by the estimated score sacrificed if they are destroyed this month."""
        pods = [pod for pod in self.pods.values() if pod.id not in rerouted_pod_ids and pod.path]
        return sorted(pods, key=lambda pod: (self.reroute_loss(pod), pod.id))[:30]

    def reroute_loss(self, pod: Pod) -> int:
        """Estimates future score lost by removing a pod from service before recreating it elsewhere."""
        loss = 0
        after = self.get_reachable_services(pod.id)
        for (pad_id, astronaut_type), (_, distance) in self.get_reachable_services().items():
            if (pad_id, astronaut_type) not in after:
                count = self.buildings[pad_id].demand[astronaut_type]
                delivered = monthly_pod_deliveries(count, distance, 1)
                loss += monthly_score(delivered, distance, 0) * self.months_left()
        return loss

    def schedule_without_pod(self, edge_schedule: Counter[tuple[tuple[int, int], int]], pod: Pod) -> Counter[tuple[tuple[int, int], int]]:
        """Returns an edge-day schedule with one existing pod itinerary removed."""
        schedule = edge_schedule.copy()
        for edge, day in path_edge_days(pod.path):
            schedule[(edge, day)] -= 1
            if schedule[(edge, day)] <= 0:
                del schedule[(edge, day)]
        return schedule

    def directed_schedule_without_pod(self, schedule: Counter[tuple[tuple[int, int], int]], pod: Pod) -> Counter[tuple[tuple[int, int], int]]:
        """Returns a directed edge-day schedule with one existing pod itinerary removed."""
        result = schedule.copy()
        for edge, day in directed_path_edge_days(pod.path):
            result[(edge, day)] -= 1
            if result[(edge, day)] <= 0:
                del result[(edge, day)]
        return result

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

    def path_upgrade_plan(self, path: list[int], new_tubes: list[tuple[int, int]], tubes: dict[tuple[int, int], int],
                          edge_schedule: Counter[tuple[tuple[int, int], int]]) -> tuple[int, list[tuple[int, int]]]:
        """Gets tube upgrades needed before a new pod can use a path without same-day edge conflicts."""
        new_keys = {route_key(a, b) for a, b in new_tubes}
        required_capacities = Counter()
        for edge, day in path_edge_days(path):
            required_capacities[edge] = max(required_capacities[edge], edge_schedule[(edge, day)] + 1)

        upgrades = []
        cost = 0
        for edge, required_capacity in required_capacities.items():
            capacity = 1 if edge in new_keys else tubes[edge]
            for new_capacity in range(capacity + 1, required_capacity + 1):
                a, b = edge
                cost += tube_cost(self.buildings[a], self.buildings[b]) * new_capacity
                upgrades.append(edge)
        return cost, upgrades

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
                                edge_schedule: Counter[tuple[tuple[int, int], int]], rerouted_pod_ids: set[int],
                                teleported_pairs: set[tuple[int, int]], budget: int, pod_ids: set[int]) -> Candidate | None:
        """Finds the best affordable new or rerouted pod capacity for already served direct routes."""
        best = None
        directed_schedule = self.get_directed_edge_schedule()
        old_max_pressure, old_total_pressure, old_node_pressure = self.waiting_pressure_metrics(directed_schedule)
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
                    path = [pad.id, module.id, pad.id]
                    old_score = monthly_score(monthly_pod_deliveries(count, 1, pods_on_edge), 1, 0, pods_on_edge)
                    new_score = monthly_score(monthly_pod_deliveries(count, 1, pods_on_edge + 1), 1, 0, pods_on_edge + 1)
                    score_gain = (new_score - old_score) * self.months_left()
                    if len(pod_ids) < MAX_PODS:
                        upgrade_cost, upgrades = self.path_upgrade_plan(path, [], tubes, edge_schedule)
                        cost = POD_COST + upgrade_cost
                        if cost <= budget:
                            candidate = Candidate(score_gain, cost, pad.id, module.id, astronaut_type, path, upgrades=upgrades, delivered=count)
                            self.set_capacity_pressure(candidate, directed_schedule, old_max_pressure, old_total_pressure, old_node_pressure)
                            if self.is_good_capacity_pressure(candidate) and self.is_better_capacity_candidate(candidate, best):
                                best = candidate
                    for old_pod in self.reroutable_pods(rerouted_pod_ids):
                        if old_pod.path == path:
                            continue
                        removed_schedule = self.schedule_without_pod(edge_schedule, old_pod)
                        upgrade_cost, upgrades = self.path_upgrade_plan(path, [], tubes, removed_schedule)
                        cost = REROUTE_COST + upgrade_cost
                        if cost > budget:
                            continue
                        lost_score = self.reroute_loss(old_pod)
                        candidate = Candidate(score_gain - lost_score, cost, pad.id, module.id, astronaut_type, path, upgrades=upgrades, delivered=count,
                                              reroute_pod_id=old_pod.id, lost_score=lost_score)
                        changed_schedule = self.directed_schedule_without_pod(directed_schedule, old_pod)
                        self.set_capacity_pressure(candidate, changed_schedule, old_max_pressure, old_total_pressure, old_node_pressure)
                        if self.is_good_capacity_pressure(candidate) and self.is_better_capacity_candidate(candidate, best):
                            best = candidate
        return best

    def set_capacity_pressure(self, candidate: Candidate, base_schedule: Counter[tuple[tuple[int, int], int]], old_max_pressure: int, old_total_pressure: int,
                              old_node_pressure: Counter[int]):
        """Stores how much a capacity candidate improves total and maximum estimated waiting pressure."""
        schedule = base_schedule.copy()
        for edge, day in directed_path_edge_days(candidate.path):
            schedule[(edge, day)] += 1
        new_max_pressure, new_total_pressure, new_node_pressure = self.waiting_pressure_metrics(schedule)
        candidate.max_pressure_gain = old_max_pressure - new_max_pressure
        candidate.pressure_gain = old_total_pressure - new_total_pressure
        if candidate.reroute_pod_id is not None and candidate.max_pressure_gain == 0:
            for node_id, pressure in new_node_pressure.items():
                if pressure > old_node_pressure[node_id]:
                    candidate.pressure_gain = min(candidate.pressure_gain, -1)

    def waiting_pressure_metrics(self, directed_schedule: Counter[tuple[tuple[int, int], int]]) -> tuple[int, int, Counter[int]]:
        """Estimates maximum, total, and per-building passenger-days spent waiting for pod departures."""
        node_pressure = Counter()
        pod_edges = {edge for edge, _ in directed_schedule}
        for (pad_id, astronaut_type), path in self.get_service_paths().items():
            queues = [0] * len(path)
            queues[0] = self.buildings[pad_id].demand[astronaut_type]
            for day in range(MONTH_DAYS):
                self.apply_instant_edges(path, queues, pod_edges)
                for index, waiting in enumerate(queues[:-1]):
                    if waiting and (path[index], path[index + 1]) in pod_edges:
                        node_pressure[path[index]] += waiting
                moved = [0] * len(path)
                for index, waiting in enumerate(queues[:-1]):
                    edge = (path[index], path[index + 1])
                    boarded = min(waiting, 10 * directed_schedule[(edge, day)]) if edge in pod_edges else 0
                    queues[index] -= boarded
                    moved[index + 1] += boarded
                for index, count in enumerate(moved):
                    queues[index] += count
        return max(node_pressure.values(), default=0), sum(node_pressure.values()), node_pressure

    def apply_instant_edges(self, path: list[int], queues: list[int], pod_edges: set[tuple[int, int]]):
        """Moves queued passengers through non-pod path edges before daily waiting is measured."""
        changed = True
        while changed:
            changed = False
            for index, waiting in enumerate(queues[:-1]):
                if waiting and (path[index], path[index + 1]) not in pod_edges:
                    queues[index + 1] += waiting
                    queues[index] = 0
                    changed = True

    def get_service_paths(self) -> dict[tuple[int, int], list[int]]:
        """Maps each served landing-pad demand to the shortest current pod or teleporter building path."""
        adjacency = self.get_pod_adjacency()
        paths = {}
        for pad in self.get_landing_pads():
            queue = deque([pad.id])
            parent = {pad.id: pad.id}
            while queue:
                building_id = queue.popleft()
                building = self.buildings[building_id]
                if building.kind > 0 and building.kind in pad.demand and (pad.id, building.kind) not in paths:
                    paths[(pad.id, building.kind)] = unwind_path(parent, pad.id, building_id)
                for neighbor_id in adjacency.get(building_id, []):
                    if neighbor_id in parent:
                        continue
                    parent[neighbor_id] = building_id
                    queue.append(neighbor_id)
        return paths

    def is_good_capacity_pressure(self, candidate: Candidate) -> bool:
        """Checks whether a capacity candidate reduces queue pressure instead of merely moving it elsewhere."""
        if candidate.max_pressure_gain > 0:
            return True
        return candidate.max_pressure_gain == 0 and candidate.pressure_gain > 0

    def is_better_capacity_candidate(self, candidate: Candidate, best: Candidate | None) -> bool:
        """Compares capacity candidates by worst waiting pressure, total waiting pressure, then score efficiency."""
        if best is None:
            return True
        candidate_key = (candidate.max_pressure_gain, candidate.pressure_gain, candidate.efficiency, candidate.score)
        best_key = (best.max_pressure_gain, best.pressure_gain, best.efficiency, best.score)
        return candidate_key > best_key

    def best_speed_candidate(self, serviced: set[tuple[int, int]], tubes: dict[tuple[int, int], int], direct_counts: Counter[tuple[int, int]],
                             edge_schedule: Counter[tuple[tuple[int, int], int]], rerouted_pod_ids: set[int], teleport_used: set[int],
                             teleported_pairs: set[tuple[int, int]], budget: int, pod_ids: set[int]) -> tuple[str, Candidate] | None:
        """Finds the best currently spendable speed improvement candidate."""
        candidates = []
        capacity_candidate = self.best_capacity_candidate(serviced, tubes, direct_counts, edge_schedule, rerouted_pod_ids, teleported_pairs, budget, pod_ids)
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

    def apply_candidate(self, reason: str, candidate: Candidate, actions: list[str], serviced: set[tuple[int, int]],
                        service_counts: Counter[tuple[int, int]], module_load: Counter[int], degrees: Counter[int], teleport_used: set[int],
                        tubes: dict[tuple[int, int], int], direct_pod_counts: Counter[tuple[int, int]],
                        edge_schedule: Counter[tuple[tuple[int, int], int]], budget: int, pod_ids: set[int]) -> int:
        """Appends a chosen candidate to the action list, updates planned state, and returns the remaining budget."""
        action_start = len(actions)
        if candidate.reroute_pod_id is not None:
            old_pod = self.pods[candidate.reroute_pod_id]
            actions.append(f"DESTROY {candidate.reroute_pod_id}")
            if len(old_pod.path) == 3 and old_pod.path[0] == old_pod.path[2]:
                direct_pod_counts[(old_pod.path[0], old_pod.path[1])] -= 1
            for edge, day in path_edge_days(old_pod.path):
                edge_schedule[(edge, day)] -= 1
                if edge_schedule[(edge, day)] <= 0:
                    del edge_schedule[(edge, day)]
            for pad_id, astronaut_type, module_id, _ in self.pod_services(old_pod):
                service_counts[(pad_id, astronaut_type)] -= 1
                if service_counts[(pad_id, astronaut_type)] <= 0:
                    del service_counts[(pad_id, astronaut_type)]
                    serviced.discard((pad_id, astronaut_type))
                    module_load[module_id] -= self.buildings[pad_id].demand[astronaut_type]
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
            pod_id = candidate.reroute_pod_id if candidate.reroute_pod_id is not None else next_pod_id(pod_ids)
            pod_ids.add(pod_id)
            actions.append("POD {} {}".format(pod_id, " ".join(map(str, candidate.path))))
            if len(candidate.path) == 3 and candidate.path[0] == candidate.path[2]:
                direct_pod_counts[(candidate.path[0], candidate.path[1])] += 1
            for edge, day in path_edge_days(candidate.path):
                edge_schedule[(edge, day)] += 1
        if reason == "service":
            for pad_id, astronaut_type, module_id, delivered in candidate.services or [(candidate.pad_id, candidate.astronaut_type, candidate.module_id,
                                                                                        candidate.delivered)]:
                serviced.add((pad_id, astronaut_type))
                service_counts[(pad_id, astronaut_type)] += 1
                module_load[module_id] += delivered
        elif candidate.path and reason == "capacity":
            serviced.add((candidate.pad_id, candidate.astronaut_type))
            service_counts[(candidate.pad_id, candidate.astronaut_type)] += 1
        new_budget = budget - candidate.cost
        message = f"[M{self.month + 1:02d}] apply {reason} budget={budget}->{new_budget} "
        message += f"actions={format_items(actions[action_start:], 10)} candidate={describe_candidate(candidate)}"
        print(message, file=sys.stderr)
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
        f"delivered={candidate.delivered}"]
    if candidate.teleport is not None:
        parts.append(f"teleport={candidate.teleport[0]}->{candidate.teleport[1]}")
    if candidate.services:
        parts.append(f"services={format_services(candidate.services)}")
    if candidate.reroute_pod_id is not None:
        parts.append(f"reroute_pod={candidate.reroute_pod_id}")
    if candidate.lost_score:
        parts.append(f"lost_score={candidate.lost_score}")
    if candidate.pressure_gain or candidate.max_pressure_gain:
        parts.append(f"pressure_gain={candidate.pressure_gain}")
        parts.append(f"max_wait_gain={candidate.max_pressure_gain}")
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


def format_edge_schedule(schedule: Counter[tuple[tuple[int, int], int]]) -> str:
    """Formats maximum scheduled pod occupancy for each edge."""
    max_by_edge = Counter()
    for edge, day in schedule:
        max_by_edge[edge] = max(max_by_edge[edge], schedule[(edge, day)])
    return format_pair_counter(max_by_edge)


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


def format_services(services: list[tuple[int, int, int, int]]) -> str:
    """Formats grouped pad, astronaut type, module, and delivered-count service details."""
    return format_items([f"{pad_id}:t{astronaut_type}->m{module_id}/{delivered}" for pad_id, astronaut_type, module_id, delivered in services])


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


def path_edge_days(path: list[int]) -> list[tuple[tuple[int, int], int]]:
    """Gets the undirected tube edges used by a pod on each day of a lunar month."""
    edges = [route_key(a, b) for a, b in zip(path, path[1:])]
    if not edges:
        return []
    if path[0] == path[-1]:
        return [(edges[day % len(edges)], day) for day in range(MONTH_DAYS)]
    return [(edges[day], day) for day in range(min(MONTH_DAYS, len(edges)))]


def directed_path_edge_days(path: list[int]) -> list[tuple[tuple[int, int], int]]:
    """Gets the directed tube edges used by a pod on each day of a lunar month."""
    edges = list(zip(path, path[1:]))
    if not edges:
        return []
    if path[0] == path[-1]:
        return [(edges[day % len(edges)], day) for day in range(MONTH_DAYS)]
    return [(edges[day], day) for day in range(min(MONTH_DAYS, len(edges)))]


def unwind_path(parent: dict[int, int], start_id: int, finish_id: int) -> list[int]:
    """Reconstructs an ordered BFS path from parent links between the start and finish building ids."""
    path = [finish_id]
    while path[-1] != start_id:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def monthly_pod_deliveries(count: int, distance: int, pod_count: int, period: int | None = None) -> int:
    """Estimates how many passengers a shuttling pod group delivers in one month."""
    delivered = 0
    day = distance
    period = 2 * distance if period is None else period
    while day <= MONTH_DAYS and delivered < count:
        delivered += min(count - delivered, 10 * pod_count)
        day += period
    return delivered


def monthly_score(count: int, distance: int, current_load: int, pod_count: int = 1, period: int | None = None) -> int:
    """Estimates one month of score from pod deliveries, travel distance, module load, and synchronized pod count."""
    score = 0
    delivered = 0
    day = distance
    period = 2 * distance if period is None else period
    while day <= MONTH_DAYS and delivered < count:
        batch = min(count - delivered, 10 * pod_count)
        for _ in range(batch):
            score += max(0, 50 - day) + max(0, 50 - current_load - delivered)
            delivered += 1
        day += period
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
