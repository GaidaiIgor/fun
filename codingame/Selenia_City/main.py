"""Builds a heuristic transport network for the Selenia City CodinGame puzzle."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import permutations, product
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
DEBUG_PAIR_COST_LIMIT = 60


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
    extra_paths: list[list[int]] = field(default_factory=list)
    replaced_services: list[tuple[int, int, int, int]] = field(default_factory=list)

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
        planned_pods = {pod_id: pod.path[:] for pod_id, pod in self.pods.items()}
        planned_teleports = dict(self.teleports)
        tubes = dict(self.tubes)
        direct_pod_counts = self.get_direct_service_pod_counts()
        edge_schedule = self.get_edge_schedule()
        dedicated_edge_counts = self.get_dedicated_pod_edge_counts()
        rerouted_pod_ids = set()
        retired_pod_ids = set()
        budget = self.resources
        modules_by_type = self.get_modules_by_type()
        unserved_demands = self.get_unserved_demands(serviced)
        min_efficiency = self.min_efficiency()
        self.debug_scores("before", planned_pods, planned_teleports)

        for pad, astronaut_type, count in unserved_demands:
            if (pad.id, astronaut_type) in serviced:
                continue
            candidate = self.best_service_candidate(pad, astronaut_type, count, modules_by_type, module_load, degrees, teleport_used, tubes, edge_schedule,
                                                    service_counts, planned_pods, rerouted_pod_ids, budget, pod_ids)
            if candidate is not None and candidate.efficiency >= min_efficiency:
                budget = self.apply_candidate("service", candidate, actions, serviced, service_counts, module_load, degrees, teleport_used,
                                              planned_teleports, tubes,
                                              direct_pod_counts, edge_schedule, dedicated_edge_counts, planned_pods, budget, pod_ids)
                if candidate.reroute_pod_id is not None:
                    rerouted_pod_ids.add(candidate.reroute_pod_id)
                if candidate.teleport is not None:
                    teleported_pairs.add((candidate.pad_id, candidate.astronaut_type))

        while True:
            budget = self.destroy_obsolete_pods(actions, service_counts, edge_schedule, planned_pods, rerouted_pod_ids, retired_pod_ids, budget, pod_ids)
            speed_pick = self.best_speed_candidate(serviced, service_counts, tubes, direct_pod_counts, edge_schedule, dedicated_edge_counts, planned_pods,
                                                   planned_teleports, rerouted_pod_ids, teleport_used, teleported_pairs, budget, pod_ids)
            if speed_pick is None:
                break
            reason, candidate = speed_pick
            if candidate.score <= 0:
                break
            budget = self.apply_candidate(reason, candidate, actions, serviced, service_counts, module_load, degrees, teleport_used, planned_teleports, tubes,
                                          direct_pod_counts, edge_schedule, dedicated_edge_counts, planned_pods, budget, pod_ids)
            if candidate.reroute_pod_id is not None:
                rerouted_pod_ids.add(candidate.reroute_pod_id)
            if candidate.teleport is not None:
                teleported_pairs.add((candidate.pad_id, candidate.astronaut_type))

        print(f"[M{self.month + 1:02d}] plan resources_after={budget} spent={self.resources - budget}", file=sys.stderr)
        self.debug_scores("after", planned_pods, planned_teleports)
        return actions

    def destroy_obsolete_pods(self, actions: list[str], service_counts: Counter[tuple[int, int]], edge_schedule: Counter[tuple[tuple[int, int], int]],
                              planned_pods: dict[int, list[int]], rerouted_pod_ids: set[int], retired_pod_ids: set[int], budget: int,
                              pod_ids: set[int]) -> int:
        """Destroys current multi-service pods whose served demands are covered by other routes and returns the updated budget."""
        while obsolete_pod := self.obsolete_pod(service_counts, rerouted_pod_ids | retired_pod_ids):
            actions.append(f"DESTROY {obsolete_pod.id}")
            retired_pod_ids.add(obsolete_pod.id)
            pod_ids.discard(obsolete_pod.id)
            del planned_pods[obsolete_pod.id]
            for edge, day in path_edge_days(obsolete_pod.path):
                edge_schedule[(edge, day)] -= 1
                if edge_schedule[(edge, day)] <= 0:
                    del edge_schedule[(edge, day)]
            services = self.pod_services(obsolete_pod)
            for pad_id, astronaut_type, _, _ in services:
                service_counts[(pad_id, astronaut_type)] -= 1
            budget += POD_REFUND
        return budget

    def obsolete_pod(self, service_counts: Counter[tuple[int, int]], blocked_pod_ids: set[int]) -> Pod | None:
        """Finds one current non-dedicated pod whose landing-pad services remain covered if it is removed."""
        for pod in sorted(self.pods.values(), key=lambda item: item.id):
            if pod.id in blocked_pod_ids or len(pod.path) == 3 and pod.path[0] == pod.path[2]:
                continue
            services = self.pod_services(pod)
            if services and all(service_counts[(pad_id, astronaut_type)] > 1 for pad_id, astronaut_type, _, _ in services):
                return pod
        return None

    def debug_month_input(self, route_count: int, pod_count: int, new_buildings: list[Building]):
        """Prints the parsed monthly input snapshot to the debug log."""
        pads = self.get_landing_pads()
        total_demand = sum(sum(pad.demand.values()) for pad in pads)
        message = f"[M{self.month + 1:02d}] input resources={self.resources} route_lines={route_count} tubes={len(self.tubes)} "
        message += f"teleports={len(self.teleports)} pod_lines={pod_count} pods={len(self.pods)} new_buildings={len(new_buildings)} "
        message += f"total_buildings={len(self.buildings)} pads={len(pads)} total_monthly_demand={total_demand}"
        print(message, file=sys.stderr)
        for building in sorted(self.buildings.values(), key=lambda item: item.id):
            print(f"[M{self.month + 1:02d}] input node {format_debug_node(building)}", file=sys.stderr)
        for a, b in sorted(self.tubes):
            print(f"[M{self.month + 1:02d}] input tube a={a} b={b} capacity={self.tubes[(a, b)]}", file=sys.stderr)
        for a in sorted(self.teleports):
            print(f"[M{self.month + 1:02d}] input teleport in={a} out={self.teleports[a]}", file=sys.stderr)
        self.debug_pair_costs(new_buildings)
        for pod_id in sorted(self.pods):
            path = self.pods[pod_id].path
            path_text = "-".join(map(str, path))
            print(f"[M{self.month + 1:02d}] input pod_route id={pod_id} path={path_text}", file=sys.stderr)

    def debug_pair_costs(self, new_buildings: list[Building]):
        """Prints a bounded set of construction or upgrade costs for unordered building pairs."""
        degrees = self.get_tube_degrees()
        building_ids = sorted(self.buildings)
        new_ids = {building.id for building in new_buildings}
        total_pairs = len(building_ids) * (len(building_ids) - 1) // 2
        printed = set()
        if total_pairs <= DEBUG_PAIR_COST_LIMIT:
            pair_keys = [route_key(a, b) for index, a in enumerate(building_ids) for b in building_ids[index + 1:]]
        else:
            message = f"[M{self.month + 1:02d}] input pair_cost summary total={total_pairs} limit={DEBUG_PAIR_COST_LIMIT} "
            message += "policy=existing,new,cheapest"
            print(message, file=sys.stderr)
            cheapest = sorted((tube_cost(self.buildings[a], self.buildings[b]), route_key(a, b)) for index, a in enumerate(building_ids)
                              for b in building_ids[index + 1:] if route_key(a, b) not in self.tubes)
            pair_keys = sorted(self.tubes) + [route_key(new_id, building_id) for new_id in sorted(new_ids) for building_id in building_ids
                                             if building_id != new_id] + [key for _, key in cheapest]
        for a, b in pair_keys:
            if (a, b) in printed:
                continue
            cost = self.debug_pair_cost_text(a, b, degrees)
            if cost is None:
                continue
            printed.add((a, b))
            print(f"[M{self.month + 1:02d}] input pair_cost ({a}, {b}) -> {cost}", file=sys.stderr)
            if len(printed) >= DEBUG_PAIR_COST_LIMIT:
                break
        if total_pairs > len(printed):
            print(f"[M{self.month + 1:02d}] input pair_cost omitted={total_pairs - len(printed)}", file=sys.stderr)

    def debug_pair_cost_text(self, a: int, b: int, degrees: Counter[int]) -> str | None:
        """Formats one pair cost as a build cost, upgrade cost, or impossible marker."""
        key = route_key(a, b)
        if key in self.tubes:
            return str(tube_cost(self.buildings[a], self.buildings[b]) * (self.tubes[key] + 1))
        if degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] >= MAX_TUBES_PER_BUILDING or not self.can_build_tube(a, b, self.tubes, []):
            return None
        return str(tube_cost(self.buildings[a], self.buildings[b]))

    def debug_scores(self, label: str, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int]):
        """Prints estimated score details for a named planning snapshot."""
        score, speed, balance, delivered, _, _ = self.score_from_pods(planned_pods, planned_teleports)
        demand = sum(sum(pad.demand.values()) for pad in self.get_landing_pads())
        message = f"[M{self.month + 1:02d}] plan score_{label}_total={score} speed={speed} diversity={balance} "
        message += f"delivered={delivered}/{demand} stranded={demand - delivered}"
        print(message, file=sys.stderr)

    def score_from_pods(self, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int] | None = None) -> tuple:
        """Estimates monthly score, score components, module arrivals, and service details from a planned pod network."""
        service_paths = self.service_paths_from_adjacency(self.adjacency_from_paths(list(planned_pods.values()), planned_teleports))
        directed_schedule = self.directed_schedule_from_paths(list(planned_pods.values()))
        pod_edges = {edge for edge, _ in directed_schedule}
        queues = {}
        for (pad_id, astronaut_type), path in service_paths.items():
            counts = [0] * len(path)
            counts[0] = self.buildings[pad_id].demand[astronaut_type]
            queues[(pad_id, astronaut_type)] = counts
        module_arrivals = Counter()
        module_balance = Counter()
        service_delivered = Counter()
        service_speed = Counter()
        service_balance = Counter()

        for day in range(MONTH_DAYS):
            for pair, path in service_paths.items():
                self.apply_instant_edges(path, queues[pair], pod_edges)
            self.settle_score_arrivals(day, service_paths, queues, module_arrivals, module_balance, service_delivered, service_speed, service_balance)
            moved = {pair: [0] * len(path) for pair, path in service_paths.items()}
            edge_waiters = {}
            for pair, path in service_paths.items():
                for index, waiting in enumerate(queues[pair][:-1]):
                    edge = (path[index], path[index + 1])
                    if waiting and edge in pod_edges:
                        edge_waiters.setdefault(edge, []).append((pair, index))
            for edge, waiters in edge_waiters.items():
                capacity = 10 * directed_schedule[(edge, day)]
                for pair, index in sorted(waiters):
                    boarded = min(queues[pair][index], capacity)
                    queues[pair][index] -= boarded
                    moved[pair][index + 1] += boarded
                    capacity -= boarded
                    if capacity <= 0:
                        break
            for pair, path in service_paths.items():
                for index, count in enumerate(moved[pair]):
                    queues[pair][index] += count
            self.settle_score_arrivals(day + 1, service_paths, queues, module_arrivals, module_balance, service_delivered, service_speed, service_balance)

        speed = sum(service_speed.values())
        balance = sum(service_balance.values())
        delivered = sum(service_delivered.values())
        return speed + balance, speed, balance, delivered, service_delivered, service_paths

    def settle_score_arrivals(self, day: int, service_paths: dict[tuple[int, int], list[int]], queues: dict[tuple[int, int], list[int]],
                              module_arrivals: Counter[int], module_balance: Counter[int], service_delivered: Counter[tuple[int, int]],
                              service_speed: Counter[tuple[int, int]], service_balance: Counter[tuple[int, int]]):
        """Scores queued passengers that have reached their destination module on a given day."""
        for pair, path in service_paths.items():
            arrived = queues[pair][-1]
            if arrived <= 0:
                continue
            module_id = path[-1]
            queues[pair][-1] = 0
            for _ in range(arrived):
                speed_points = max(0, 50 - day)
                balance_points = max(0, 50 - module_arrivals[module_id])
                service_speed[pair] += speed_points
                service_balance[pair] += balance_points
                service_delivered[pair] += 1
                module_balance[module_id] += balance_points
                module_arrivals[module_id] += 1

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
        return self.adjacency_from_paths([pod.path for pod in self.pods.values() if pod.id != skip_pod_id])

    def adjacency_from_paths(self, paths: list[list[int]], teleports: dict[int, int] | None = None) -> dict[int, list[int]]:
        """Builds directed pod and teleporter reachability from explicit pod paths."""
        adjacency = {}
        for path in paths:
            for a, b in zip(path, path[1:]):
                adjacency.setdefault(a, []).append(b)
        for a, b in (self.teleports if teleports is None else teleports).items():
            adjacency.setdefault(a, []).append(b)
        return adjacency

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

    def directed_schedule_from_paths(self, paths: list[list[int]]) -> Counter[tuple[tuple[int, int], int]]:
        """Counts explicit pod paths departing through each directed tube edge on each day of the lunar month."""
        schedule = Counter()
        for path in paths:
            for edge, day in directed_path_edge_days(path):
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
                               edge_schedule: Counter[tuple[tuple[int, int], int]], service_counts: Counter[tuple[int, int]],
                               planned_pods: dict[int, list[int]], rerouted_pod_ids: set[int], budget: int, pod_ids: set[int]) -> Candidate | None:
        """Finds the strongest affordable service candidate for one pad demand, using current planned routes and remaining budget."""
        best = None
        best_key = None
        planned_adjacency = self.adjacency_from_paths(list(planned_pods.values()))
        for module in self.best_modules(modules_by_type[astronaut_type], pad, module_load):
            candidates = self.service_candidates(pad, module, astronaut_type, count, module_load[module.id], module_load, degrees, teleport_used, tubes,
                                                 edge_schedule, service_counts, planned_pods, planned_adjacency, rerouted_pod_ids, budget, pod_ids)
            for candidate in candidates:
                candidate_key = (candidate.score, candidate.delivered, len(candidate.services) or 1, candidate.efficiency)
                if best is None or candidate_key > best_key:
                    best = candidate
                    best_key = candidate_key
        for candidate in self.multi_service_candidates(pad, astronaut_type, modules_by_type, module_load, degrees, tubes, edge_schedule, service_counts,
                                                       budget, pod_ids):
            candidate_key = (candidate.score, candidate.delivered, len(candidate.services) or 1, candidate.efficiency)
            if best is None or candidate_key > best_key:
                best = candidate
                best_key = candidate_key
        for candidate in self.transfer_service_candidates(pad, astronaut_type, module_load, degrees, tubes, edge_schedule, service_counts,
                                                          planned_adjacency, budget, pod_ids):
            candidate_key = (candidate.score, candidate.delivered, len(candidate.services) or 1, candidate.efficiency)
            if best is None or candidate_key > best_key:
                best = candidate
                best_key = candidate_key
        for candidate in self.strategic_reroute_service_candidates(pad, astronaut_type, modules_by_type, module_load, degrees, tubes, edge_schedule,
                                                                   planned_pods, rerouted_pod_ids, budget, pod_ids):
            candidate_key = (candidate.score, candidate.delivered, len(candidate.services) or 1, candidate.efficiency)
            if best is None or candidate_key > best_key:
                best = candidate
                best_key = candidate_key
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
                star_path = [pad.id]
                for _, _, module in ordered:
                    star_path.extend([module.id, pad.id])
                chain_path = loop_path([pad.id] + [module.id for _, _, module in ordered])
                for path, first_days in ((star_path, [2 * index - 1 for index in range(1, size + 1)]), (chain_path, list(range(1, size + 1)))):
                    new_tubes = unique_new_tubes(path, tubes)
                    if not self.can_add_tubes(new_tubes, degrees, tubes):
                        continue
                    upgrade_cost, upgrades = self.path_upgrade_plan(path, new_tubes, tubes, edge_schedule)
                    cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in new_tubes) + POD_COST + upgrade_cost
                    if cost > budget:
                        continue
                    score = 0
                    delivered_total = 0
                    services = []
                    period = len(path) - 1
                    for first_day, (astronaut_type, count, module) in zip(first_days, ordered):
                        delivered = monthly_pod_deliveries(count, first_day, 1, period)
                        score += monthly_score(delivered, first_day, module_load[module.id], period=period) * self.months_left()
                        delivered_total += delivered
                        services.append((pad.id, astronaut_type, module.id, delivered))
                    first_type, _, first_module = ordered[0]
                    candidate = Candidate(score, cost, pad.id, first_module.id, first_type, path, new_tubes, upgrades, delivered=delivered_total,
                                          services=services)
                    candidates.append(candidate)
        return candidates

    def transfer_service_candidates(self, pad: Building, required_type: int, module_load: Counter[int], degrees: Counter[int],
                                    tubes: dict[tuple[int, int], int], edge_schedule: Counter[tuple[tuple[int, int], int]],
                                    service_counts: Counter[tuple[int, int]], planned_adjacency: dict[int, list[int]], budget: int,
                                    pod_ids: set[int]) -> list[Candidate]:
        """Builds candidates that connect a landing pad to useful modules or already planned shuttles."""
        if len(pod_ids) >= MAX_PODS:
            return []
        entries = []
        for building in sorted(self.buildings.values(), key=lambda item: tube_cost(pad, item)):
            if building.id == pad.id or route_key(pad.id, building.id) in tubes:
                continue
            services = self.reachable_service_entries(pad, building.id, service_counts, planned_adjacency)
            if services:
                entries.append((building, services))
        candidates = []
        for size in range(1, min(3, len(entries)) + 1):
            for ordered in permutations(entries[:8], size):
                if not any(required_type in [astronaut_type for astronaut_type, _, _ in services] for _, services in ordered):
                    continue
                path = [pad.id]
                for entry, _ in ordered:
                    path.extend([entry.id, pad.id])
                new_tubes = unique_new_tubes(path, tubes)
                if not self.can_add_tubes(new_tubes, degrees, tubes):
                    continue
                upgrade_cost, upgrades = self.path_upgrade_plan(path, new_tubes, tubes, edge_schedule)
                cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in new_tubes) + POD_COST + upgrade_cost
                if cost > budget:
                    continue
                score = 0
                delivered_total = 0
                services = []
                period = len(path) - 1
                seen_types = set()
                for entry_index, (_, entry_services) in enumerate(ordered, 1):
                    for astronaut_type, module_id, distance in entry_services:
                        if astronaut_type in seen_types:
                            continue
                        seen_types.add(astronaut_type)
                        first_day = 2 * entry_index - 1 + distance
                        count = pad.demand[astronaut_type]
                        delivered = monthly_pod_deliveries(count, first_day, 1, period)
                        score += monthly_score(delivered, first_day, module_load[module_id], period=period) * self.months_left()
                        delivered_total += delivered
                        services.append((pad.id, astronaut_type, module_id, delivered))
                if not services or all(astronaut_type != required_type for _, astronaut_type, _, _ in services):
                    continue
                _, first_type, first_module, _ = services[0]
                candidates.append(Candidate(score, cost, pad.id, first_module, first_type, path, new_tubes, upgrades, delivered=delivered_total,
                                            services=services))
        return candidates

    def strategic_reroute_service_candidates(self, pad: Building, required_type: int, modules_by_type: dict[int, list[Building]],
                                             module_load: Counter[int], degrees: Counter[int], tubes: dict[tuple[int, int], int],
                                             edge_schedule: Counter[tuple[tuple[int, int], int]], planned_pods: dict[int, list[int]],
                                             rerouted_pod_ids: set[int], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds score-aware bundles that reroute one old pod while adding split shuttles for one unserved demand."""
        if len(pod_ids) + 2 > MAX_PODS:
            return []
        old_score = self.score_from_pods(planned_pods)[0]
        base_paths = self.service_paths_from_adjacency(self.adjacency_from_paths(list(planned_pods.values())))
        candidates = []
        for module in self.best_modules(modules_by_type[required_type], pad, module_load)[:3]:
            route_options = self.split_route_path_options(pad, module, tubes)
            for old_pod in self.reroutable_pods(rerouted_pod_ids)[:12]:
                if old_pod.id not in planned_pods:
                    continue
                replaced_services = self.services_lost_by_removing_pod(old_pod, planned_pods, base_paths, modules_by_type)
                if not replaced_services:
                    continue
                for replacement_path, replacement_services in self.replacement_star_path_options(replaced_services, modules_by_type, module_load)[:24]:
                    for route_paths in route_options:
                        if len(pod_ids) + len(route_paths) > MAX_PODS:
                            continue
                        paths = [replacement_path] + route_paths
                        new_tubes = unique_new_tubes_for_paths(paths, tubes)
                        if not self.can_add_tubes(new_tubes, degrees, tubes):
                            continue
                        upgrade_cost, upgrades = self.combined_upgrade_plan(paths, old_pod, new_tubes, tubes, edge_schedule)
                        cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in new_tubes) + REROUTE_COST + POD_COST * len(route_paths) \
                            + upgrade_cost
                        if cost > budget:
                            continue
                        new_pods = {pod_id: path[:] for pod_id, path in planned_pods.items() if pod_id != old_pod.id}
                        new_pods[old_pod.id] = replacement_path[:]
                        for fake_id, route_path in enumerate(route_paths, MAX_PODS + 1):
                            new_pods[fake_id] = route_path[:]
                        new_score, _, _, _, service_delivered, service_paths = self.score_from_pods(new_pods)
                        if (pad.id, required_type) not in service_paths:
                            continue
                        score_gain = (new_score - old_score) * self.months_left()
                        if score_gain <= 0:
                            continue
                        service_pairs = {(pad.id, required_type)}
                        service_pairs.update((pad_id, astronaut_type) for pad_id, astronaut_type, _, _ in replacement_services)
                        services = []
                        for service_pair in sorted(service_pairs):
                            if service_pair in service_paths:
                                services.append((service_pair[0], service_pair[1], service_paths[service_pair][-1], service_delivered[service_pair]))
                        delivered = sum(service[3] for service in services)
                        candidate = Candidate(score_gain, cost, pad.id, service_paths[(pad.id, required_type)][-1], required_type, replacement_path,
                                              new_tubes, upgrades, delivered=delivered, services=services, reroute_pod_id=old_pod.id,
                                              extra_paths=[path[:] for path in route_paths], replaced_services=replaced_services)
                        candidates.append(candidate)
        return candidates

    def split_route_path_options(self, pad: Building, module: Building, tubes: dict[tuple[int, int], int]) -> list[list[list[int]]]:
        """Lists split-shuttle three-hop routes that may serve a pad through one transfer station."""
        options = []
        first_choices = [building for building in sorted(self.buildings.values(), key=lambda building: tube_cost(pad, building) + tube_cost(building, module))
                         if building.id not in (pad.id, module.id)][:12]
        for first in first_choices:
            second_choices = [building for building in sorted(self.buildings.values(),
                                                              key=lambda building: tube_cost(first, building) + tube_cost(building, module))
                              if building.id not in (pad.id, first.id, module.id)][:12]
            for second in second_choices:
                route = [pad.id, first.id, second.id, module.id]
                if len(set(route)) < len(route):
                    continue
                new_tubes = unique_new_tubes_for_paths([[pad.id, first.id, pad.id], [first.id, second.id, module.id, second.id, first.id]], tubes)
                if not new_tubes:
                    continue
                options.append([[pad.id, first.id, pad.id], [first.id, second.id, module.id, second.id, first.id]])
                if len(options) >= 80:
                    return options
        return options

    def services_lost_by_removing_pod(self, old_pod: Pod, planned_pods: dict[int, list[int]], base_paths: dict[tuple[int, int], list[int]],
                                      modules_by_type: dict[int, list[Building]]) -> list[tuple[int, int, int, int]]:
        """Lists landing-pad services whose current route disappears if one pod is rerouted."""
        if old_pod.path[0] not in self.buildings or self.buildings[old_pod.path[0]].kind != 0:
            return []
        changed_pods = {pod_id: path[:] for pod_id, path in planned_pods.items() if pod_id != old_pod.id}
        changed_paths = self.service_paths_from_adjacency(self.adjacency_from_paths(list(changed_pods.values())))
        services = []
        for pair, path in base_paths.items():
            if pair[0] != old_pod.path[0] or pair in changed_paths or pair[1] not in modules_by_type:
                continue
            services.append((pair[0], pair[1], path[-1], self.buildings[pair[0]].demand[pair[1]]))
        return sorted(services, key=lambda service: (-service[3], service[1]))[:3]

    def replacement_star_path_options(self, replaced_services: list[tuple[int, int, int, int]], modules_by_type: dict[int, list[Building]],
                                      module_load: Counter[int]) -> list[tuple[list[int], list[tuple[int, int, int, int]]]]:
        """Lists star-loop reroutes that restore lost services while preferring less-loaded modules."""
        pad_id = replaced_services[0][0]
        choices = []
        for _, astronaut_type, _, _ in replaced_services:
            modules = sorted(modules_by_type[astronaut_type], key=lambda module: (module_load[module.id], tube_cost(self.buildings[pad_id], module),
                                                                                  module.id))[:2]
            choices.append([(astronaut_type, module) for module in modules])
        options = []
        seen = set()
        for selected_modules in product(*choices):
            for ordered in permutations(list(enumerate(selected_modules)), len(selected_modules)):
                path = [pad_id]
                services = []
                for service_index, (_, module) in ordered:
                    path.extend([module.id, pad_id])
                    old_pad_id, astronaut_type, _, count = replaced_services[service_index]
                    services.append((old_pad_id, astronaut_type, module.id, count))
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                options.append((path, services))
        return sorted(options, key=lambda option: (sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in zip(option[0], option[0][1:])),
                                                  sum(module_load[module_id] for _, _, module_id, _ in option[1]), option[0]))

    def combined_upgrade_plan(self, paths: list[list[int]], old_pod: Pod, new_tubes: list[tuple[int, int]], tubes: dict[tuple[int, int], int],
                              edge_schedule: Counter[tuple[tuple[int, int], int]]) -> tuple[int, list[tuple[int, int]]]:
        """Gets upgrade costs for several new paths after removing one existing pod from the schedule."""
        schedule = self.schedule_without_pod(edge_schedule, old_pod)
        upgrade_cost = 0
        upgrades = []
        for path in paths:
            path_upgrade_cost, path_upgrades = self.path_upgrade_plan(path, new_tubes, tubes, schedule)
            upgrade_cost += path_upgrade_cost
            upgrades.extend(path_upgrades)
            for edge, day in path_edge_days(path):
                schedule[(edge, day)] += 1
        return upgrade_cost, upgrades

    def service_candidates(self, pad: Building, module: Building, astronaut_type: int, count: int, current_load: int, module_load: Counter[int],
                           degrees: Counter[int], teleport_used: set[int], tubes: dict[tuple[int, int], int],
                           edge_schedule: Counter[tuple[tuple[int, int], int]], service_counts: Counter[tuple[int, int]],
                           planned_pods: dict[int, list[int]], planned_adjacency: dict[int, list[int]], rerouted_pod_ids: set[int], budget: int,
                           pod_ids: set[int]) -> list[Candidate]:
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
                if candidate_cost <= budget:
                    score, delivered, services = self.entry_service_bundle(pad, module.id, len(existing_path) - 1, 2 * (len(existing_path) - 1),
                                                                           module_load, service_counts, planned_adjacency)
                    candidate = Candidate(score, candidate_cost, pad.id, module.id, astronaut_type, candidate_path, upgrades=upgrades,
                                          delivered=delivered, services=services)
                    candidates.append(candidate)
                    has_tube_candidate = True
            if len(existing_path) > 2 and self.path_has_pod_coverage(existing_path[:-1], planned_pods):
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
        if self.can_add_tubes(direct_tubes, degrees, tubes):
            route_options.append((path, direct_tubes, 1, 2))
            if len(pod_ids) < MAX_PODS:
                upgrade_cost, upgrades = self.path_upgrade_plan(path, direct_tubes, tubes, edge_schedule)
                direct_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in direct_tubes) + POD_COST + upgrade_cost
                if direct_cost <= budget:
                    score, delivered, services = self.entry_service_bundle(pad, module.id, 1, 2, module_load, service_counts, planned_adjacency)
                    candidates.append(Candidate(score, direct_cost, pad.id, module.id, astronaut_type, path, direct_tubes, upgrades, delivered=delivered,
                                                services=services))
                    has_tube_candidate = True

        if not has_tube_candidate and count >= 8:
            two_hop_options = 0
            for via in self.two_hop_buildings(pad, module):
                if via.id in (pad.id, module.id):
                    continue
                path = [pad.id, via.id, module.id]
                two_hop_tubes = [edge for edge in zip(path, path[1:]) if route_key(edge[0], edge[1]) not in tubes]
                candidate_path = loop_path(path)
                if not self.can_add_tubes(two_hop_tubes, degrees, tubes):
                    continue
                route_options.append((candidate_path, two_hop_tubes, 2, 4))
                two_hop_options += 1
                added_two_hop = False
                if len(pod_ids) < MAX_PODS:
                    upgrade_cost, upgrades = self.path_upgrade_plan(candidate_path, two_hop_tubes, tubes, edge_schedule)
                    two_hop_cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in two_hop_tubes) + POD_COST + upgrade_cost
                    if two_hop_cost <= budget:
                        score, delivered, services = self.entry_service_bundle(pad, module.id, 2, 4, module_load, service_counts, planned_adjacency)
                        candidate = Candidate(score, two_hop_cost, pad.id, module.id, astronaut_type, candidate_path, two_hop_tubes, upgrades,
                                              delivered=delivered, services=services)
                        candidates.append(candidate)
                        added_two_hop = True
                    segment_candidate = self.segment_shuttle_candidate(pad, module, astronaut_type, count, current_load, path, two_hop_tubes, tubes,
                                                                       edge_schedule, budget, pod_ids)
                    if segment_candidate is not None:
                        candidates.append(segment_candidate)
                        added_two_hop = True
                if self.path_has_pod_coverage([pad.id, via.id], planned_pods):
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

    def segment_shuttle_candidate(self, pad: Building, module: Building, astronaut_type: int, count: int, current_load: int, path: list[int],
                                  new_tubes: list[tuple[int, int]], tubes: dict[tuple[int, int], int],
                                  edge_schedule: Counter[tuple[tuple[int, int], int]], budget: int, pod_ids: set[int]) -> Candidate | None:
        """Builds a pipelined candidate with one shuttle pod per segment of a multi-hop route."""
        segment_paths = [[a, b, a] for a, b in zip(path, path[1:])]
        if len(pod_ids) + len(segment_paths) > MAX_PODS:
            return None
        schedule = edge_schedule.copy()
        upgrade_cost = 0
        upgrades = []
        for segment_path in segment_paths:
            segment_upgrade_cost, segment_upgrades = self.path_upgrade_plan(segment_path, new_tubes, tubes, schedule)
            upgrade_cost += segment_upgrade_cost
            upgrades.extend(segment_upgrades)
            for edge, day in path_edge_days(segment_path):
                schedule[(edge, day)] += 1
        cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in new_tubes) + POD_COST * len(segment_paths) + upgrade_cost
        if cost > budget:
            return None
        first_day = 2 * (len(path) - 1) - 1
        delivered = monthly_pod_deliveries(count, first_day, 1, 2)
        score = monthly_score(delivered, first_day, current_load, period=2) * self.months_left()
        services = [(pad.id, astronaut_type, module.id, delivered)]
        return Candidate(score, cost, pad.id, module.id, astronaut_type, segment_paths[0], new_tubes, upgrades, delivered=delivered, services=services,
                         extra_paths=segment_paths[1:])

    def entry_service_bundle(self, pad: Building, entry_id: int, entry_day: int, period: int, module_load: Counter[int],
                             service_counts: Counter[tuple[int, int]], planned_adjacency: dict[int, list[int]]) \
            -> tuple[int, int, list[tuple[int, int, int, int]]]:
        """Estimates services gained after a candidate pod reaches one module connected to the current pod network."""
        score = 0
        delivered_total = 0
        services = []
        for astronaut_type, module_id, distance in self.reachable_service_entries(pad, entry_id, service_counts, planned_adjacency):
            first_day = entry_day + distance
            count = pad.demand[astronaut_type]
            delivered = monthly_pod_deliveries(count, first_day, 1, max(2, period))
            score += monthly_score(delivered, first_day, module_load[module_id], period=max(2, period)) * self.months_left()
            delivered_total += delivered
            services.append((pad.id, astronaut_type, module_id, delivered))
        return score, delivered_total, services

    def reachable_service_entries(self, pad: Building, entry_id: int, service_counts: Counter[tuple[int, int]],
                                  adjacency: dict[int, list[int]]) -> list[tuple[int, int, int]]:
        """Lists unserved pad demand types reachable from one entry module through planned pod and teleporter edges."""
        entries = {}
        queue = deque([entry_id])
        distances = {entry_id: 0}
        while queue:
            building_id = queue.popleft()
            building = self.buildings[building_id]
            if building.kind > 0 and building.kind in pad.demand and (pad.id, building.kind) not in service_counts and building.kind not in entries:
                entries[building.kind] = (building.id, distances[building_id])
            for neighbor_id in adjacency.get(building_id, []):
                if neighbor_id in distances:
                    continue
                distances[neighbor_id] = distances[building_id] + 1
                queue.append(neighbor_id)
        return [(astronaut_type, module_id, distance) for astronaut_type, (module_id, distance) in entries.items()]

    def path_has_pod_coverage(self, path: list[int], planned_pods: dict[int, list[int]]) -> bool:
        """Checks whether planned pods can move passengers along every directed hop of a path."""
        edges = set()
        for planned_path in planned_pods.values():
            edges.update(zip(planned_path, planned_path[1:]))
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

    def can_add_tubes(self, new_tubes: list[tuple[int, int]], degrees: Counter[int], tubes: dict[tuple[int, int], int]) -> bool:
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

    def score_added_path(self, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], old_score: int, path: list[int]) -> int:
        """Returns future score gained by adding one pod path to a planned network."""
        new_pods = {pod_id: pod_path[:] for pod_id, pod_path in planned_pods.items()}
        new_pods[MAX_PODS + 1] = path[:]
        return (self.score_from_pods(new_pods, planned_teleports)[0] - old_score) * self.months_left()

    def best_capacity_candidate(self, serviced: set[tuple[int, int]], tubes: dict[tuple[int, int], int], direct_counts: Counter[tuple[int, int]],
                                edge_schedule: Counter[tuple[tuple[int, int], int]], planned_pods: dict[int, list[int]],
                                planned_teleports: dict[int, int], teleported_pairs: set[tuple[int, int]], budget: int, pod_ids: set[int]) -> Candidate | None:
        """Finds the best affordable extra pod capacity for already served direct routes."""
        best = None
        if len(pod_ids) >= MAX_PODS:
            return None
        old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
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
                    upgrade_cost, upgrades = self.path_upgrade_plan(path, [], tubes, edge_schedule)
                    cost = POD_COST + upgrade_cost
                    score_gain = self.score_added_path(planned_pods, planned_teleports, old_score, path)
                    if cost <= budget and score_gain > 0:
                        candidate = Candidate(score_gain, cost, pad.id, module.id, astronaut_type, path, upgrades=upgrades, delivered=count)
                        if best is None or (candidate.score, candidate.efficiency) > (best.score, best.efficiency):
                            best = candidate
        return best

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
        return self.service_paths_from_adjacency(self.get_pod_adjacency())

    def service_paths_from_adjacency(self, adjacency: dict[int, list[int]]) -> dict[tuple[int, int], list[int]]:
        """Maps each served landing-pad demand to the shortest path through explicit reachability."""
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

    def best_baseline_direct_candidate(self, serviced: set[tuple[int, int]], tubes: dict[tuple[int, int], int],
                                       dedicated_edge_counts: Counter[tuple[int, int]], edge_schedule: Counter[tuple[tuple[int, int], int]],
                                       planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], teleported_pairs: set[tuple[int, int]],
                                       budget: int, pod_ids: set[int]) -> Candidate | None:
        """Finds the best missing dedicated pod for an already useful direct tube."""
        if len(pod_ids) >= MAX_PODS or budget < POD_COST:
            return None
        best = None
        old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
        for pad in self.get_landing_pads():
            for astronaut_type, count in pad.demand.items():
                if (pad.id, astronaut_type) not in serviced or (pad.id, astronaut_type) in teleported_pairs:
                    continue
                for module in self.get_modules_by_type()[astronaut_type]:
                    edge = route_key(pad.id, module.id)
                    if edge not in tubes or dedicated_edge_counts[edge]:
                        continue
                    path = [pad.id, module.id, pad.id]
                    upgrade_cost, upgrades = self.path_upgrade_plan(path, [], tubes, edge_schedule)
                    score = self.score_added_path(planned_pods, planned_teleports, old_score, path)
                    if POD_COST + upgrade_cost > budget or score <= 0:
                        continue
                    candidate = Candidate(score, POD_COST + upgrade_cost, pad.id, module.id, astronaut_type, path, upgrades=upgrades, delivered=count)
                    if best is None or candidate.efficiency > best.efficiency or candidate.efficiency == best.efficiency and candidate.score > best.score:
                        best = candidate
        return best

    def get_dedicated_pod_edge_counts(self) -> Counter[tuple[int, int]]:
        """Counts tube edges that already have a two-stop shuttle pod."""
        counts = Counter()
        for pod in self.pods.values():
            if len(pod.path) == 3 and pod.path[0] == pod.path[2]:
                counts[route_key(pod.path[0], pod.path[1])] += 1
        return counts

    def best_baseline_path_candidate(self, tubes: dict[tuple[int, int], int], dedicated_edge_counts: Counter[tuple[int, int]],
                                     edge_schedule: Counter[tuple[tuple[int, int], int]], planned_pods: dict[int, list[int]],
                                     planned_teleports: dict[int, int], teleported_pairs: set[tuple[int, int]], budget: int,
                                     pod_ids: set[int]) -> Candidate | None:
        """Finds the best missing dedicated shuttle on an edge of an already served path."""
        if len(pod_ids) >= MAX_PODS or budget < POD_COST:
            return None
        best = None
        old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
        for (pad_id, astronaut_type), path in self.get_service_paths().items():
            if (pad_id, astronaut_type) in teleported_pairs:
                continue
            for a, b in zip(path, path[1:]):
                edge = route_key(a, b)
                if edge not in tubes or dedicated_edge_counts[edge]:
                    continue
                count = self.buildings[pad_id].demand[astronaut_type]
                candidate_path = [a, b, a]
                upgrade_cost, upgrades = self.path_upgrade_plan(candidate_path, [], tubes, edge_schedule)
                score = self.score_added_path(planned_pods, planned_teleports, old_score, candidate_path)
                if POD_COST + upgrade_cost > budget or score <= 0:
                    continue
                candidate = Candidate(score, POD_COST + upgrade_cost, pad_id, path[-1], astronaut_type, candidate_path, upgrades=upgrades, delivered=count)
                if best is None or candidate.efficiency > best.efficiency or candidate.efficiency == best.efficiency and candidate.score > best.score:
                    best = candidate
        return best

    def best_baseline_replacement_candidate(self, dedicated_edge_counts: Counter[tuple[int, int]], planned_pods: dict[int, list[int]],
                                            service_counts: Counter[tuple[int, int]], teleported_pairs: set[tuple[int, int]], budget: int,
                                            pod_ids: set[int], rerouted_pod_ids: set[int]) -> Candidate | None:
        """Finds the best bundle replacing one shared pod with missing dedicated edge shuttles."""
        best = None
        old_score = self.score_from_pods(planned_pods)[0]
        best_key = None
        for pod_id, path in planned_pods.items():
            if pod_id not in self.pods or pod_id in rerouted_pod_ids or len(path) == 3 and path[0] == path[2]:
                continue
            services = self.pod_services(Pod(pod_id, path))
            if not services or any((pad_id, astronaut_type) in teleported_pairs for pad_id, astronaut_type, _, _ in services):
                continue
            seen_edges = set()
            missing_paths = []
            for a, b in zip(path, path[1:]):
                edge = route_key(a, b)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                if dedicated_edge_counts[edge] <= 0:
                    missing_paths.append([a, b, a])
            if not missing_paths or len(pod_ids) - 1 + len(missing_paths) > MAX_PODS:
                continue
            cost = REROUTE_COST + POD_COST * (len(missing_paths) - 1)
            if cost > budget:
                continue
            new_pods = {current_id: current_path[:] for current_id, current_path in planned_pods.items() if current_id != pod_id}
            for fake_id, missing_path in enumerate(missing_paths, MAX_PODS + 1):
                new_pods[fake_id] = missing_path
            new_service_paths = self.service_paths_from_adjacency(self.adjacency_from_paths(list(new_pods.values())))
            if any((pad_id, astronaut_type) not in new_service_paths for pad_id, astronaut_type, _, _ in services):
                continue
            score = (self.score_from_pods(new_pods)[0] - old_score) * self.months_left()
            if score <= 0:
                continue
            restored_services = []
            for pad_id, astronaut_type, _, _ in services:
                if service_counts[(pad_id, astronaut_type)] <= 1:
                    restored_services.append((pad_id, astronaut_type, new_service_paths[(pad_id, astronaut_type)][-1],
                                              self.buildings[pad_id].demand[astronaut_type]))
            if restored_services:
                pad_id, astronaut_type, module_id, _ = restored_services[0]
                delivered = sum(service[3] for service in restored_services)
            else:
                pad_id, astronaut_type, module_id, _ = services[0]
                delivered = sum(self.buildings[service[0]].demand[service[1]] for service in services)
            candidate = Candidate(score, cost, pad_id, module_id, astronaut_type, missing_paths[0], delivered=delivered, services=restored_services,
                                  reroute_pod_id=pod_id, extra_paths=missing_paths[1:])
            candidate_key = (candidate.score, len(missing_paths), candidate.efficiency)
            if best is None or candidate_key > best_key:
                best = candidate
                best_key = candidate_key
        return best

    def best_speed_candidate(self, serviced: set[tuple[int, int]], service_counts: Counter[tuple[int, int]], tubes: dict[tuple[int, int], int],
                             direct_counts: Counter[tuple[int, int]], edge_schedule: Counter[tuple[tuple[int, int], int]],
                             dedicated_edge_counts: Counter[tuple[int, int]], planned_pods: dict[int, list[int]],
                             planned_teleports: dict[int, int], rerouted_pod_ids: set[int], teleport_used: set[int],
                             teleported_pairs: set[tuple[int, int]], budget: int, pod_ids: set[int]) -> tuple[str, Candidate] | None:
        """Finds the best currently affordable speed improvement candidate."""
        candidates = []
        replacement_candidate = self.best_baseline_replacement_candidate(dedicated_edge_counts, planned_pods, service_counts, teleported_pairs, budget,
                                                                        pod_ids, rerouted_pod_ids)
        if replacement_candidate is not None:
            candidates.append(("baseline_replace", replacement_candidate))
        baseline_candidate = self.best_baseline_direct_candidate(serviced, tubes, dedicated_edge_counts, edge_schedule, planned_pods, planned_teleports,
                                                                teleported_pairs, budget, pod_ids)
        if baseline_candidate is not None:
            candidates.append(("baseline_pod", baseline_candidate))
        path_candidate = self.best_baseline_path_candidate(tubes, dedicated_edge_counts, edge_schedule, planned_pods, planned_teleports, teleported_pairs,
                                                           budget, pod_ids)
        if path_candidate is not None:
            candidates.append(("baseline_pod", path_candidate))
        if candidates:
            return max(candidates, key=lambda item: (item[0] == "baseline_replace", item[1].score, item[1].efficiency))
        capacity_candidate = self.best_capacity_candidate(serviced, tubes, direct_counts, edge_schedule, planned_pods, planned_teleports, teleported_pairs,
                                                          budget, pod_ids)
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
                        planned_teleports: dict[int, int], tubes: dict[tuple[int, int], int], direct_pod_counts: Counter[tuple[int, int]],
                        edge_schedule: Counter[tuple[tuple[int, int], int]], dedicated_edge_counts: Counter[tuple[int, int]],
        planned_pods: dict[int, list[int]], budget: int, pod_ids: set[int]) -> int:
        """Appends a chosen candidate to the action list, updates planned state, and returns the remaining budget."""
        if candidate.reroute_pod_id is not None:
            old_pod = self.pods[candidate.reroute_pod_id]
            removed_pairs = set()
            actions.append(f"DESTROY {candidate.reroute_pod_id}")
            del planned_pods[candidate.reroute_pod_id]
            if len(old_pod.path) == 3 and old_pod.path[0] == old_pod.path[2]:
                if self.buildings[old_pod.path[0]].kind == 0 and self.buildings[old_pod.path[1]].kind > 0:
                    direct_pod_counts[(old_pod.path[0], old_pod.path[1])] -= 1
                dedicated_edge_counts[route_key(old_pod.path[0], old_pod.path[1])] -= 1
            for edge, day in path_edge_days(old_pod.path):
                edge_schedule[(edge, day)] -= 1
                if edge_schedule[(edge, day)] <= 0:
                    del edge_schedule[(edge, day)]
            for pad_id, astronaut_type, module_id, _ in self.pod_services(old_pod):
                removed_pairs.add((pad_id, astronaut_type))
                service_counts[(pad_id, astronaut_type)] -= 1
                if service_counts[(pad_id, astronaut_type)] <= 0:
                    del service_counts[(pad_id, astronaut_type)]
                    serviced.discard((pad_id, astronaut_type))
                    module_load[module_id] -= self.buildings[pad_id].demand[astronaut_type]
            for pad_id, astronaut_type, module_id, delivered in candidate.replaced_services:
                if (pad_id, astronaut_type) in removed_pairs or service_counts[(pad_id, astronaut_type)] <= 0:
                    continue
                service_counts[(pad_id, astronaut_type)] -= 1
                if service_counts[(pad_id, astronaut_type)] <= 0:
                    del service_counts[(pad_id, astronaut_type)]
                    serviced.discard((pad_id, astronaut_type))
                    module_load[module_id] -= delivered
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
            planned_teleports[a] = b
            teleport_used.add(a)
            teleport_used.add(b)
        created_paths = [candidate.path] + candidate.extra_paths if candidate.path else candidate.extra_paths
        for index, path in enumerate(created_paths):
            pod_id = candidate.reroute_pod_id if index == 0 and candidate.reroute_pod_id is not None else next_pod_id(pod_ids)
            pod_ids.add(pod_id)
            planned_pods[pod_id] = path[:]
            actions.append("POD {} {}".format(pod_id, " ".join(map(str, path))))
            if len(path) == 3 and path[0] == path[2]:
                if self.buildings[path[0]].kind == 0 and self.buildings[path[1]].kind > 0:
                    direct_pod_counts[(path[0], path[1])] += 1
                dedicated_edge_counts[route_key(path[0], path[1])] += 1
            for edge, day in path_edge_days(path):
                edge_schedule[(edge, day)] += 1
        if reason == "service":
            for pad_id, astronaut_type, module_id, delivered in candidate.services or [(candidate.pad_id, candidate.astronaut_type, candidate.module_id,
                                                                                        candidate.delivered)]:
                serviced.add((pad_id, astronaut_type))
                service_counts[(pad_id, astronaut_type)] += 1
                module_load[module_id] += delivered
        elif candidate.path and reason in ("baseline_pod", "baseline_replace", "capacity"):
            for pad_id, astronaut_type, module_id, delivered in candidate.services or [(candidate.pad_id, candidate.astronaut_type, candidate.module_id,
                                                                                        candidate.delivered)]:
                was_serviced = (pad_id, astronaut_type) in serviced
                serviced.add((pad_id, astronaut_type))
                service_counts[(pad_id, astronaut_type)] += 1
                if not was_serviced:
                    module_load[module_id] += delivered
        return budget - candidate.cost

    def months_left(self) -> int:
        """Returns the number of lunar months that can still benefit from new construction."""
        return MAX_MONTHS - self.month

    def min_efficiency(self) -> float:
        """Returns the minimum estimated score per resource worth spending this month."""
        return 0.55 if self.month < 14 else 0.8

    def teleport_threshold(self) -> int:
        """Returns the minimum monthly demand for considering a direct teleporter."""
        if self.month <= 3:
            return 55
        if self.month <= 10:
            return 70
        return 90


def format_debug_node(building: Building) -> str:
    """Formats one building with type, coordinates, and landing-pad demand."""
    if building.kind == 0:
        demand = ",".join(f"{astronaut_type}:{building.demand[astronaut_type]}" for astronaut_type in sorted(building.demand)) or "none"
        return f"id={building.id} kind=landing x={building.x} y={building.y} demand={demand} total={sum(building.demand.values())}"
    return f"id={building.id} kind=module module_type={building.kind} x={building.x} y={building.y}"


def unique_new_tubes(path: list[int], tubes: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    """Gets unique tube segments from a path that are not already present."""
    new_tubes = []
    seen = set()
    for a, b in zip(path, path[1:]):
        key = route_key(a, b)
        if key not in tubes and key not in seen:
            new_tubes.append((a, b))
            seen.add(key)
    return new_tubes


def unique_new_tubes_for_paths(paths: list[list[int]], tubes: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    """Gets unique missing tube segments from several pod paths."""
    new_tubes = []
    seen = set()
    for path in paths:
        for a, b in zip(path, path[1:]):
            key = route_key(a, b)
            if key not in tubes and key not in seen:
                new_tubes.append((a, b))
                seen.add(key)
    return new_tubes


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
