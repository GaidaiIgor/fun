"""Builds a heuristic transport network for Selenia City."""

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
EXACT_PATH_SHORTLIST = 4
EXACT_ROUTE_LIMIT = 48
EXACT_SEARCH_ROUNDS = 8


@dataclass(slots=True)
class Building:
    """Stores a building id, type, coordinates, and monthly demand."""
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
    """Stores an action bundle with cost, score, routes, and deliveries."""
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
    """Maintains city state and chooses monthly construction actions."""
    buildings: dict[int, Building]
    month: int
    resources: int
    score_so_far: int
    tubes: dict[tuple[int, int], int]
    teleports: dict[int, int]
    pods: dict[int, Pod]

    def __init__(self):
        """Initializes empty persistent city state before the first input month."""
        self.buildings = {}
        self.month = 0
        self.resources = 0
        self.score_so_far = 0
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
        self.debug_month_input(new_buildings)

    def choose_actions(self) -> list[str]:
        """Chooses action fragments for the current month."""
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
        baseline_score = self.actual_score_from_pods(planned_pods, planned_teleports, tubes)[0]
        print(self.score_debug_text("before", planned_pods, planned_teleports, tubes), file=sys.stderr)

        exact_actions = []
        exact_pods = {pod_id: path[:] for pod_id, path in planned_pods.items()}
        exact_teleports = dict(planned_teleports)
        exact_tubes = dict(tubes)
        exact_budget = self.exact_plan(exact_actions, exact_pods, exact_teleports, exact_tubes, budget, set(pod_ids))
        exact_score = -1 if exact_budget is None else self.actual_score_from_pods(exact_pods, exact_teleports, exact_tubes)[0]

        for pad, astronaut_type, count in unserved_demands:
            if (pad.id, astronaut_type) in serviced:
                continue
            candidate = self.best_service_candidate(pad, astronaut_type, count, modules_by_type, module_load, degrees, teleport_used, tubes, edge_schedule,
                                                    service_counts, planned_pods, planned_teleports, rerouted_pod_ids, budget, pod_ids)
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

        chosen_score = self.actual_score_from_pods(planned_pods, planned_teleports, tubes)[0]
        if (exact_score, exact_budget or 0) > (chosen_score, budget):
            actions, planned_pods, planned_teleports = exact_actions, exact_pods, exact_teleports
            tubes, budget, chosen_score = exact_tubes, exact_budget, exact_score
        if (chosen_score, budget) <= (baseline_score, self.resources):
            actions = []
            planned_pods = {pod_id: pod.path[:] for pod_id, pod in self.pods.items()}
            planned_teleports = dict(self.teleports)
            tubes = dict(self.tubes)
            budget = self.resources
            chosen_score = baseline_score
        after_score = self.score_debug_text("after", planned_pods, planned_teleports, tubes)
        self.score_so_far += chosen_score
        print(f"resources_after {budget} spent {self.resources - budget} {after_score}", file=sys.stderr)
        return actions

    def exact_plan(self, actions: list[str], planned_pods: dict[int, list[int]], planned_teleports: dict[int, int],
                   tubes: dict[tuple[int, int], int], budget: int, pod_ids: set[int]) -> int | None:
        """Finds exact-scored moves and returns the remaining budget."""
        score = self.actual_score_from_pods
        start_score = score(planned_pods, planned_teleports, tubes)[0]
        degrees = Counter()
        for a, b in tubes:
            degrees[a] += 1
            degrees[b] += 1
        edge_schedule = Counter()
        for path in planned_pods.values():
            for edge, day in path_edge_days(path):
                edge_schedule[(edge, day)] += 1
        teleport_used = set(planned_teleports)
        teleport_used.update(planned_teleports.values())
        destroyed_pod_ids = set()
        budget = self.exact_direct_teleports(actions, planned_pods, planned_teleports, tubes, teleport_used, budget)
        for _ in range(EXACT_SEARCH_ROUNDS):
            candidate = self.best_exact_path_candidate(planned_pods, planned_teleports, tubes, degrees, edge_schedule, budget, pod_ids)
            if candidate is None:
                old_budget = budget
                budget = self.exact_destroy_neutral(actions, planned_pods, tubes, planned_teleports, edge_schedule, pod_ids, destroyed_pod_ids, budget)
                if budget == old_budget:
                    break
                continue
            budget = self.apply_exact_path(candidate, actions, planned_pods, tubes, degrees, edge_schedule, pod_ids, budget)
        return budget if actions and score(planned_pods, planned_teleports, tubes)[0] >= start_score else None

    def exact_direct_teleports(self, actions: list[str], planned_pods: dict[int, list[int]], planned_teleports: dict[int, int],
                               tubes: dict[tuple[int, int], int], teleport_used: set[int], budget: int) -> int:
        """Adds exact-positive teleports for single-type landing pads."""
        modules_by_type = self.get_modules_by_type()
        score = self.actual_score_from_pods
        while budget >= TELEPORT_COST:
            old_score = score(planned_pods, planned_teleports, tubes)[0]
            best = None
            for pad in self.get_landing_pads():
                if len(pad.demand) != 1 or pad.id in teleport_used:
                    continue
                astronaut_type = next(iter(pad.demand))
                for module in modules_by_type[astronaut_type]:
                    if module.id in teleport_used:
                        continue
                    new_teleports = dict(planned_teleports)
                    new_teleports[pad.id] = module.id
                    gain = score(planned_pods, new_teleports, tubes)[0] - old_score
                    if gain > 0 and (best is None or (gain, -tube_cost(pad, module)) > best[0]):
                        best = ((gain, -tube_cost(pad, module)), pad.id, module.id)
            if best is None:
                break
            _, entrance_id, exit_id = best
            actions.append(f"TELEPORT {entrance_id} {exit_id}")
            planned_teleports[entrance_id] = exit_id
            teleport_used.add(entrance_id)
            teleport_used.add(exit_id)
            budget -= TELEPORT_COST
        return budget

    def exact_destroy_neutral(self, actions: list[str], planned_pods: dict[int, list[int]], tubes: dict[tuple[int, int], int],
                              planned_teleports: dict[int, int], edge_schedule: Counter[tuple[tuple[int, int], int]], pod_ids: set[int],
                              destroyed_pod_ids: set[int], budget: int) -> int:
        """Destroys original pods whose removal preserves exact score."""
        score = self.actual_score_from_pods
        while True:
            old_score = score(planned_pods, planned_teleports, tubes)[0]
            neutral_pod_id = None
            for pod_id in sorted(set(self.pods) - destroyed_pod_ids):
                if pod_id not in planned_pods:
                    continue
                new_pods = {current_id: path[:] for current_id, path in planned_pods.items() if current_id != pod_id}
                if score(new_pods, planned_teleports, tubes)[0] >= old_score:
                    neutral_pod_id = pod_id
                    break
            if neutral_pod_id is None:
                return budget
            actions.append(f"DESTROY {neutral_pod_id}")
            destroyed_pod_ids.add(neutral_pod_id)
            pod_ids.discard(neutral_pod_id)
            for edge, day in path_edge_days(planned_pods.pop(neutral_pod_id)):
                edge_schedule[(edge, day)] -= 1
                if edge_schedule[(edge, day)] <= 0:
                    del edge_schedule[(edge, day)]
            budget += POD_REFUND

    def best_exact_path_candidate(self, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], tubes: dict[tuple[int, int], int],
                                  degrees: Counter[int], edge_schedule: Counter[tuple[tuple[int, int], int]], budget: int,
                                  pod_ids: set[int]) -> Candidate | None:
        """Finds the exact-best affordable service loop or shuttle."""
        if len(pod_ids) >= MAX_PODS or budget < POD_COST:
            return None
        score = self.actual_score_from_pods
        fast_old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
        old_score = score(planned_pods, planned_teleports, tubes)[0]
        shortlist = []
        best = None
        for path in self.exact_path_options(tubes):
            new_tubes = unique_new_tubes(path, tubes)
            if not self.can_add_tubes(new_tubes, degrees, tubes):
                continue
            upgrade_cost, upgrades = self.path_upgrade_plan(path, new_tubes, tubes, edge_schedule)
            cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in new_tubes) + POD_COST + upgrade_cost
            if cost > budget:
                continue
            new_pods = {pod_id: pod_path[:] for pod_id, pod_path in planned_pods.items()}
            new_pods[MAX_PODS + 1] = path[:]
            fast_gain = self.score_from_pods(new_pods, planned_teleports)[0] - fast_old_score
            if fast_gain > 0:
                shortlist.append((fast_gain, -cost, path, new_tubes, upgrades, cost))
        for _, _, path, new_tubes, upgrades, cost in sorted(shortlist, reverse=True)[:EXACT_PATH_SHORTLIST]:
            new_pods = {pod_id: pod_path[:] for pod_id, pod_path in planned_pods.items()}
            new_pods[MAX_PODS + 1] = path[:]
            new_tube_state = dict(tubes)
            for a, b in new_tubes:
                new_tube_state[route_key(a, b)] = 1
            for a, b in upgrades:
                new_tube_state[route_key(a, b)] += 1
            gain = score(new_pods, planned_teleports, new_tube_state)[0] - old_score
            if gain <= 0:
                continue
            candidate = Candidate(gain, cost, 0, 0, 0, path, new_tubes, upgrades)
            if best is None or (candidate.score, -candidate.cost) > (best.score, -best.cost):
                best = candidate
        return best

    def exact_path_options(self, tubes: dict[tuple[int, int], int]) -> list[list[int]]:
        """Lists bounded pod routes used by exact search."""
        paths = []
        seen = set()
        modules_by_type = self.get_modules_by_type()
        for pad in self.get_landing_pads():
            module_ids = [min(modules_by_type[astronaut_type], key=lambda item: tube_cost(pad, item)).id for astronaut_type in sorted(pad.demand)]
            if 1 < len(module_ids) <= 3:
                for ordered in permutations(module_ids):
                    path = [pad.id]
                    for module_id in ordered:
                        path.extend([module_id, pad.id])
                    if tuple(path) not in seen:
                        paths.append(path)
                        seen.add(tuple(path))
                    path = loop_path([pad.id, *ordered])
                    if tuple(path) not in seen:
                        paths.append(path)
                        seen.add(tuple(path))
        building_ids = sorted(self.buildings)
        cheap_pairs = sorted((tube_cost(self.buildings[a], self.buildings[b]), route_key(a, b)) for index, a in enumerate(building_ids)
                             for b in building_ids[index + 1:] if route_key(a, b) not in tubes)[:24]
        for a, b in list(tubes) + [pair for _, pair in cheap_pairs]:
            for path in ([a, b, a], [b, a, b]):
                if tuple(path) not in seen:
                    paths.append(path)
                    seen.add(tuple(path))
        for pad in self.get_landing_pads():
            for astronaut_type in pad.demand:
                for module in sorted(modules_by_type[astronaut_type], key=lambda item: tube_cost(pad, item))[:2]:
                    path = [pad.id, module.id, pad.id]
                    if tuple(path) not in seen:
                        paths.append(path)
                        seen.add(tuple(path))
                    for via in sorted((building for building in self.buildings.values() if building.id not in (pad.id, module.id)),
                                      key=lambda item: tube_cost(pad, item) + tube_cost(item, module))[:2]:
                        path = [pad.id, via.id, module.id, via.id, pad.id]
                        if tuple(path) not in seen:
                            paths.append(path)
                            seen.add(tuple(path))
        return paths[:EXACT_ROUTE_LIMIT]

    def apply_exact_path(self, candidate: Candidate, actions: list[str], planned_pods: dict[int, list[int]], tubes: dict[tuple[int, int], int],
                         degrees: Counter[int], edge_schedule: Counter[tuple[tuple[int, int], int]], pod_ids: set[int], budget: int) -> int:
        """Applies one exact-scored pod route and returns remaining budget."""
        for a, b in candidate.tubes:
            actions.append(f"TUBE {a} {b}")
            tubes[route_key(a, b)] = 1
            degrees[a] += 1
            degrees[b] += 1
        for a, b in candidate.upgrades:
            actions.append(f"UPGRADE {a} {b}")
            tubes[route_key(a, b)] += 1
        pod_id = next_pod_id(pod_ids)
        pod_ids.add(pod_id)
        planned_pods[pod_id] = candidate.path[:]
        actions.append("POD {} {}".format(pod_id, " ".join(map(str, candidate.path))))
        for edge, day in path_edge_days(candidate.path):
            edge_schedule[(edge, day)] += 1
        return budget - candidate.cost

    def destroy_obsolete_pods(self, actions: list[str], service_counts: Counter[tuple[int, int]], edge_schedule: Counter[tuple[tuple[int, int], int]],
                              planned_pods: dict[int, list[int]], rerouted_pod_ids: set[int], retired_pod_ids: set[int], budget: int,
                              pod_ids: set[int]) -> int:
        """Destroys covered multi-service pods and returns the updated budget."""
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
        """Finds one non-dedicated pod whose services stay covered after removal."""
        for pod in sorted(self.pods.values(), key=lambda item: item.id):
            if pod.id in blocked_pod_ids or len(pod.path) == 3 and pod.path[0] == pod.path[2]:
                continue
            services = self.pod_services(pod)
            if services and all(service_counts[(pad_id, astronaut_type)] > 1 for pad_id, astronaut_type, _, _ in services):
                return pod
        return None

    def debug_month_input(self, new_buildings: list[Building]):
        """Prints the parsed monthly input snapshot to the debug log."""
        print(f"month {self.month + 1}", file=sys.stderr)
        print(f"resources {self.resources}", file=sys.stderr)
        for building in sorted(self.buildings.values(), key=lambda item: item.id):
            print(format_debug_node(building), file=sys.stderr)
        for a, b in sorted(self.tubes):
            print(f"tube {a} {b} {self.tubes[(a, b)]}", file=sys.stderr)
        for a in sorted(self.teleports):
            print(f"teleport {a} {self.teleports[a]}", file=sys.stderr)
        for pod_id in sorted(self.pods):
            path = self.pods[pod_id].path
            path_text = "-".join(map(str, path))
            print(f"pod {pod_id} {path_text}", file=sys.stderr)
        self.debug_pair_costs(new_buildings)

    def debug_pair_costs(self, new_buildings: list[Building]):
        """Prints bounded construction or upgrade costs for building pairs."""
        degrees = self.get_tube_degrees()
        building_ids = sorted(self.buildings)
        new_ids = {building.id for building in new_buildings}
        total_pairs = len(building_ids) * (len(building_ids) - 1) // 2
        printed = set()
        if total_pairs <= DEBUG_PAIR_COST_LIMIT:
            pair_keys = [route_key(a, b) for index, a in enumerate(building_ids) for b in building_ids[index + 1:]]
        else:
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
            print(f"pair_cost ({a}, {b}) -> {cost}", file=sys.stderr)
            if len(printed) >= DEBUG_PAIR_COST_LIMIT:
                break
        if total_pairs > len(printed):
            print(f"pair_cost omitted {total_pairs - len(printed)}", file=sys.stderr)

    def debug_pair_cost_text(self, a: int, b: int, degrees: Counter[int]) -> str | None:
        """Formats one pair cost as a build cost, upgrade cost, or impossible marker."""
        key = route_key(a, b)
        if key in self.tubes:
            return str(tube_cost(self.buildings[a], self.buildings[b]) * (self.tubes[key] + 1))
        if degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] >= MAX_TUBES_PER_BUILDING or not self.can_build_tube(a, b, self.tubes, []):
            return None
        return str(tube_cost(self.buildings[a], self.buildings[b]))

    def score_debug_text(self, label: str, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int],
                         planned_tubes: dict[tuple[int, int], int] | None = None) -> str:
        """Formats estimated score details for a named planning snapshot."""
        score, speed, balance, delivered, _, _ = self.actual_score_from_pods(planned_pods, planned_teleports, planned_tubes)
        demand = sum(sum(pad.demand.values()) for pad in self.get_landing_pads())
        return f"score_{label} month {score} total_so_far {self.score_so_far + score} speed {speed} diversity {balance} " \
            f"delivered {delivered}/{demand} stranded {demand - delivered}"

    def actual_score_from_pods(self, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int] | None = None,
                               planned_tubes: dict[tuple[int, int], int] | None = None) -> tuple:
        """Simulates monthly score for debug and cumulative reporting."""
        teleports = self.teleports if planned_teleports is None else planned_teleports
        tubes = dict(self.tubes if planned_tubes is None else planned_tubes)
        if planned_tubes is None:
            for path in planned_pods.values():
                for a, b in zip(path, path[1:]):
                    tubes.setdefault(route_key(a, b), 1)
        reverse_edges = {}
        for a, b in tubes:
            reverse_edges.setdefault(a, []).append((b, 1))
            reverse_edges.setdefault(b, []).append((a, 1))
        for a, b in teleports.items():
            reverse_edges.setdefault(b, []).append((a, 0))
        distances = {}
        for astronaut_type in {astronaut_type for pad in self.get_landing_pads() for astronaut_type in pad.demand}:
            distances[astronaut_type] = {building_id: 10 ** 9 for building_id in self.buildings}
            queue = deque()
            for building in self.buildings.values():
                if building.kind == astronaut_type:
                    distances[astronaut_type][building.id] = 0
                    queue.append(building.id)
            while queue:
                building_id = queue.popleft()
                for neighbor_id, cost in reverse_edges.get(building_id, []):
                    distance = distances[astronaut_type][building_id] + cost
                    if distance < distances[astronaut_type][neighbor_id]:
                        distances[astronaut_type][neighbor_id] = distance
                        if cost == 0:
                            queue.appendleft(neighbor_id)
                        else:
                            queue.append(neighbor_id)
        queues = {}
        for pad in self.get_landing_pads():
            for astronaut_type, count in pad.demand.items():
                queues.setdefault(pad.id, Counter())[(pad.id, astronaut_type)] = count
        module_arrivals = Counter()
        service_delivered = Counter()
        service_speed = Counter()
        service_balance = Counter()
        pod_positions = {pod_id: 0 for pod_id in planned_pods}
        capacity_tubes = planned_tubes if planned_tubes is not None else None

        for day in range(MONTH_DAYS):
            self.apply_teleport_phase(queues, distances, teleports)
            self.settle_node_arrivals(day, queues, module_arrivals, service_delivered, service_speed, service_balance)
            self.launch_pods(queues, self.daily_pod_moves(planned_pods, pod_positions, capacity_tubes), distances, pod_positions, planned_pods)
            self.settle_node_arrivals(day + 1, queues, module_arrivals, service_delivered, service_speed, service_balance)

        speed = sum(service_speed.values())
        balance = sum(service_balance.values())
        delivered = sum(service_delivered.values())
        return speed + balance, speed, balance, delivered, service_delivered, {}

    def apply_teleport_phase(self, queues: dict[int, Counter[tuple[int, int]]], distances: dict[int, dict[int, int]], teleports: dict[int, int]):
        """Moves passengers through mandatory useful teleporters."""
        for entrance_id, exit_id in sorted(teleports.items()):
            for group, count in list(queues.get(entrance_id, Counter()).items()):
                _, astronaut_type = group
                if count and distances[astronaut_type][exit_id] <= distances[astronaut_type][entrance_id]:
                    queues.setdefault(exit_id, Counter())[group] += count
                    del queues[entrance_id][group]

    def settle_node_arrivals(self, day: int, queues: dict[int, Counter[tuple[int, int]]], module_arrivals: Counter[int],
                             service_delivered: Counter[tuple[int, int]], service_speed: Counter[tuple[int, int]],
                             service_balance: Counter[tuple[int, int]]):
        """Scores passengers queued at a module of their target type on a given day."""
        for building_id in sorted(queues):
            building = self.buildings[building_id]
            if building.kind <= 0:
                continue
            for group, count in list(queues[building_id].items()):
                _, astronaut_type = group
                if building.kind != astronaut_type:
                    continue
                queues[building_id][group] = 0
                for _ in range(count):
                    service_speed[group] += max(0, 50 - day)
                    service_balance[group] += max(0, 50 - module_arrivals[building_id])
                    service_delivered[group] += 1
                    module_arrivals[building_id] += 1

    def daily_pod_moves(self, planned_pods: dict[int, list[int]], pod_positions: dict[int, int],
                        capacity_tubes: dict[tuple[int, int], int] | None) -> dict[int, tuple[int, int]]:
        """Allocates pods to one daily tube move."""
        requests = {}
        for pod_id, path in planned_pods.items():
            index = pod_positions[pod_id]
            if index >= len(path) - 1:
                if path[0] != path[-1]:
                    continue
                index = 0
                pod_positions[pod_id] = 0
            requests[pod_id] = (path[index], path[index + 1])
        if capacity_tubes is None:
            return requests
        moves = {}
        by_tube = {}
        for pod_id, move in requests.items():
            by_tube.setdefault(route_key(*move), []).append((pod_id, move))
        for edge, pods in by_tube.items():
            for pod_id, move in sorted(pods)[:capacity_tubes.get(edge, 0)]:
                moves[pod_id] = move
        return moves

    def launch_pods(self, queues: dict[int, Counter[tuple[int, int]]], moves: dict[int, tuple[int, int]], distances: dict[int, dict[int, int]],
                    pod_positions: dict[int, int], planned_pods: dict[int, list[int]]):
        """Boards passengers, launches pods, and advances pod positions."""
        onboard = {}
        seats = Counter({pod_id: 10 for pod_id in moves})
        for building_id in sorted(queues):
            for group in sorted(list(queues[building_id])):
                _, astronaut_type = group
                while queues[building_id].get(group, 0) > 0:
                    options = [pod_id for pod_id, (a, b) in moves.items() if a == building_id and seats[pod_id] > 0
                               and distances[astronaut_type][b] < distances[astronaut_type][a]]
                    if not options:
                        break
                    pod_id = min(options)
                    boarded = min(queues[building_id][group], seats[pod_id])
                    queues[building_id][group] -= boarded
                    seats[pod_id] -= boarded
                    onboard.setdefault(pod_id, Counter())[group] += boarded
                    if queues[building_id][group] <= 0:
                        del queues[building_id][group]
        for pod_id, (a, b) in moves.items():
            for group, count in onboard.get(pod_id, Counter()).items():
                queues.setdefault(b, Counter())[group] += count
            pod_positions[pod_id] += 1
            if pod_positions[pod_id] >= len(planned_pods[pod_id]) - 1 and planned_pods[pod_id][0] == planned_pods[pod_id][-1]:
                pod_positions[pod_id] = 0

    def score_from_pods(self, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int] | None = None) -> tuple:
        """Estimates monthly score quickly for optimizer comparisons."""
        service_paths = self.service_paths_from_adjacency(self.adjacency_from_paths(list(planned_pods.values()), planned_teleports))
        directed_schedule = self.directed_schedule_from_paths(list(planned_pods.values()))
        pod_edges = {edge for edge, _ in directed_schedule}
        teleport_edges = set((self.teleports if planned_teleports is None else planned_teleports).items())
        queues = {}
        for (pad_id, astronaut_type), path in service_paths.items():
            counts = [0] * len(path)
            counts[0] = self.buildings[pad_id].demand[astronaut_type]
            queues[(pad_id, astronaut_type)] = counts
        module_arrivals = Counter()
        service_delivered = Counter()
        service_speed = Counter()
        service_balance = Counter()

        for day in range(MONTH_DAYS):
            for pair, path in service_paths.items():
                self.apply_instant_edges(path, queues[pair], pod_edges, teleport_edges)
            self.settle_score_arrivals(day, service_paths, queues, module_arrivals, service_delivered, service_speed, service_balance)
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
            self.settle_score_arrivals(day + 1, service_paths, queues, module_arrivals, service_delivered, service_speed, service_balance)

        speed = sum(service_speed.values())
        balance = sum(service_balance.values())
        delivered = sum(service_delivered.values())
        return speed + balance, speed, balance, delivered, service_delivered, service_paths

    def apply_instant_edges(self, path: list[int], queues: list[int], pod_edges: set[tuple[int, int]], teleport_edges: set[tuple[int, int]]):
        """Moves passengers through non-pod path edges before daily waiting."""
        changed = True
        while changed:
            changed = False
            for index, waiting in enumerate(queues[:-1]):
                edge = (path[index], path[index + 1])
                if waiting and (edge in teleport_edges or edge not in pod_edges):
                    queues[index + 1] += waiting
                    queues[index] = 0
                    changed = True

    def settle_score_arrivals(self, day: int, service_paths: dict[tuple[int, int], list[int]], queues: dict[tuple[int, int], list[int]],
                              module_arrivals: Counter[int], service_delivered: Counter[tuple[int, int]], service_speed: Counter[tuple[int, int]],
                              service_balance: Counter[tuple[int, int]]):
        """Scores passengers that reached their destination module on a given day."""
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
                module_arrivals[module_id] += 1

    def get_serviced_pairs(self) -> set[tuple[int, int]]:
        """Gets landing-pad and type pairs already served by routes."""
        return set(self.get_service_counts())

    def get_service_counts(self) -> Counter[tuple[int, int]]:
        """Counts routes serving each landing-pad and astronaut-type pair."""
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
        """Lists pad, type, module, and distance pairs served by a pod."""
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
        """Maps each reachable pad demand to its nearest module and distance."""
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
        """Builds directed pod and teleporter reachability, optionally skipping one pod."""
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
        """Counts directed tube edge departures by lunar day."""
        schedule = Counter()
        for path in paths:
            for edge, day in directed_path_edge_days(path):
                schedule[(edge, day)] += 1
        return schedule

    def get_unserved_demands(self, serviced: set[tuple[int, int]]) -> list[tuple[Building, int, int]]:
        """Gets unserved monthly demands in priority order."""
        demands = []
        for pad in self.get_landing_pads():
            for astronaut_type, count in pad.demand.items():
                if (pad.id, astronaut_type) not in serviced:
                    demands.append((pad, astronaut_type, count))
        return sorted(demands, key=lambda demand: (-demand[2], demand[0].id, demand[1]))[:260]

    def best_service_candidate(self, pad: Building, astronaut_type: int, count: int, modules_by_type: dict[int, list[Building]], module_load: Counter[int],
                               degrees: Counter[int], teleport_used: set[int], tubes: dict[tuple[int, int], int],
                               edge_schedule: Counter[tuple[tuple[int, int], int]], service_counts: Counter[tuple[int, int]],
                               planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], rerouted_pod_ids: set[int], budget: int,
                               pod_ids: set[int]) -> Candidate | None:
        """Finds the strongest affordable service candidate for one pad demand."""
        best = None
        best_key = None
        planned_adjacency = self.adjacency_from_paths(list(planned_pods.values()), planned_teleports)
        targets = [(module, module_load[module.id]) for module in self.best_modules(modules_by_type[astronaut_type], pad, module_load)]
        for entrance_id, exit_id in planned_teleports.items():
            if self.buildings[exit_id].kind == astronaut_type and entrance_id != pad.id:
                targets.append((self.buildings[entrance_id], module_load[exit_id]))
        for module, current_load in targets:
            candidates = self.service_candidates(pad, module, astronaut_type, count, current_load, module_load, degrees, teleport_used, tubes, edge_schedule,
                                                 service_counts, planned_pods, planned_teleports, planned_adjacency, rerouted_pod_ids, budget, pod_ids)
            for candidate in candidates:
                candidate_key = (candidate.score, candidate.delivered, len(candidate.services) or 1, candidate.efficiency)
                if best is None or candidate_key > best_key:
                    best = candidate
                    best_key = candidate_key
        for candidate in self.multi_service_candidates(pad, astronaut_type, modules_by_type, module_load, degrees, tubes, edge_schedule, service_counts,
                                                       planned_pods, planned_teleports, budget, pod_ids):
            candidate_key = (candidate.score, candidate.delivered, len(candidate.services) or 1, candidate.efficiency)
            if best is None or candidate_key > best_key:
                best = candidate
                best_key = candidate_key
        for candidate in self.transfer_service_candidates(pad, astronaut_type, degrees, tubes, edge_schedule, service_counts, planned_adjacency,
                                                          planned_pods, planned_teleports, budget, pod_ids):
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
        """Orders modules for one pad by load-adjusted distance."""
        return sorted(modules, key=lambda module: (module_load[module.id] // 20, tube_cost(pad, module)))[:4]

    def multi_service_candidates(self, pad: Building, required_type: int, modules_by_type: dict[int, list[Building]], module_load: Counter[int],
                                 degrees: Counter[int], tubes: dict[tuple[int, int], int], edge_schedule: Counter[tuple[tuple[int, int], int]],
                                 service_counts: Counter[tuple[int, int]], planned_pods: dict[int, list[int]],
                                 planned_teleports: dict[int, int], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds one-pod candidates visiting several modules from one pad."""
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
        old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
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
                    score = self.score_added_path(planned_pods, planned_teleports, old_score, path)
                    if score <= 0:
                        continue
                    delivered_total = 0
                    services = []
                    period = len(path) - 1
                    for first_day, (astronaut_type, count, module) in zip(first_days, ordered):
                        delivered = monthly_pod_deliveries(count, first_day, 1, period)
                        delivered_total += delivered
                        services.append((pad.id, astronaut_type, module.id, delivered))
                    first_type, _, first_module = ordered[0]
                    candidate = Candidate(score, cost, pad.id, first_module.id, first_type, path, new_tubes, upgrades, delivered=delivered_total,
                                          services=services)
                    candidates.append(candidate)
        return candidates

    def transfer_service_candidates(self, pad: Building, required_type: int, degrees: Counter[int], tubes: dict[tuple[int, int], int],
                                    edge_schedule: Counter[tuple[tuple[int, int], int]], service_counts: Counter[tuple[int, int]],
                                    planned_adjacency: dict[int, list[int]], planned_pods: dict[int, list[int]],
                                    planned_teleports: dict[int, int], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds candidates connecting a pad to useful modules or planned shuttles."""
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
        old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
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
                score = self.score_added_path(planned_pods, planned_teleports, old_score, path)
                if score <= 0:
                    continue
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
                        delivered_total += delivered
                        services.append((pad.id, astronaut_type, module_id, delivered))
                if not services or all(astronaut_type != required_type for _, astronaut_type, _, _ in services):
                    continue
                _, first_type, first_module, _ = services[0]
                candidates.append(Candidate(score, cost, pad.id, first_module, first_type, path, new_tubes, upgrades, delivered=delivered_total,
                                            services=services))
        return candidates

    def service_candidates(self, pad: Building, module: Building, astronaut_type: int, count: int, current_load: int, module_load: Counter[int],
                           degrees: Counter[int], teleport_used: set[int], tubes: dict[tuple[int, int], int],
                           edge_schedule: Counter[tuple[tuple[int, int], int]], service_counts: Counter[tuple[int, int]],
                           planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], planned_adjacency: dict[int, list[int]],
                           rerouted_pod_ids: set[int], budget: int, pod_ids: set[int]) -> list[Candidate]:
        """Builds affordable route bundles for one pad demand."""
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
                    segment_candidate = self.segment_shuttle_candidate(pad, module, astronaut_type, path, two_hop_tubes, tubes, edge_schedule,
                                                                       planned_pods, planned_teleports, budget, pod_ids)
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

        if (count >= self.teleport_threshold() or not candidates) and pad.id not in teleport_used and module.id not in teleport_used \
                and TELEPORT_COST <= budget:
            score = monthly_teleport_score(count, current_load) * self.months_left()
            candidates.append(Candidate(score, TELEPORT_COST, pad.id, module.id, astronaut_type, teleport=(pad.id, module.id), delivered=count))
        return candidates

    def segment_shuttle_candidate(self, pad: Building, module: Building, astronaut_type: int, path: list[int], new_tubes: list[tuple[int, int]],
                                  tubes: dict[tuple[int, int], int], edge_schedule: Counter[tuple[tuple[int, int], int]],
                                  planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], budget: int,
                                  pod_ids: set[int]) -> Candidate | None:
        """Builds one-shuttle-per-segment route candidates."""
        segment_options = [[[path[0], path[1], path[0]]]]
        for a, b in zip(path[1:], path[2:]):
            segment_options.append([[a, b, a], [b, a, b]])
        if len(pod_ids) + len(segment_options) > MAX_PODS:
            return None
        old_score = self.score_from_pods(planned_pods, planned_teleports)[0]
        best = None
        for selected_paths in product(*segment_options):
            segment_paths = [list(item) for item in selected_paths]
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
                continue
            new_pods = {pod_id: pod_path[:] for pod_id, pod_path in planned_pods.items()}
            for fake_id, segment_path in enumerate(segment_paths, MAX_PODS + 1):
                new_pods[fake_id] = segment_path
            new_score, _, _, _, service_delivered, service_paths = self.score_from_pods(new_pods, planned_teleports)
            if (pad.id, astronaut_type) not in service_paths:
                continue
            score = (new_score - old_score) * self.months_left()
            if score <= 0:
                continue
            delivered = service_delivered[(pad.id, astronaut_type)]
            services = [(pad.id, astronaut_type, service_paths[(pad.id, astronaut_type)][-1], delivered)]
            candidate = Candidate(score, cost, pad.id, module.id, astronaut_type, segment_paths[0], new_tubes, upgrades, delivered=delivered,
                                  services=services, extra_paths=segment_paths[1:])
            if best is None or (candidate.score, candidate.delivered, candidate.efficiency) > (best.score, best.delivered, best.efficiency):
                best = candidate
        return best

    def entry_service_bundle(self, pad: Building, entry_id: int, entry_day: int, period: int, module_load: Counter[int],
                             service_counts: Counter[tuple[int, int]], planned_adjacency: dict[int, list[int]]) \
            -> tuple[int, int, list[tuple[int, int, int, int]]]:
        """Estimates services gained after a pod reaches the current network."""
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
        """Lists pad demand types reachable through planned pod and teleporter edges."""
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
        """Gets the shortest existing tube path between two buildings."""
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
        """Checks whether one tube can be built without crossing or passing through buildings."""
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

    def best_speed_candidate(self, serviced: set[tuple[int, int]], service_counts: Counter[tuple[int, int]], tubes: dict[tuple[int, int], int],
                             direct_counts: Counter[tuple[int, int]], edge_schedule: Counter[tuple[tuple[int, int], int]],
                             dedicated_edge_counts: Counter[tuple[int, int]], planned_pods: dict[int, list[int]],
                             planned_teleports: dict[int, int], rerouted_pod_ids: set[int], teleport_used: set[int],
                             teleported_pairs: set[tuple[int, int]], budget: int, pod_ids: set[int]) -> tuple[str, Candidate] | None:
        """Finds the best currently affordable speed improvement candidate."""
        candidates = []
        baseline_candidate = self.best_baseline_direct_candidate(serviced, tubes, dedicated_edge_counts, edge_schedule, planned_pods, planned_teleports,
                                                                teleported_pairs, budget, pod_ids)
        if baseline_candidate is not None:
            candidates.append(("baseline_pod", baseline_candidate))
        path_candidate = self.best_baseline_path_candidate(tubes, dedicated_edge_counts, edge_schedule, planned_pods, planned_teleports, teleported_pairs,
                                                           budget, pod_ids)
        if path_candidate is not None:
            candidates.append(("baseline_pod", path_candidate))
        if candidates:
            return max(candidates, key=lambda item: (item[1].score, item[1].efficiency))
        capacity_candidate = self.best_capacity_candidate(serviced, tubes, direct_counts, edge_schedule, planned_pods, planned_teleports, teleported_pairs,
                                                          budget, pod_ids)
        if capacity_candidate is not None:
            candidates.append(("capacity", capacity_candidate))
        teleport_candidate = self.best_teleport_speed_candidate(planned_pods, planned_teleports, teleport_used, budget)
        if teleport_candidate is not None:
            candidates.append(("speed_teleport", teleport_candidate))
        if not candidates:
            return None
        if self.months_left() <= 2:
            return max(candidates, key=lambda item: (item[1].score, item[1].efficiency))
        return max(candidates, key=lambda item: (item[1].efficiency, item[1].score))

    def best_teleport_speed_candidate(self, planned_pods: dict[int, list[int]], planned_teleports: dict[int, int], teleport_used: set[int],
                                      budget: int) -> Candidate | None:
        """Finds the best teleporter that improves an already served route in the planned network."""
        if budget < TELEPORT_COST:
            return None
        best = None
        old_score, _, _, _, _, service_paths = self.score_from_pods(planned_pods, planned_teleports)
        for (pad_id, astronaut_type), _ in service_paths.items():
            if pad_id in teleport_used:
                continue
            for module in self.get_modules_by_type()[astronaut_type]:
                if module.id in teleport_used:
                    continue
                new_teleports = dict(planned_teleports)
                new_teleports[pad_id] = module.id
                new_score = self.score_from_pods(planned_pods, new_teleports)[0]
                candidate = Candidate((new_score - old_score) * self.months_left(), TELEPORT_COST, pad_id, module.id, astronaut_type,
                                      teleport=(pad_id, module.id), delivered=self.buildings[pad_id].demand[astronaut_type])
                if candidate.score > 0 and (best is None or candidate.score > best.score):
                    best = candidate
        return best

    def apply_candidate(self, reason: str, candidate: Candidate, actions: list[str], serviced: set[tuple[int, int]],
                        service_counts: Counter[tuple[int, int]], module_load: Counter[int], degrees: Counter[int], teleport_used: set[int],
                        planned_teleports: dict[int, int], tubes: dict[tuple[int, int], int], direct_pod_counts: Counter[tuple[int, int]],
                        edge_schedule: Counter[tuple[tuple[int, int], int]], dedicated_edge_counts: Counter[tuple[int, int]],
        planned_pods: dict[int, list[int]], budget: int, pod_ids: set[int]) -> int:
        """Applies a candidate to planned state and returns the remaining budget."""
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
        elif candidate.path and reason in ("baseline_pod", "capacity"):
            if candidate.services:
                services = candidate.services
            elif reason in ("baseline_pod", "capacity"):
                services = [(candidate.pad_id, candidate.astronaut_type, candidate.module_id, candidate.delivered)]
            else:
                services = []
            for pad_id, astronaut_type, module_id, delivered in services:
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
        return f"landing {building.id} {building.x} {building.y} {demand}"
    return f"module {building.id} {building.kind} {building.x} {building.y}"


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
    """Estimates one month of pod score from deliveries, distance, and module load."""
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
    """Estimates one month of direct-teleporter score into a loaded module."""
    return sum(50 + max(0, 50 - current_load - passenger_ind) for passenger_ind in range(count))


def next_pod_id(used_ids: set[int]) -> int:
    """Returns the smallest available pod id from existing and planned ids."""
    for pod_id in range(1, MAX_PODS + 1):
        if pod_id not in used_ids:
            return pod_id
    raise RuntimeError("No pod identifiers remain")


if __name__ == "__main__":
    Planner().play()
