"""Solves Selenia City by greedily exchanging current resources for simulated monthly score."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import isqrt
from sys import stderr

MONTH_DAYS = 20
MAX_TUBES_PER_BUILDING = 5
MAX_PODS = 500
POD_CAPACITY = 10
POD_COST = 1000
POD_REFUND = 750
REROUTE_COST = POD_COST - POD_REFUND
TELEPORT_COST = 5000
MAX_POD_ADDITIONS = 3
MAX_UPGRADES = 3
MAX_TUBE_HOPS = 4
INF = 10 ** 9

Pair = tuple[int, int]
DirectedPair = tuple[int, int]
Pool = tuple[int, int]


@dataclass(slots=True)
class Building:
    """Stores a building, where demand and order describe landing pad astronauts."""
    id: int
    kind: int
    x: int
    y: int
    demand: Counter[int] = field(default_factory=Counter)
    order: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Pod:
    """Stores a known pod path from the game input."""
    id: int
    path: list[int]


@dataclass(slots=True)
class Passenger:
    """Stores a monthly passenger, where pad_id and index define movement priority."""
    pad_id: int
    index: int
    kind: int

    @property
    def id(self) -> int:
        """Returns the passenger priority id derived from pad_id and index."""
        return self.pad_id * 1000 + self.index


@dataclass(slots=True)
class PodPlan:
    """Stores a projected pod, where service_area drives dynamic routing."""
    id: int
    path: list[int] = field(default_factory=list)
    service_area: set[Pair] = field(default_factory=set)
    dynamic: bool = False


@dataclass(slots=True)
class PodSpec:
    """Describes a pod action, where pod_id zero means creating a fresh pod."""
    pod_id: int
    service_area: frozenset[Pair]


@dataclass(slots=True)
class Bundle:
    """Describes one selectable action bundle for a speed pool."""
    pool: Pool
    rank_cost: int = 0
    tubes: tuple[Pair, ...] = ()
    teleport: Pair = (-1, -1)
    pod_specs: tuple[PodSpec, ...] = ()
    upgrades: tuple[Pair, ...] = ()
    label: str = "empty"

    @property
    def fingerprint(self) -> tuple:
        """Returns a stable identity for bundle comparison."""
        return self.tubes, self.teleport, tuple((spec.pod_id, tuple(sorted(spec.service_area))) for spec in self.pod_specs), self.upgrades


@dataclass(slots=True)
class PlanState:
    """Stores projected infrastructure plus output actions and cost."""
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, PodPlan]
    service_areas: dict[int, set[Pair]]
    actions: list[str] = field(default_factory=list)
    placeholders: list[tuple[int, int]] = field(default_factory=list)
    planned_pods: set[int] = field(default_factory=set)
    cost: int = 0


@dataclass(slots=True)
class SimulationResult:
    """Stores monthly score details and congestion diagnostics."""
    score: int = 0
    speed: int = 0
    diversity: int = 0
    delivered: int = 0
    wait_by_edge: Counter[Pair] = field(default_factory=Counter)
    congestion_by_edge: Counter[Pair] = field(default_factory=Counter)
    dynamic_paths: dict[int, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    """Stores a possible greedy replacement for one pool."""
    pool: Pool
    bundle: Bundle
    score_gain: int
    cost_gain: int
    new_score: int
    new_cost: int

    @property
    def efficiency(self) -> float:
        """Returns score_gain per added resource."""
        if self.cost_gain <= 0:
            return float("inf")
        return self.score_gain / self.cost_gain


class Planner:
    """Maintains cross-month state and chooses actions for Selenia City."""
    buildings: dict[int, Building]
    resources: int
    month: int
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, Pod]
    service_areas: dict[int, set[Pair]]

    def __init__(self):
        """Initializes empty game state before the first input month."""
        self.buildings = {}
        self.resources = 0
        self.month = 0
        self.tubes = {}
        self.teleports = {}
        self.pods = {}
        self.service_areas = {}

    def play(self):
        """Reads months until EOF and prints one action command per month."""
        while True:
            try:
                self.read_month()
            except EOFError:
                return
            actions = self.choose_actions()
            print(";".join(actions) if actions else "WAIT")
            self.month += 1

    def read_month(self):
        """Reads the next month from stdin and updates buildings, routes, pods, and resources."""
        self.resources = int(input())
        self.tubes = {}
        self.teleports = {}
        for _ in range(int(input())):
            a, b, capacity = map(int, input().split())
            if capacity == 0:
                self.teleports[a] = b
            else:
                self.tubes[route_key(a, b)] = capacity
        self.pods = {}
        returned_pod_ids = set()
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            pod_id = values[0]
            returned_pod_ids.add(pod_id)
            if pod_id not in self.service_areas:
                raise AssertionError(f"pod {pod_id} has no persisted service area")
            self.pods[pod_id] = Pod(pod_id, values[2:])
        for pod_id in list(self.service_areas):
            if pod_id not in returned_pod_ids:
                del self.service_areas[pod_id]
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            if values[0] == 0:
                self.buildings[values[1]] = Building(values[1], 0, values[2], values[3], Counter(values[5:]), values[5:])
            else:
                self.buildings[values[1]] = Building(values[1], values[0], values[2], values[3])
        self.print_debug_input()

    def choose_actions(self) -> list[str]:
        """Chooses greedy bundle replacements and returns concrete game actions."""
        selected = {}
        current_state = self.replay_bundles(selected)
        current_result = self.simulate(current_state)
        print(self.score_debug("before", current_result, current_state.cost), file=stderr)
        while True:
            best = self.best_candidate(selected, current_result.score, current_state.cost)
            if best is None or best.new_cost > self.resources:
                break
            selected[best.pool] = best.bundle
            current_state = self.replay_bundles(selected)
            current_result = self.simulate(current_state)
            print(f"selected {best.pool} {best.bundle.label} gain {best.score_gain} cost {best.cost_gain} efficiency {best.efficiency:.3f}", file=stderr)
        final_state = self.replay_bundles(selected)
        final_result = self.simulate(final_state, keep_dynamic_paths=True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths)
        self.service_areas = {pod_id: set(pod.service_area) for pod_id, pod in final_state.pods.items() if pod.service_area}
        print(self.score_debug("after", final_result, final_state.cost), file=stderr)
        action_order = {"TUBE": 0, "TELEPORT": 0, "UPGRADE": 1, "DESTROY": 2, "POD": 3}
        return sorted((action for action in final_state.actions if action), key=lambda action: action_order[action.split()[0]])

    def best_candidate(self, selected: dict[Pool, Bundle], current_score: int, current_cost: int) -> Candidate:
        """Finds the most efficient affordable current-generation candidate for selected."""
        best = None
        for pool in self.speed_pools():
            bundle = self.next_bundle(pool, selected)
            if bundle is None:
                continue
            candidate_selected = dict(selected)
            candidate_selected[pool] = bundle
            try:
                candidate_state = self.replay_bundles(candidate_selected)
            except ValueError:
                continue
            if candidate_state.cost > self.resources:
                continue
            candidate_result = self.simulate(candidate_state)
            score_gain = candidate_result.score - current_score
            cost_gain = candidate_state.cost - current_cost
            if score_gain <= 0:
                continue
            candidate = Candidate(pool, bundle, score_gain, cost_gain, candidate_result.score, candidate_state.cost)
            if best is None or (candidate.efficiency, candidate.score_gain, -candidate.new_cost) > (best.efficiency, best.score_gain, -best.new_cost):
                best = candidate
        return best

    def next_bundle(self, pool: Pool, selected: dict[Pool, Bundle]) -> Bundle:
        """Returns the next improving bundle for pool after the currently selected rank_cost."""
        current = selected.get(pool, Bundle(pool))
        other_selected = {other_pool: bundle for other_pool, bundle in selected.items() if other_pool != pool}
        try:
            other_state = self.replay_bundles(other_selected)
        except ValueError:
            return None
        baseline_selected = dict(other_selected)
        baseline_selected[pool] = current
        try:
            baseline_state = self.replay_bundles(baseline_selected)
            baseline_score = self.simulate(baseline_state).score
            baseline_cost = baseline_state.cost
        except ValueError:
            return None
        for bundle in sorted(self.generate_bundles(pool, other_state), key=lambda item: (item.rank_cost, item.fingerprint)):
            if bundle.rank_cost <= current.rank_cost or bundle.fingerprint == current.fingerprint:
                continue
            candidate_selected = dict(other_selected)
            candidate_selected[pool] = bundle
            try:
                state = self.replay_bundles(candidate_selected)
            except ValueError:
                continue
            score_gain = self.simulate(state).score - baseline_score
            cost_gain = state.cost - baseline_cost
            efficiency = float("inf") if cost_gain <= 0 and score_gain > 0 else score_gain / max(1, cost_gain)
            print(f"bundle: {pool}, {self.bundle_action_text(bundle)}, {score_gain}, {cost_gain}, {efficiency:.3f}", file=stderr)
            if state.cost > self.resources:
                continue
            if score_gain > 0:
                return bundle
        return None

    def generate_bundles(self, pool: Pool, state: PlanState) -> list[Bundle]:
        """Builds representative path, pod, and upgrade bundles for pool."""
        pad_id, astronaut_type = pool
        path_options = self.path_options(pad_id, astronaut_type, state)
        bundles = []
        for label, path, tubes, teleport in path_options:
            if teleport != (-1, -1):
                cost = TELEPORT_COST
                bundles.append(Bundle(pool, cost, teleport=teleport, label=label))
                continue
            base_specs = self.coverage_specs(path, state)
            if base_specs is None:
                continue
            base_bundle = Bundle(pool, self.nominal_cost(tubes, base_specs, (), state), tuple(tubes), pod_specs=tuple(base_specs), label=label)
            bundles.extend(self.expand_pod_upgrade_bundles(base_bundle, state))
        unique = []
        seen = set()
        for bundle in bundles:
            if bundle.fingerprint in seen:
                continue
            unique.append(bundle)
            seen.add(bundle.fingerprint)
        return unique

    def bundle_action_text(self, bundle: Bundle) -> str:
        """Formats bundle actions for debug output."""
        actions = []
        actions.extend(f"TUBE {a} {b}" for a, b in bundle.tubes)
        if bundle.teleport != (-1, -1):
            actions.append(f"TELEPORT {bundle.teleport[0]} {bundle.teleport[1]}")
        actions.extend(f"UPGRADE {a} {b}" for a, b in bundle.upgrades)
        for spec in bundle.pod_specs:
            pod_label = str(spec.pod_id) if spec.pod_id else "NEW"
            area_text = " ".join(f"{a}-{b}" for a, b in sorted(spec.service_area))
            actions.append(f"POD {pod_label} AUTO({area_text})")
        return ";".join(actions) if actions else "WAIT"

    def path_options(self, pad_id: int, astronaut_type: int, state: PlanState) -> list[tuple[str, list[int], list[Pair], Pair]]:
        """Returns candidate infrastructure paths for pad_id and astronaut_type."""
        options = []
        module_ids = [building.id for building in self.buildings.values() if building.kind == astronaut_type]
        existing_path = self.shortest_existing_tube_path(pad_id, module_ids, state.tubes)
        if existing_path:
            options.append(("existing", existing_path, [], (-1, -1)))
        elif path := self.cheapest_connecting_path(pad_id, module_ids, state):
            options.append(("connect", path, unique_new_tubes(path, state.tubes), (-1, -1)))
        existing_length = len(existing_path) - 1 if existing_path else INF
        max_hops = min(existing_length - 1, MAX_TUBE_HOPS)
        for hop_count in range(1, max_hops + 1):
            path = self.cheapest_hop_path(pad_id, module_ids, hop_count, state)
            if path:
                options.append((f"short-{hop_count}", path, unique_new_tubes(path, state.tubes), (-1, -1)))
                break
        used = self.teleport_used_buildings(state.teleports)
        for module_id in sorted(module_ids, key=lambda item: tube_cost(self.buildings[pad_id], self.buildings[item])):
            if pad_id not in used and module_id not in used:
                options.append((f"teleport-{module_id}", [], [], (pad_id, module_id)))
        return options

    def coverage_specs(self, path: list[int], state: PlanState) -> list[PodSpec]:
        """Returns pod specs that make path edges serviced in state."""
        if not path:
            return []
        path_edges = [route_key(a, b) for a, b in zip(path, path[1:])]
        service_counts = self.service_counts(state)
        unserviced = [edge for edge in path_edges if service_counts[edge] == 0]
        if not unserviced:
            return []
        specs = []
        used_pods = set()
        remaining = set(unserviced)
        while remaining:
            edge = min(remaining)
            pod_id = self.best_adjacent_pod(edge, state, used_pods)
            if pod_id:
                area = set(state.service_areas[pod_id])
                area.add(edge)
                if service_area_connected(area):
                    specs.append(PodSpec(pod_id, frozenset(area)))
                    used_pods.add(pod_id)
                    remaining.remove(edge)
                    continue
            area = set(path_edges)
            if not service_area_connected(area):
                return None
            specs.append(PodSpec(0, frozenset(area)))
            break
        return specs

    def best_adjacent_pod(self, edge: Pair, state: PlanState, used_pods: set[int]) -> int:
        """Finds the largest adjacent persisted service pod for edge."""
        candidates = []
        for pod_id, area in state.service_areas.items():
            if pod_id in used_pods:
                continue
            nodes = {node for area_edge in area for node in area_edge}
            if edge[0] in nodes or edge[1] in nodes:
                candidates.append((-len(area), pod_id))
        return min(candidates)[1] if candidates else 0

    def expand_pod_upgrade_bundles(self, base_bundle: Bundle, state: PlanState) -> list[Bundle]:
        """Adds lazy wait-pod and congestion-upgrade variants for base_bundle."""
        bundles = []
        pod_specs = list(base_bundle.pod_specs)
        for pod_level in range(MAX_POD_ADDITIONS + 1):
            pod_bundle = Bundle(base_bundle.pool, self.nominal_cost(base_bundle.tubes, pod_specs, (), state), base_bundle.tubes,
                pod_specs=tuple(pod_specs), label=f"{base_bundle.label}-pods-{pod_level}")
            bundles.extend(self.expand_upgrade_bundles(pod_bundle, state))
            if pod_level == MAX_POD_ADDITIONS:
                break
            try:
                projected = self.replay_bundle_on_state(state, pod_bundle)
            except ValueError:
                break
            result = self.simulate(projected)
            edge = self.best_wait_edge(base_bundle.tubes, result.wait_by_edge)
            if edge == (-1, -1):
                break
            pod_specs.append(PodSpec(0, frozenset({edge})))
        return bundles

    def expand_upgrade_bundles(self, bundle: Bundle, state: PlanState) -> list[Bundle]:
        """Adds congestion-upgrade variants for bundle."""
        bundles = []
        upgrades = []
        for level in range(MAX_UPGRADES + 1):
            rank_cost = self.nominal_cost(bundle.tubes, bundle.pod_specs, upgrades, state)
            bundles.append(Bundle(bundle.pool, rank_cost, bundle.tubes, bundle.teleport, bundle.pod_specs, tuple(upgrades), f"{bundle.label}-up-{level}"))
            if level == MAX_UPGRADES:
                break
            try:
                projected = self.replay_bundle_on_state(state, bundles[-1])
            except ValueError:
                break
            result = self.simulate(projected)
            edge = result.congestion_by_edge.most_common(1)[0][0] if result.congestion_by_edge else (-1, -1)
            if edge == (-1, -1):
                break
            upgrades.append(edge)
        return bundles

    def best_wait_edge(self, tubes: tuple[Pair, ...], wait_by_edge: Counter[Pair]) -> Pair:
        """Returns the highest-wait edge relevant to pool and tubes."""
        if not wait_by_edge:
            return -1, -1
        tube_set = set(tubes)
        if tube_set:
            candidates = [(wait, edge) for edge, wait in wait_by_edge.items() if edge in tube_set]
            if candidates:
                return max(candidates, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1]
        return max(wait_by_edge.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]

    def nominal_cost(self, tubes: tuple[Pair, ...] | list[Pair], specs: tuple[PodSpec, ...] | list[PodSpec],
            upgrades: tuple[Pair, ...] | list[Pair], state: PlanState) -> int:
        """Estimates bundle cost from tubes, specs, upgrades, and state."""
        cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes if route_key(a, b) not in state.tubes)
        cost += sum(0 if spec.pod_id in state.planned_pods else REROUTE_COST if spec.pod_id else POD_COST for spec in specs)
        capacities = dict(state.tubes)
        for a, b in tubes:
            capacities.setdefault(route_key(a, b), 1)
        for edge in upgrades:
            if edge in capacities:
                cost += tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * (capacities[edge] + 1)
                capacities[edge] += 1
        return cost

    def replay_bundle_on_state(self, state: PlanState, bundle: Bundle) -> PlanState:
        """Applies bundle to a copied state for local candidate generation."""
        pods = {pod_id: PodPlan(pod.id, pod.path[:], set(pod.service_area), pod.dynamic) for pod_id, pod in state.pods.items()}
        service_areas = {pod_id: set(area) for pod_id, area in state.service_areas.items()}
        copied = PlanState(dict(state.tubes), dict(state.teleports), pods, service_areas, list(state.actions), list(state.placeholders),
            set(state.planned_pods), state.cost)
        self.apply_bundle(copied, bundle)
        return copied

    def replay_bundles(self, selected: dict[Pool, Bundle]) -> PlanState:
        """Replays selected bundles from the real month-start state."""
        pods = {pod_id: PodPlan(pod_id, pod.path[:], set(self.service_areas[pod_id]), False) for pod_id, pod in self.pods.items()}
        service_areas = {pod_id: set(area) for pod_id, area in self.service_areas.items()}
        state = PlanState(dict(self.tubes), dict(self.teleports), pods, service_areas)
        for pool in sorted(selected):
            self.apply_bundle(state, selected[pool])
        return state

    def apply_bundle(self, state: PlanState, bundle: Bundle):
        """Applies bundle to state, recording cost and action placeholders."""
        degrees = self.tube_degrees(state.tubes)
        for a, b in bundle.tubes:
            key = route_key(a, b)
            if key in state.tubes:
                continue
            if not self.can_build_tube(a, b, state.tubes):
                raise ValueError("invalid tube")
            if degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] >= MAX_TUBES_PER_BUILDING:
                raise ValueError("too many tubes")
            state.tubes[key] = 1
            degrees[a] += 1
            degrees[b] += 1
            state.cost += tube_cost(self.buildings[a], self.buildings[b])
            state.actions.append(f"TUBE {a} {b}")
        if bundle.teleport != (-1, -1):
            a, b = bundle.teleport
            used = self.teleport_used_buildings(state.teleports)
            if a in used or b in used or a == b:
                raise ValueError("invalid teleport")
            state.teleports[a] = b
            state.cost += TELEPORT_COST
            state.actions.append(f"TELEPORT {a} {b}")
        for edge in bundle.upgrades:
            if edge not in state.tubes:
                raise ValueError("missing upgrade tube")
            state.tubes[edge] += 1
            state.cost += tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * state.tubes[edge]
            state.actions.append(f"UPGRADE {edge[0]} {edge[1]}")
        for spec in bundle.pod_specs:
            if spec.pod_id:
                if spec.pod_id not in state.pods:
                    raise ValueError("missing reroute pod")
                del state.pods[spec.pod_id]
                pod_id = spec.pod_id
                if pod_id not in state.planned_pods:
                    state.cost += REROUTE_COST
                    state.actions.append(f"DESTROY {pod_id}")
                    state.planned_pods.add(pod_id)
                    state.placeholders.append((len(state.actions), pod_id))
                    state.actions.append("")
            else:
                pod_id = self.next_pod_id(state.pods)
                state.cost += POD_COST
                state.planned_pods.add(pod_id)
                state.placeholders.append((len(state.actions), pod_id))
                state.actions.append("")
            area = set(spec.service_area)
            if not area or not service_area_connected(area):
                raise ValueError("invalid service area")
            for edge in area:
                if edge not in state.tubes:
                    raise ValueError("service area missing tube")
            state.service_areas[pod_id] = area
            state.pods[pod_id] = PodPlan(pod_id, [], area, True)

    def fill_dynamic_actions(self, state: PlanState, dynamic_paths: dict[int, list[int]]):
        """Replaces pod action placeholders with concrete dynamic_paths."""
        for index, pod_id in state.placeholders:
            path = dynamic_paths[pod_id]
            assert len(path) >= 2, f"dynamic pod {pod_id} produced an empty route"
            state.actions[index] = "POD {} {}".format(pod_id, " ".join(map(str, normalize_month_path(path))))

    def simulate(self, state: PlanState, keep_dynamic_paths: bool = False) -> SimulationResult:
        """Simulates one month of astronaut movement through state."""
        distances = self.distances_to_types(state)
        queues = self.initial_queues(distances)
        result = SimulationResult()
        pod_positions = {pod_id: 0 for pod_id, pod in state.pods.items() if not pod.dynamic}
        dynamic_current = {pod_id: -1 for pod_id, pod in state.pods.items() if pod.dynamic}
        dynamic_paths = {pod_id: [] for pod_id, pod in state.pods.items() if pod.dynamic}
        module_arrivals = Counter()
        for day in range(MONTH_DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, module_arrivals, result)
            demand = self.directed_demand(queues, distances, state.tubes)
            requests, month_over = self.pod_requests(state, pod_positions, dynamic_current, demand)
            if month_over:
                break
            moves = self.allocate_tube_capacity(requests, state.tubes, result)
            self.board_and_launch(queues, distances, state, moves, pod_positions, dynamic_current, dynamic_paths, result)
            self.settle(day + 1, queues, module_arrivals, result)
        if keep_dynamic_paths:
            result.dynamic_paths = {pod_id: normalize_month_path(path) for pod_id, path in dynamic_paths.items()}
        return result

    def distances_to_types(self, state: PlanState) -> dict[int, dict[int, int]]:
        """Calculates shortest distances from every building to each demanded module type."""
        demanded = {kind for pad in self.landing_pads() for kind in pad.demand}
        reverse_edges = {}
        for a, b in state.tubes:
            reverse_edges.setdefault(a, []).append((b, 1))
            reverse_edges.setdefault(b, []).append((a, 1))
        for a, b in state.teleports.items():
            reverse_edges.setdefault(b, []).append((a, 0))
        distances = {}
        for kind in demanded:
            distances[kind] = {building_id: INF for building_id in self.buildings}
            queue = deque()
            for building in self.buildings.values():
                if building.kind == kind:
                    distances[kind][building.id] = 0
                    queue.append(building.id)
            while queue:
                building_id = queue.popleft()
                for neighbor_id, cost in reverse_edges.get(building_id, []):
                    distance = distances[kind][building_id] + cost
                    if distance < distances[kind][neighbor_id]:
                        distances[kind][neighbor_id] = distance
                        if cost:
                            queue.append(neighbor_id)
                        else:
                            queue.appendleft(neighbor_id)
        return distances

    def initial_queues(self, distances: dict[int, dict[int, int]]) -> dict[int, list[Passenger]]:
        """Creates starting passenger queues using distances to drop impossible groups."""
        queues = {}
        for pad in self.landing_pads():
            passengers = [Passenger(pad.id, index, kind) for index, kind in enumerate(pad.order) if distances[kind][pad.id] < INF]
            if passengers:
                queues[pad.id] = passengers
        return queues

    def teleport_phase(self, queues: dict[int, list[Passenger]], distances: dict[int, dict[int, int]], teleports: dict[int, int]):
        """Moves passengers through teleports when teleports reduce or preserve target distance."""
        for entrance_id, exit_id in sorted(teleports.items()):
            if entrance_id not in queues:
                continue
            remaining = []
            for passenger in queues[entrance_id]:
                if distances[passenger.kind][exit_id] <= distances[passenger.kind][entrance_id]:
                    queues.setdefault(exit_id, []).append(passenger)
                else:
                    remaining.append(passenger)
            if remaining:
                queues[entrance_id] = remaining
            else:
                del queues[entrance_id]

    def settle(self, day: int, queues: dict[int, list[Passenger]], module_arrivals: Counter[int], result: SimulationResult):
        """Settles passengers already standing in a matching module on day."""
        for building_id in sorted(list(queues)):
            building = self.buildings[building_id]
            if building.kind <= 0:
                continue
            remaining = []
            for passenger in queues[building_id]:
                if passenger.kind != building.kind:
                    remaining.append(passenger)
                    continue
                speed = max(0, 50 - day)
                diversity = max(0, 50 - module_arrivals[building_id])
                result.score += speed + diversity
                result.speed += speed
                result.diversity += diversity
                result.delivered += 1
                module_arrivals[building_id] += 1
            if remaining:
                queues[building_id] = remaining
            else:
                del queues[building_id]

    def directed_demand(self, queues: dict[int, list[Passenger]], distances: dict[int, dict[int, int]], tubes: dict[Pair, int]) -> Counter[DirectedPair]:
        """Counts best directed tube demand from queues under distances."""
        graph = tube_graph(tubes)
        demand = Counter()
        for building_id, passengers in queues.items():
            for passenger in passengers:
                options = [neighbor_id for neighbor_id in graph.get(building_id, []) if
                    distances[passenger.kind][neighbor_id] < distances[passenger.kind][building_id]]
                if options:
                    target_id = min(options, key=lambda item: (distances[passenger.kind][item], item))
                    demand[building_id, target_id] += 1
        return demand

    def pod_requests(self, state: PlanState, pod_positions: dict[int, int], dynamic_current: dict[int, int],
            demand: Counter[DirectedPair]) -> tuple[dict[int, DirectedPair], bool]:
        """Returns daily pod move requests, with dynamic pods choosing requests after fixed pods."""
        requests = {}
        for pod_id, pod in sorted(state.pods.items()):
            if pod.dynamic:
                continue
            index = pod_positions[pod_id]
            next_index = fixed_next_index(pod.path, index)
            if next_index == index:
                continue
            requests[pod_id] = (pod.path[index], pod.path[next_index])
        if state.pods and any(pod.dynamic for pod in state.pods.values()) and not demand:
            return requests, True
        service_counts = self.service_counts(state)
        full_graph = tube_graph(state.tubes)
        for pod_id, pod in sorted(state.pods.items()):
            if not pod.dynamic:
                continue
            move = self.dynamic_move(pod, dynamic_current[pod_id], demand, service_counts, full_graph)
            if move != (-1, -1):
                requests[pod_id] = move
        return requests, False

    def dynamic_move(self, pod: PodPlan, current_id: int, demand: Counter[DirectedPair], service_counts: Counter[Pair],
            full_graph: dict[int, list[int]]) -> DirectedPair:
        """Chooses one dynamic pod move according to service_area and demand."""
        graph = tube_graph({edge: 1 for edge in pod.service_area})
        area_nodes = set(graph)
        loads = {edge: count for edge, count in demand.items() if route_key(*edge) in pod.service_area}
        active_graph = graph
        if not loads:
            loads = dict(demand)
            active_graph = full_graph
            area_nodes = set(full_graph)
        if not loads:
            return -1, -1
        current = None if current_id == -1 else current_id
        best_edge = (-1, -1)
        best_key = (INF, INF, INF, INF, INF)
        for source_id, target_id in loads:
            distance = 0 if current is None else graph_distance(active_graph, current, source_id)
            key = (-min(loads[source_id, target_id], POD_CAPACITY), distance, service_counts[route_key(source_id, target_id)], source_id, target_id)
            if key < best_key:
                best_key = key
                best_edge = source_id, target_id
        source_id, target_id = best_edge
        if current is None:
            return source_id, target_id
        if current == source_id:
            return source_id, target_id
        if current not in area_nodes:
            active_graph = full_graph
        next_id = next_step(active_graph, current, source_id)
        return current, next_id

    def allocate_tube_capacity(self, requests: dict[int, DirectedPair], tubes: dict[Pair, int], result: SimulationResult) -> dict[int, DirectedPair]:
        """Applies tube capacities to pod requests and records congestion days."""
        moves = {}
        by_tube = {}
        for pod_id, move in requests.items():
            by_tube.setdefault(route_key(*move), []).append((pod_id, move))
        for edge, pods in by_tube.items():
            capacity = tubes[edge]
            if len(pods) > capacity:
                result.congestion_by_edge[edge] += 1
            for pod_id, move in sorted(pods)[:capacity]:
                moves[pod_id] = move
        return moves

    def board_and_launch(self, queues: dict[int, list[Passenger]], distances: dict[int, dict[int, int]], state: PlanState,
            moves: dict[int, DirectedPair], pod_positions: dict[int, int], dynamic_current: dict[int, int],
            dynamic_paths: dict[int, list[int]], result: SimulationResult):
        """Boards passengers into moves, launches pods, and updates wait counters."""
        by_start = {}
        for pod_id, (source_id, target_id) in moves.items():
            by_start.setdefault(source_id, []).append((pod_id, target_id))
        for candidates in by_start.values():
            candidates.sort()
        seats = Counter({pod_id: POD_CAPACITY for pod_id in moves})
        onboard = {}
        for building_id in sorted(list(queues)):
            candidates = by_start.get(building_id, [])
            remaining = []
            for passenger in sorted(queues[building_id], key=lambda item: item.id):
                wanted = self.best_wanted_edge(building_id, passenger.kind, distances, state.tubes)
                chosen_pod = 0
                for pod_id, target_id in candidates:
                    if seats[pod_id] and distances[passenger.kind][target_id] < distances[passenger.kind][building_id]:
                        chosen_pod = pod_id
                        break
                if chosen_pod:
                    seats[chosen_pod] -= 1
                    onboard.setdefault(chosen_pod, []).append(passenger)
                else:
                    remaining.append(passenger)
                    if wanted != (-1, -1):
                        result.wait_by_edge[route_key(*wanted)] += 1
            if remaining:
                queues[building_id] = remaining
            else:
                del queues[building_id]
        for pod_id, (_, target_id) in moves.items():
            if pod_id in pod_positions:
                pod_positions[pod_id] = fixed_next_index(state.pods[pod_id].path, pod_positions[pod_id])
            else:
                source_id = moves[pod_id][0]
                if not dynamic_paths[pod_id]:
                    dynamic_paths[pod_id].append(source_id)
                dynamic_paths[pod_id].append(target_id)
                dynamic_current[pod_id] = target_id
            if onboard.get(pod_id):
                queues.setdefault(target_id, []).extend(onboard[pod_id])

    def best_wanted_edge(self, building_id: int, kind: int, distances: dict[int, dict[int, int]], tubes: dict[Pair, int]) -> DirectedPair:
        """Returns the best directed edge wanted from building_id for kind."""
        options = [neighbor_id for neighbor_id in tube_graph(tubes).get(building_id, []) if distances[kind][neighbor_id] < distances[kind][building_id]]
        if not options:
            return -1, -1
        return building_id, min(options, key=lambda item: (distances[kind][item], item))

    def shortest_existing_tube_path(self, start_id: int, targets: list[int], tubes: dict[Pair, int]) -> list[int]:
        """Finds the shortest existing tube path from start_id to targets."""
        graph = tube_graph(tubes)
        queue = deque([start_id])
        parent = {start_id: start_id}
        target_set = set(targets)
        while queue:
            building_id = queue.popleft()
            if building_id in target_set and building_id != start_id:
                return unwind_path(parent, start_id, building_id)
            for neighbor_id in graph.get(building_id, []):
                if neighbor_id not in parent:
                    parent[neighbor_id] = building_id
                    queue.append(neighbor_id)
        return []

    def cheapest_connecting_path(self, start_id: int, targets: list[int], state: PlanState) -> list[int]:
        """Finds the cheapest infrastructure path from start_id to targets."""
        return self.cheapest_path_with_hop_limit(start_id, targets, MAX_TUBE_HOPS, state)

    def cheapest_hop_path(self, start_id: int, targets: list[int], hop_count: int, state: PlanState) -> list[int]:
        """Finds the cheapest valid path with exactly hop_count tube hops."""
        path = self.cheapest_path_with_hop_limit(start_id, targets, hop_count, state, exact_hops=True)
        return path

    def cheapest_path_with_hop_limit(self, start_id: int, targets: list[int], hop_limit: int, state: PlanState,
            exact_hops: bool = False) -> list[int]:
        """Searches buildable and existing tube paths up to hop_limit from start_id to targets."""
        target_set = set(targets)
        edge_graph = self.build_candidate_edge_graph(state.tubes)
        costs = {(start_id, 0): 0}
        parents = {}
        queue = deque([(start_id, 0)])
        while queue:
            building_id, hops = queue.popleft()
            if hops >= hop_limit:
                continue
            for neighbor_id, edge_cost in edge_graph.get(building_id, []):
                next_hops = hops + 1
                cost = costs[building_id, hops] + edge_cost
                key = neighbor_id, next_hops
                if cost >= costs.get(key, INF):
                    continue
                costs[key] = cost
                parents[key] = building_id, hops
                queue.append(key)
        best_key = None
        for target_id in target_set:
            for hops in range(1, hop_limit + 1):
                if exact_hops and hops != hop_limit or (target_id, hops) not in costs:
                    continue
                key = target_id, hops
                candidate_order = costs[key], tube_cost(self.buildings[start_id], self.buildings[target_id])
                best_order = (INF, INF) if best_key is None else (costs[best_key], tube_cost(self.buildings[start_id], self.buildings[best_key[0]]))
                if candidate_order < best_order:
                    best_key = key
        if best_key is None:
            return []
        path = []
        key = best_key
        while key in parents:
            path.append(key[0])
            key = parents[key]
        path.append(start_id)
        path.reverse()
        return path if self.can_add_tubes(unique_new_tubes(path, state.tubes), state.tubes) else []

    def build_candidate_edge_graph(self, tubes: dict[Pair, int]) -> dict[int, list[tuple[int, int]]]:
        """Builds existing and individually buildable tube edges with infrastructure costs."""
        graph = {building_id: [] for building_id in self.buildings}
        building_ids = sorted(self.buildings)
        for index, a in enumerate(building_ids):
            for b in building_ids[index + 1:]:
                edge = route_key(a, b)
                if edge in tubes:
                    cost = 0
                elif self.can_build_tube(a, b, tubes):
                    cost = tube_cost(self.buildings[a], self.buildings[b])
                else:
                    continue
                graph[a].append((b, cost))
                graph[b].append((a, cost))
        for neighbors in graph.values():
            neighbors.sort(key=lambda item: (item[1], item[0]))
        return graph

    def can_add_tubes(self, tubes: list[Pair], existing_tubes: dict[Pair, int]) -> bool:
        """Checks whether tubes can be added together to existing_tubes."""
        test_tubes = dict(existing_tubes)
        degrees = self.tube_degrees(test_tubes)
        for a, b in tubes:
            key = route_key(a, b)
            if key in test_tubes:
                continue
            if degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] >= MAX_TUBES_PER_BUILDING:
                return False
            if not self.can_build_tube(a, b, test_tubes):
                return False
            test_tubes[key] = 1
            degrees[a] += 1
            degrees[b] += 1
        return True

    def can_build_tube(self, a: int, b: int, tubes: dict[Pair, int]) -> bool:
        """Checks geometry rules for building one tube a-b against tubes."""
        if a == b or route_key(a, b) in tubes:
            return False
        first = self.buildings[a]
        second = self.buildings[b]
        for building in self.buildings.values():
            if building.id not in (a, b) and point_on_segment(building, first, second):
                return False
        for c, d in tubes:
            if len({a, b, c, d}) == 4 and segments_intersect(first, second, self.buildings[c], self.buildings[d]):
                return False
        return True

    def speed_pools(self) -> list[Pool]:
        """Returns all speed pools sorted by landing pad and astronaut type."""
        pools = []
        for pad in self.landing_pads():
            pools.extend((pad.id, kind) for kind in sorted(pad.demand))
        return pools

    def landing_pads(self) -> list[Building]:
        """Returns landing pads sorted by id."""
        return [building for building in sorted(self.buildings.values(), key=lambda item: item.id) if building.kind == 0]

    def tube_degrees(self, tubes: dict[Pair, int]) -> Counter[int]:
        """Counts tube degree for each endpoint in tubes."""
        degrees = Counter()
        for a, b in tubes:
            degrees[a] += 1
            degrees[b] += 1
        return degrees

    def teleport_used_buildings(self, teleports: dict[int, int]) -> set[int]:
        """Returns buildings already used by teleports."""
        used = set(teleports)
        used.update(teleports.values())
        return used

    def service_counts(self, state: PlanState) -> Counter[Pair]:
        """Counts how many persisted service areas contain each tube edge."""
        counts = Counter()
        for area in state.service_areas.values():
            for edge in area:
                counts[edge] += 1
        return counts

    def next_pod_id(self, pods: dict[int, PodPlan]) -> int:
        """Returns the smallest available pod id not present in pods."""
        for pod_id in range(1, MAX_PODS + 1):
            if pod_id not in pods:
                return pod_id
        raise RuntimeError("No pod identifiers remain")

    def print_debug_input(self):
        """Prints a compact month snapshot for debugging."""
        print(f"month {self.month + 1}", file=stderr)
        print(f"resources {self.resources}", file=stderr)
        for building in sorted(self.buildings.values(), key=lambda item: item.id):
            if building.kind == 0:
                demand = ",".join(map(str, building.order)) if building.order else "none"
                print(f"landing {building.id} {building.x} {building.y} {demand}", file=stderr)
            else:
                print(f"module {building.id} {building.kind} {building.x} {building.y}", file=stderr)
        for a, b in sorted(self.tubes):
            print(f"tube {a} {b} {self.tubes[a, b]}", file=stderr)
        for a, b in sorted(self.teleports.items()):
            print(f"teleport {a} {b}", file=stderr)
        for pod_id in sorted(self.pods):
            path_text = "-".join(map(str, self.pods[pod_id].path))
            print(f"pod {pod_id} {path_text}", file=stderr)

    def score_debug(self, label: str, result: SimulationResult, cost: int) -> str:
        """Formats score diagnostics for label, result, and cost."""
        demand = sum(sum(pad.demand.values()) for pad in self.landing_pads())
        return f"score_{label} month {result.score} speed {result.speed} diversity {result.diversity} " \
            f"delivered {result.delivered}/{demand} stranded {demand - result.delivered} resources_before {self.resources} " \
            f"resources_after {self.resources - cost} cost {cost}"


def route_key(a: int, b: int) -> Pair:
    """Returns canonical unordered route key for a and b."""
    return (a, b) if a < b else (b, a)


def tube_cost(a: Building, b: Building) -> int:
    """Returns floored tube cost between buildings a and b."""
    return isqrt(100 * ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)))


def orientation(a: Building, b: Building, c: Building) -> int:
    """Returns signed area orientation for buildings a, b, and c."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def point_on_segment(point: Building, a: Building, b: Building) -> bool:
    """Checks whether point lies on segment a-b."""
    return orientation(a, b, point) == 0 and min(a.x, b.x) <= point.x <= max(a.x, b.x) and min(a.y, b.y) <= point.y <= max(a.y, b.y)


def segments_intersect(a: Building, b: Building, c: Building, d: Building) -> bool:
    """Checks whether segments a-b and c-d intersect."""
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 == 0 and point_on_segment(c, a, b) or o2 == 0 and point_on_segment(d, a, b):
        return True
    if o3 == 0 and point_on_segment(a, c, d) or o4 == 0 and point_on_segment(b, c, d):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def unique_new_tubes(path: list[int], tubes: dict[Pair, int]) -> list[Pair]:
    """Returns new canonical tube edges used by path but missing from tubes."""
    result = []
    seen = set()
    for a, b in zip(path, path[1:]):
        edge = route_key(a, b)
        if edge not in tubes and edge not in seen:
            result.append(edge)
            seen.add(edge)
    return result


def tube_graph(tubes: dict[Pair, int]) -> dict[int, list[int]]:
    """Builds sorted adjacency from tube keys."""
    graph = {}
    for a, b in tubes:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return {node: sorted(neighbors) for node, neighbors in graph.items()}


def graph_distance(graph: dict[int, list[int]], start_id: int, finish_id: int) -> int:
    """Returns shortest edge count from start_id to finish_id in graph."""
    if start_id == finish_id:
        return 0
    queue = deque([(start_id, 0)])
    seen = {start_id}
    while queue:
        building_id, distance = queue.popleft()
        for neighbor_id in graph.get(building_id, []):
            if neighbor_id == finish_id:
                return distance + 1
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                queue.append((neighbor_id, distance + 1))
    return INF


def next_step(graph: dict[int, list[int]], start_id: int, finish_id: int) -> int:
    """Returns the first shortest-path step from start_id toward finish_id in graph."""
    if start_id == finish_id:
        return start_id
    queue = deque([start_id])
    parent = {start_id: start_id}
    while queue and finish_id not in parent:
        building_id = queue.popleft()
        for neighbor_id in graph.get(building_id, []):
            if neighbor_id not in parent:
                parent[neighbor_id] = building_id
                queue.append(neighbor_id)
    step = finish_id
    while parent[step] != start_id:
        step = parent[step]
    return step


def unwind_path(parent: dict[int, int], start_id: int, finish_id: int) -> list[int]:
    """Returns path from start_id to finish_id using parent links."""
    path = [finish_id]
    while path[-1] != start_id:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def service_area_connected(area: set[Pair]) -> bool:
    """Checks whether area edges form one connected component."""
    if not area:
        return False
    first = next(iter(area))
    connected = {first}
    nodes = set(first)
    while len(connected) < len(area):
        for edge in area:
            if edge in connected or edge[0] not in nodes and edge[1] not in nodes:
                continue
            connected.add(edge)
            nodes.update(edge)
            break
        else:
            return False
    return True


def normalize_month_path(path: list[int]) -> list[int]:
    """Extends or trims path to the game month path length."""
    if len(path) >= MONTH_DAYS + 1:
        return path[:MONTH_DAYS + 1]
    if len(path) < 2:
        return path[:]
    edges = list(zip(path, path[1:]))
    if path[0] != path[-1]:
        edges.extend(zip(path[-1:0:-1], path[-2::-1]))
    result = [path[0]]
    for day in range(MONTH_DAYS):
        result.append(edges[day % len(edges)][1])
    return result


def fixed_next_index(path: list[int], index: int) -> int:
    """Returns the next path index for a fixed pod path and current index."""
    if len(path) < 2:
        return index
    if index < len(path) - 1:
        return index + 1
    if path[0] == path[-1]:
        return 1
    return index


if __name__ == "__main__":
    Planner().play()
