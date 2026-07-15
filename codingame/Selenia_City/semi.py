"""Solves Selenia City by greedily exchanging current resources for simulated monthly score."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import inf, isqrt
import sys

MONTH_DAYS = 20
MAX_TUBES_PER_BUILDING = 5
MAX_PODS = 500
POD_CAPACITY = 10
POD_COST = 1000
POD_REFUND = 750
REROUTE_COST = POD_COST - POD_REFUND
TELEPORT_COST = 5000
MAX_TUBE_HOPS = 4
INF = 10 ** 9
OVERRIDE_MONTH = -1
OVERRIDE_COMMAND = "TUBE 0 2; TUBE 2 5; TUBE 2 3; TUBE 3 4; TUBE 1 4; TUBE 4 6; POD 1 AUTO(0-2, 2-3, 2-5); POD 2 AUTO(1-4, 3-4, 4-6)"

Pair = tuple[int, int]
DirectedPair = tuple[int, int]
Pool = tuple[int, int]
PoolOwner = Pool | int


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
    id: int


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
    """Describes one selectable action bundle whose destination, path_length, and path identify its transport route."""
    pool: PoolOwner
    rank_cost: int = 0
    tubes: tuple[Pair, ...] = ()
    teleport: Pair = (-1, -1)
    pod_specs: tuple[PodSpec, ...] = ()
    upgrades: tuple[Pair, ...] = ()
    label: str = "empty"
    path_edges: tuple[Pair, ...] = ()
    destination: int = -1
    path_length: int = 0
    path: tuple[int, ...] = ()

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
    planned_pod_edges: dict[int, set[Pair]] = field(default_factory=dict)
    planned_pod_pools: dict[int, PoolOwner] = field(default_factory=dict)
    planned_tubes: set[Pair] = field(default_factory=set)
    cost: int = 0


@dataclass(slots=True)
class SimulationResult:
    """Stores monthly score details and congestion diagnostics."""
    score: int = 0
    speed: int = 0
    diversity: int = 0
    delivered: int = 0
    speed_by_pool: Counter[Pool] = field(default_factory=Counter)
    delivered_by_pool: Counter[Pool] = field(default_factory=Counter)
    delivery_times: dict[Pool, int] = field(default_factory=dict)
    diversity_by_module: Counter[int] = field(default_factory=Counter)
    delivered_by_module: Counter[int] = field(default_factory=Counter)
    wait_by_edge: Counter[Pair] = field(default_factory=Counter)
    preventable_wait_by_edge: Counter[Pair] = field(default_factory=Counter)
    congestion_by_edge: Counter[Pair] = field(default_factory=Counter)
    dynamic_paths: dict[int, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    """Stores a possible greedy replacement whose pair and number identify its debug listing."""
    bundle: Bundle
    pair: PoolOwner
    number: int
    points_gain: int
    cost: int

    @property
    def efficiency(self) -> float:
        """Returns marginal pool points_gain per marginal cost."""
        return self.points_gain / self.cost if self.cost > 0 else inf


class Planner:
    """Maintains cross-month state and chooses actions for Selenia City."""
    buildings: dict[int, Building]
    resources: int
    month: int
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, Pod]
    service_areas: dict[int, set[Pair]]
    simulation_cache: dict[tuple, SimulationResult]

    def __init__(self):
        """Initializes empty game state before the first input month."""
        self.buildings = {}
        self.resources = 0
        self.month = 0
        self.tubes = {}
        self.teleports = {}
        self.pods = {}
        self.service_areas = {}
        self.simulation_cache = {}

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
        """Chooses greedy bundle additions and returns concrete game actions."""
        self.simulation_cache = {}
        if self.month + 1 == OVERRIDE_MONTH:
            return self.override_actions()
        selected = []
        current_state = self.replay_bundle_sequence(selected)
        current_result = self.score_state(current_state)
        before_score = current_result.score
        print("\n" + self.score_debug("before", current_result, current_state.cost), file=sys.stderr)
        while True:
            best = self.best_candidate(selected, current_state, current_result)
            if best is None:
                break
            selected.append(best.bundle)
            current_state = self.replay_bundle_sequence(selected)
            current_result = self.score_state(current_state)
            total_text = self.state_action_text(current_state)
            score_gain = current_result.score - before_score
            efficiency = score_gain / max(1, current_state.cost)
            text = f"selected={best.pair}, bundle {best.number}; bundle={total_text}; gain={score_gain}; cost={current_state.cost}; "
            print(f"{text}efficiency={efficiency:.3f}; "
                f"resources left={self.resources - current_state.cost}", file=sys.stderr)
            print("\n" + self.status_debug(current_result), file=sys.stderr)
        final_state = self.replay_bundle_sequence(selected)
        final_result = self.score_state(final_state, True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths)
        self.service_areas = {pod_id: set(pod.service_area) for pod_id, pod in final_state.pods.items() if pod.service_area}
        print("\n" + self.score_debug("after", final_result, final_state.cost), file=sys.stderr)
        action_order = {"TUBE": 0, "TELEPORT": 0, "UPGRADE": 1, "DESTROY": 2, "POD": 3}
        return sorted((action for action in final_state.actions if action), key=lambda action: action_order[action.split()[0]])

    def override_actions(self) -> list[str]:
        """Applies OVERRIDE_COMMAND for the current month and resolves AUTO pod routes."""
        current_state = self.replay_bundle_sequence([])
        current_result = self.score_state(current_state)
        print("\n" + self.score_debug("before", current_result, current_state.cost), file=sys.stderr)
        final_state = self.override_state(OVERRIDE_COMMAND)
        final_result = self.score_state(final_state, True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths)
        self.service_areas = {pod_id: set(pod.service_area) for pod_id, pod in final_state.pods.items() if pod.service_area}
        print(f"override month {self.month + 1}: {OVERRIDE_COMMAND}", file=sys.stderr)
        print("\n" + self.score_debug("after", final_result, final_state.cost), file=sys.stderr)
        return [action for action in final_state.actions if action]

    def override_state(self, command: str) -> PlanState:
        """Builds a projected state from semicolon-separated override command actions."""
        state = self.replay_bundle_sequence([])
        if command.strip() == "WAIT":
            return state
        for action in (item.strip() for item in command.split(";")):
            if action:
                self.apply_override_action(state, action)
        return state

    def apply_override_action(self, state: PlanState, action: str):
        """Applies one override action to state, leaving AUTO pods as dynamic placeholders."""
        parts = action.split()
        command = parts[0]
        if command == "TUBE":
            a, b = int(parts[1]), int(parts[2])
            edge = route_key(a, b)
            if edge not in state.tubes:
                state.tubes[edge] = 1
                state.cost += tube_cost(self.buildings[a], self.buildings[b])
            state.actions.append(action)
        elif command == "UPGRADE":
            edge = route_key(int(parts[1]), int(parts[2]))
            state.tubes[edge] += 1
            state.cost += tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * state.tubes[edge]
            state.actions.append(action)
        elif command == "TELEPORT":
            a, b = int(parts[1]), int(parts[2])
            state.teleports[a] = b
            state.cost += TELEPORT_COST
            state.actions.append(action)
        elif command == "DESTROY":
            pod_id = int(parts[1])
            del state.pods[pod_id]
            del state.service_areas[pod_id]
            state.cost -= POD_REFUND
            state.actions.append(action)
        elif command == "POD":
            self.apply_override_pod(state, action)
        elif command != "WAIT":
            raise ValueError(f"unknown override action {command}")

    def apply_override_pod(self, state: PlanState, action: str):
        """Applies one override POD action, rerouting an existing pod and resolving AUTO service areas through simulation later."""
        _, pod_text, route_text = action.split(maxsplit=2)
        pod_id = int(pod_text)
        if pod_id in state.pods:
            del state.pods[pod_id]
            del state.service_areas[pod_id]
            state.cost -= POD_REFUND
            state.actions.append(f"DESTROY {pod_id}")
        if route_text.startswith("AUTO("):
            area = parse_auto_area(route_text)
            for edge in area:
                if edge not in state.tubes:
                    raise ValueError("override AUTO edge missing tube")
            state.cost += POD_COST
            state.service_areas[pod_id] = area
            state.pods[pod_id] = PodPlan(pod_id, [], area, True)
            state.placeholders.append((len(state.actions), pod_id))
            state.actions.append("")
            return
        path = [int(item) for item in route_text.split()]
        area = {route_key(a, b) for a, b in zip(path, path[1:])}
        state.cost += POD_COST
        state.service_areas[pod_id] = area
        state.pods[pod_id] = PodPlan(pod_id, path, area, False)
        state.actions.append(action)

    def best_candidate(self, selected: list[Bundle], current_state: PlanState, current_result: SimulationResult) -> Candidate:
        """Finds the best candidate from selected, current_state, and current_result by missing points."""
        pools = []
        for pool in self.speed_pools():
            missing = self.buildings[pool[0]].demand[pool[1]] * 50 - current_result.speed_by_pool[pool]
            if missing > 0:
                modules = sorted(building.id for building in self.buildings.values() if building.kind == pool[1])
                pools.append((missing, (0, pool[0], pool[1]), pool, [(module_id, pool, [module_id]) for module_id in modules]))
        for module in sorted(self.buildings.values(), key=lambda item: item.id):
            if module.kind <= 0:
                continue
            missing = self.max_diversity(module.kind) - current_result.diversity_by_module[module.id]
            if missing <= 0:
                continue
            groups = [pool for pool in self.speed_pools() if pool[1] == module.kind and not current_result.delivered_by_pool[pool]]
            if groups:
                pools.append((missing, (1, module.id), module.id, [(group, group, [module.id]) for group in groups]))
        pools.sort(key=lambda item: (-item[0], item[1]))
        for _, _, owner, pairs in pools:
            print(f"Considering {owner}:", file=sys.stderr)
            best = None
            for pair, group, module_ids in pairs:
                print(f"  Considering {pair}:", file=sys.stderr)
                candidate = self.next_candidate(owner, pair, group, selected, current_state, current_result,
                    self.generate_bundles(owner, group, module_ids, current_state))
                if candidate and (best is None or (candidate.efficiency, candidate.points_gain, -candidate.cost) >
                        (best.efficiency, best.points_gain, -best.cost)):
                    best = candidate
            if best:
                return best
        return None

    def next_candidate(self, owner: PoolOwner, pair: PoolOwner, group: Pool, selected: list[Bundle], current_state: PlanState,
            current_result: SimulationResult, bundles: list[Bundle]) -> Candidate:
        """Returns the most efficient candidate in bundles for owner paired with pair and group after selected."""
        best = None
        seen = set()
        current_pool_score = current_result.speed_by_pool[owner] if isinstance(owner, tuple) else current_result.diversity_by_module[owner]
        plans = []
        for bundle in bundles:
            if bundle.fingerprint == Bundle(owner).fingerprint and not bundle.path_edges:
                continue
            if bundle.path_edges and current_result.delivery_times.get(group, INF) == bundle.path_length:
                continue
            if bundle.fingerprint in seen:
                continue
            seen.add(bundle.fingerprint)
            try:
                state = self.replay_bundle_sequence([*selected, bundle])
            except ValueError:
                continue
            action_text = self.state_delta_text(current_state, state)
            if action_text == "WAIT":
                continue
            plans.append((bundle, state, action_text, state.cost - current_state.cost))
        bundle_number = 0
        for bundle, state, action_text, cost in plans:
            if bundle.label.startswith("short-") and any(other.label.startswith("short-") and other.destination == bundle.destination and
                    other.path_length < bundle.path_length and other_cost <= cost for other, _, _, other_cost in plans):
                continue
            bundle_number += 1
            path_text = ", ".join(map(str, bundle.path))
            text = f"    bundle {bundle_number}: path=[{path_text}], action={action_text}, "
            if state.cost > self.resources:
                print(f"{text}gain=-, cost={cost}, efficiency=-", file=sys.stderr)
                continue
            result = self.score_state(state)
            pool_score = result.speed_by_pool[owner] if isinstance(owner, tuple) else result.diversity_by_module[owner]
            points_gain = pool_score - current_pool_score
            efficiency = points_gain / cost if cost > 0 else inf
            print(f"{text}gain={points_gain}, cost={cost}, efficiency={efficiency:.3f}", file=sys.stderr)
            if points_gain > 0:
                candidate = Candidate(bundle, pair, bundle_number, points_gain, cost)
                if best is None or (candidate.efficiency, candidate.points_gain, -candidate.cost) > \
                        (best.efficiency, best.points_gain, -best.cost):
                    best = candidate
        return best

    def generate_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> list[Bundle]:
        """Builds bundles for owner and group toward module_ids in menu order using state."""
        bundles = []
        pad_id = group[0]
        existing_path = self.shortest_existing_tube_path(pad_id, module_ids, state.tubes)
        if not existing_path:
            connections = self.connection_bundles(owner, group, module_ids, state)
            bundles.extend(connections)
            route_length = connections[0].path_length if connections else 1
            if connections and all(any(not spec.pod_id for spec in bundle.pod_specs) for bundle in connections):
                bundles.extend(self.pod_connection_bundles(owner, group, module_ids, state))
        else:
            route_length = len(existing_path) - 1
        bundles.extend(self.shortest_route_bundles(owner, group, module_ids, route_length, state))
        bundles.extend(self.existing_path_upgrade_bundles(owner, group, module_ids, state))
        bundles.extend(self.teleport_bundles(owner, group, module_ids, state))
        return bundles

    def connection_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> list[Bundle]:
        """Builds owner and group variants of the cheapest connection to module_ids from state."""
        path = self.cheapest_connecting_path(group[0], module_ids, state)
        return self.path_bundles(owner, "connect", path, state)

    def pod_connection_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> list[Bundle]:
        """Builds owner and group variants of the cheapest state connection to module_ids through an existing pod service area."""
        best = []
        best_order = None
        for pod_id, area in sorted(state.service_areas.items()):
            via_nodes = tuple(sorted({node for edge in area for node in edge}))
            path = self.cheapest_path_with_hop_limit(group[0], module_ids, MAX_TUBE_HOPS, state, via_nodes=via_nodes)
            routes = []
            if path:
                routes.append((tuple(route_key(a, b) for a, b in zip(path, path[1:])), path[-1]))
            for module_id in module_ids:
                base_path = self.cheapest_connecting_path(group[0], [module_id], state)
                base_edges = tuple(route_key(a, b) for a, b in zip(base_path, base_path[1:]))
                for junction_id in base_path:
                    remaining_hops = MAX_TUBE_HOPS - len(base_edges)
                    connector = [junction_id] if junction_id in via_nodes else \
                        self.cheapest_path_with_hop_limit(junction_id, list(via_nodes), remaining_hops, state)
                    if not connector:
                        continue
                    edges = tuple(dict.fromkeys((*base_edges, *(route_key(a, b) for a, b in zip(connector, connector[1:])))))
                    if len(edges) <= MAX_TUBE_HOPS and self.can_add_tubes([edge for edge in edges if edge not in state.tubes], state.tubes):
                        routes.append((edges, module_id))
            for path_edges, module_id in routes:
                tubes = tuple(edge for edge in path_edges if edge not in state.tubes)
                specs = (PodSpec(pod_id, frozenset(area | set(path_edges))),)
                cost = self.nominal_cost(tubes, specs, (), state)
                source = self.buildings[group[0]]
                target = self.buildings[module_id]
                distance = (source.x - target.x) * (source.x - target.x) + (source.y - target.y) * (source.y - target.y)
                order = cost, distance, len(path_edges), len(area), path_edges
                projected_tubes = dict(state.tubes)
                projected_tubes.update((edge, 1) for edge in path_edges)
                route = self.shortest_existing_tube_path(group[0], [module_id], projected_tubes)
                bundle = Bundle(owner, cost, tubes, pod_specs=specs, label="connect-pod", path_edges=path_edges, destination=module_id,
                    path_length=len(route) - 1, path=tuple(route))
                if best_order is None or order < best_order:
                    best_order = order
                    best = [bundle]
                elif order == best_order:
                    best.append(bundle)
        return best

    def shortest_route_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], route_length: int,
            state: PlanState) -> list[Bundle]:
        """Builds owner and group exact-length route variants shorter than route_length to every module_id using state."""
        bundles = []
        for hop_count in range(route_length - 1, 0, -1):
            for module_id in module_ids:
                path = self.cheapest_hop_path(group[0], [module_id], hop_count, state)
                bundles.extend(self.path_bundles(owner, f"short-{hop_count}", path, state))
        return bundles

    def existing_path_upgrade_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> list[Bundle]:
        """Builds owner and group focus, pod, upgrade, and combined bundles for the current state path to module_ids."""
        bundles = []
        path = self.shortest_existing_tube_path(group[0], module_ids, state.tubes)
        for base_bundle in self.path_bundles(owner, "existing", path, state):
            if base_bundle.pod_specs:
                bundles.append(base_bundle)
            projected = self.replay_bundle_on_state(state, base_bundle)
            path_edges = base_bundle.path_edges
            for spec in self.focus_specs(path_edges, projected):
                specs = (*base_bundle.pod_specs, spec)
                cost = self.nominal_cost(base_bundle.tubes, specs, (), state)
                bundles.append(Bundle(owner, cost, base_bundle.tubes, pod_specs=specs, label="focus", path_edges=path_edges,
                    destination=base_bundle.destination, path_length=base_bundle.path_length, path=base_bundle.path))
            result = self.cached_simulate(projected)
            pod_edge = self.best_pod_edge(path_edges, result)
            upgrade_edge = self.best_counter_edge(path_edges, result.congestion_by_edge)
            pod_specs = (*base_bundle.pod_specs, PodSpec(0, frozenset({pod_edge}))) if pod_edge != (-1, -1) else base_bundle.pod_specs
            pod_upgrade_edge = (-1, -1)
            pod_affordable = False
            if pod_edge != (-1, -1):
                cost = self.nominal_cost(base_bundle.tubes, pod_specs, (), state)
                pod_bundle = Bundle(owner, cost, base_bundle.tubes, pod_specs=pod_specs, label="pod", path_edges=path_edges,
                    destination=base_bundle.destination, path_length=base_bundle.path_length, path=base_bundle.path)
                bundles.append(pod_bundle)
                pod_state = self.replay_bundle_on_state(state, pod_bundle)
                pod_affordable = pod_state.cost <= self.resources
                if pod_affordable:
                    pod_upgrade_edge = self.best_counter_edge(path_edges, self.cached_simulate(pod_state).congestion_by_edge)
            upgrade_affordable = True
            if upgrade_edge != (-1, -1):
                cost = self.nominal_cost(base_bundle.tubes, base_bundle.pod_specs, (upgrade_edge,), state)
                upgrade_bundle = Bundle(owner, cost, base_bundle.tubes, pod_specs=base_bundle.pod_specs, upgrades=(upgrade_edge,),
                    label="upgrade", path_edges=path_edges, destination=base_bundle.destination, path_length=base_bundle.path_length,
                    path=base_bundle.path)
                bundles.append(upgrade_bundle)
                upgrade_affordable = self.replay_bundle_on_state(state, upgrade_bundle).cost <= self.resources
            if pod_affordable and upgrade_affordable and pod_upgrade_edge != (-1, -1):
                cost = self.nominal_cost(base_bundle.tubes, pod_specs, (pod_upgrade_edge,), state)
                bundles.append(Bundle(owner, cost, base_bundle.tubes, pod_specs=pod_specs, upgrades=(pod_upgrade_edge,), label="pod-upgrade",
                    path_edges=path_edges, destination=base_bundle.destination, path_length=base_bundle.path_length, path=base_bundle.path))
        return bundles

    def teleport_bundles(self, owner: PoolOwner, group: Pool, modules: list[int], state: PlanState) -> list[Bundle]:
        """Builds direct teleporter bundles for owner and group to modules whose endpoints are unused in state."""
        pad_id = group[0]
        used = self.teleport_used_buildings(state.teleports)
        if pad_id in used:
            return []
        return [Bundle(owner, TELEPORT_COST, teleport=(pad_id, module_id), label=f"teleport-{module_id}", destination=module_id,
            path=(pad_id, module_id)) for module_id in sorted(modules, key=lambda item: tube_cost(self.buildings[pad_id], self.buildings[item]))
            if module_id not in used]

    def path_bundles(self, owner: PoolOwner, label: str, path: list[int], state: PlanState) -> list[Bundle]:
        """Builds owner tube-bundle variants named label along path using state."""
        if not path:
            return []
        spec_options = self.coverage_spec_options(path, state)
        tubes = tuple(unique_new_tubes(path, state.tubes))
        path_edges = tuple(route_key(a, b) for a, b in zip(path, path[1:]))
        return [Bundle(owner, self.nominal_cost(tubes, specs, (), state), tubes, pod_specs=tuple(specs), label=label, path_edges=path_edges,
            destination=path[-1], path_length=len(path) - 1, path=tuple(path)) for specs in spec_options]

    def state_action_text(self, state: PlanState) -> str:
        """Formats final projected debug actions from state."""
        actions = []
        for action in state.actions:
            if action and action.split()[0] in ("TUBE", "TELEPORT", "UPGRADE"):
                actions.append(action)
        for pod_id in sorted(state.planned_pods):
            actions.append(self.pod_debug_text(pod_id, state.service_areas[pod_id]))
        return ";".join(actions) if actions else "WAIT"

    def state_delta_text(self, before: PlanState, after: PlanState) -> str:
        """Formats debug actions that turn before into after."""
        actions = []
        for edge in sorted(before.planned_tubes - after.planned_tubes):
            actions.append(f"DROP TUBE {edge[0]} {edge[1]}")
        for edge in sorted(after.planned_tubes - before.planned_tubes):
            actions.append(f"TUBE {edge[0]} {edge[1]}")
        for edge in sorted(set(before.tubes) & set(after.tubes)):
            for _ in range(after.tubes[edge] - before.tubes[edge]):
                actions.append(f"UPGRADE {edge[0]} {edge[1]}")
            for _ in range(before.tubes[edge] - after.tubes[edge]):
                actions.append(f"DROP UPGRADE {edge[0]} {edge[1]}")
        for entrance_id in sorted(set(before.teleports) - set(after.teleports)):
            actions.append(f"DROP TELEPORT {entrance_id} {before.teleports[entrance_id]}")
        for entrance_id in sorted(set(after.teleports) - set(before.teleports)):
            actions.append(f"TELEPORT {entrance_id} {after.teleports[entrance_id]}")
        for pod_id in sorted(before.planned_pods - after.planned_pods):
            actions.append(f"DROP POD {pod_id}")
        for pod_id in sorted(after.planned_pods):
            if pod_id not in before.planned_pods or before.service_areas[pod_id] != after.service_areas[pod_id]:
                actions.append(self.pod_debug_text(pod_id, after.service_areas[pod_id]))
        return ";".join(actions) if actions else "WAIT"

    def pod_debug_text(self, pod_id: int, service_area: set[Pair]) -> str:
        """Formats pod_id and service_area as a debug POD action."""
        area_text = ", ".join(f"{a}-{b}" for a, b in sorted(service_area))
        return f"POD {pod_id} AUTO({area_text})"

    def coverage_spec_options(self, path: list[int], state: PlanState) -> list[list[PodSpec]]:
        """Returns pod-spec variants that make path edges serviced in state."""
        if not path:
            return [[]]
        path_edges = [route_key(a, b) for a, b in zip(path, path[1:])]
        service_counts = self.service_counts(state)
        if all(service_counts[edge] for edge in path_edges):
            return [[]]
        area = set(path_edges)
        pod_ids = self.best_adjacent_pods(area, state)
        if pod_ids:
            return [[PodSpec(pod_id, frozenset(state.service_areas[pod_id] | area))] for pod_id in pod_ids]
        return [[PodSpec(0, frozenset(area))]]

    def best_adjacent_pods(self, edges: set[Pair], state: PlanState) -> list[int]:
        """Finds all equally preferred service pods adjacent to edges in state."""
        edge_nodes = {node for edge in edges for node in edge}
        candidates = []
        for pod_id, area in state.service_areas.items():
            nodes = {node for area_edge in area for node in area_edge}
            if edge_nodes & nodes:
                candidates.append((pod_id not in state.planned_pods, len(area), pod_id))
        if not candidates:
            return []
        best = min(candidate[:2] for candidate in candidates)
        return sorted(candidate[2] for candidate in candidates if candidate[:2] == best)

    def focus_specs(self, path_edges: tuple[Pair, ...], state: PlanState) -> list[PodSpec]:
        """Returns pod specs that remove overlapping service edges from pods serving path_edges."""
        specs = []
        path_edge_set = set(path_edges)
        service_counts = self.service_counts(state)
        for pod_id, area in sorted(state.service_areas.items()):
            if len(area) <= 1 or not area & path_edge_set:
                continue
            focused = area - {edge for edge in area if service_counts[edge] > 1}
            if focused and focused & path_edge_set and focused != area and service_area_connected(focused):
                specs.append(PodSpec(pod_id, frozenset(focused)))
        return specs

    def best_pod_edge(self, path_edges: tuple[Pair, ...], result: SimulationResult) -> Pair:
        """Returns the path edge where the next dynamic pod should focus."""
        candidates = [(result.preventable_wait_by_edge[edge], result.congestion_by_edge[edge], result.wait_by_edge[edge], edge) for edge in path_edges]
        return max(candidates, key=lambda item: (item[0], item[1], item[2], -item[3][0], -item[3][1]))[3] if candidates else (-1, -1)

    def best_counter_edge(self, path_edges: tuple[Pair, ...], counts: Counter[Pair]) -> Pair:
        """Returns the highest-count edge among path_edges."""
        candidates = [(counts[edge], edge) for edge in path_edges if counts[edge]]
        return max(candidates, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1] if candidates else (-1, -1)

    def nominal_cost(self, tubes: tuple[Pair, ...] | list[Pair], specs: tuple[PodSpec, ...] | list[PodSpec],
            upgrades: tuple[Pair, ...] | list[Pair], state: PlanState) -> int:
        """Estimates bundle cost from tubes, specs, upgrades, and state."""
        cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes if route_key(a, b) not in state.tubes)
        planned_pods = set(state.planned_pods)
        for spec in specs:
            if not spec.pod_id:
                cost += POD_COST
            elif spec.pod_id not in planned_pods:
                cost += REROUTE_COST
                planned_pods.add(spec.pod_id)
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
            set(state.planned_pods), {pod_id: set(edges) for pod_id, edges in state.planned_pod_edges.items()},
            dict(state.planned_pod_pools), set(state.planned_tubes), state.cost)
        self.apply_bundle(copied, bundle)
        return copied

    def replay_bundle_sequence(self, selected: list[Bundle]) -> PlanState:
        """Replays selected bundles from the real month-start state."""
        pods = {pod_id: PodPlan(pod_id, pod.path[:], set(self.service_areas[pod_id]), False) for pod_id, pod in self.pods.items()}
        service_areas = {pod_id: set(area) for pod_id, area in self.service_areas.items()}
        state = PlanState(dict(self.tubes), dict(self.teleports), pods, service_areas)
        for bundle in selected:
            self.apply_bundle(state, bundle)
        self.prune_uncommitted_infrastructure(state, selected)
        return state

    def prune_uncommitted_infrastructure(self, state: PlanState, selected: list[Bundle]):
        """Removes planned pods and tubes unused by the latest path for each selected pool."""
        active_edges = set()
        active_by_pool = {}
        for bundle in selected:
            if bundle.path_edges or bundle.teleport != (-1, -1):
                active_by_pool[bundle.pool] = set(bundle.path_edges)
        for edges in active_by_pool.values():
            active_edges.update(edges)
        for pod_id in list(state.planned_pods):
            if pod_id in self.pods:
                original = self.service_areas[pod_id]
                selected = state.planned_pod_edges[pod_id]
                state.service_areas[pod_id] = original | ((selected - original) & active_edges) if original <= selected else \
                    selected if selected & active_edges else set()
                if state.service_areas[pod_id] == original:
                    self.remove_planned_pod(state, pod_id)
                    continue
            else:
                state.service_areas[pod_id] &= active_edges
            if state.service_areas[pod_id]:
                state.pods[pod_id].service_area = state.service_areas[pod_id]
            else:
                self.remove_planned_pod(state, pod_id)
        for edge in sorted(state.planned_tubes - active_edges):
            self.remove_planned_tube(state, edge)

    def remove_planned_pod(self, state: PlanState, pod_id: int):
        """Removes planned pod pod_id or restores its month-start version."""
        if pod_id in self.pods:
            state.cost -= REROUTE_COST
            state.pods[pod_id] = PodPlan(pod_id, self.pods[pod_id].path[:], set(self.service_areas[pod_id]), False)
            state.service_areas[pod_id] = set(self.service_areas[pod_id])
        else:
            state.cost -= POD_COST
            del state.pods[pod_id]
            del state.service_areas[pod_id]
        state.planned_pods.remove(pod_id)
        del state.planned_pod_edges[pod_id]
        del state.planned_pod_pools[pod_id]
        state.placeholders = [(index, placeholder_id) for index, placeholder_id in state.placeholders if placeholder_id != pod_id]
        for index, action in enumerate(state.actions):
            parts = action.split()
            if parts and parts[0] == "DESTROY" and int(parts[1]) == pod_id:
                state.actions[index] = ""

    def remove_planned_tube(self, state: PlanState, edge: Pair):
        """Removes planned tube edge and its planned upgrade costs from state."""
        capacity = state.tubes[edge]
        state.cost -= tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * capacity * (capacity + 1) // 2
        del state.tubes[edge]
        state.planned_tubes.remove(edge)
        for index, action in enumerate(state.actions):
            parts = action.split()
            if parts and parts[0] in ("TUBE", "UPGRADE") and route_key(int(parts[1]), int(parts[2])) == edge:
                state.actions[index] = ""

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
            state.planned_tubes.add(key)
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
            path_edges = set(bundle.path_edges)
            state.planned_pod_edges[pod_id] = area if pod_id in self.pods else area & path_edges if path_edges else set(area)
            state.planned_pod_pools[pod_id] = bundle.pool
            state.service_areas[pod_id] = area
            state.pods[pod_id] = PodPlan(pod_id, [], area, True)

    def fill_dynamic_actions(self, state: PlanState, dynamic_paths: dict[int, list[int]]):
        """Replaces pod action placeholders with concrete dynamic_paths."""
        for index, pod_id in state.placeholders:
            path = dynamic_paths[pod_id]
            assert len(path) >= 2, f"dynamic pod {pod_id} produced an empty route"
            state.actions[index] = "POD {} {}".format(pod_id, " ".join(map(str, normalize_month_path(path))))

    def score_state(self, state: PlanState, keep_dynamic_paths: bool = False) -> SimulationResult:
        """Scores state after replacing dynamic pods with the concrete paths they would print."""
        if not any(pod.dynamic for pod in state.pods.values()):
            return self.cached_simulate(state, keep_dynamic_paths)
        dynamic_result = self.cached_simulate(state, True)
        fixed_result = self.cached_simulate(self.fixed_dynamic_state(state, dynamic_result.dynamic_paths))
        if keep_dynamic_paths:
            fixed_result.dynamic_paths = dynamic_result.dynamic_paths
        return fixed_result

    def cached_simulate(self, state: PlanState, keep_dynamic_paths: bool = False) -> SimulationResult:
        """Returns a cached simulation result for state and keep_dynamic_paths."""
        key = self.simulation_cache_key(state, keep_dynamic_paths)
        if key not in self.simulation_cache:
            self.simulation_cache[key] = self.simulate(state, keep_dynamic_paths)
        return self.copy_simulation_result(self.simulation_cache[key])

    def simulation_cache_key(self, state: PlanState, keep_dynamic_paths: bool) -> tuple:
        """Builds a hashable simulation key from state and keep_dynamic_paths."""
        pods = tuple(sorted((pod_id, tuple(pod.path), pod.dynamic, tuple(sorted(pod.service_area))) for pod_id, pod in state.pods.items()))
        return tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), pods, keep_dynamic_paths

    def copy_simulation_result(self, result: SimulationResult) -> SimulationResult:
        """Copies result so cached objects are not mutated by callers."""
        copy = SimulationResult(result.score, result.speed, result.diversity, result.delivered)
        copy.speed_by_pool = Counter(result.speed_by_pool)
        copy.delivered_by_pool = Counter(result.delivered_by_pool)
        copy.delivery_times = dict(result.delivery_times)
        copy.diversity_by_module = Counter(result.diversity_by_module)
        copy.delivered_by_module = Counter(result.delivered_by_module)
        copy.wait_by_edge = Counter(result.wait_by_edge)
        copy.preventable_wait_by_edge = Counter(result.preventable_wait_by_edge)
        copy.congestion_by_edge = Counter(result.congestion_by_edge)
        copy.dynamic_paths = {pod_id: path[:] for pod_id, path in result.dynamic_paths.items()}
        return copy

    def fixed_dynamic_state(self, state: PlanState, dynamic_paths: dict[int, list[int]]) -> PlanState:
        """Returns a copy of state with dynamic pod paths fixed to dynamic_paths."""
        pods = {}
        for pod_id, pod in state.pods.items():
            path = normalize_month_path(dynamic_paths[pod_id]) if pod.dynamic else pod.path[:]
            pods[pod_id] = PodPlan(pod.id, path, set(pod.service_area), False)
        return PlanState(dict(state.tubes), dict(state.teleports), pods, {pod_id: set(area) for pod_id, area in state.service_areas.items()},
            list(state.actions), list(state.placeholders), set(state.planned_pods),
            {pod_id: set(edges) for pod_id, edges in state.planned_pod_edges.items()}, dict(state.planned_pod_pools), set(state.planned_tubes), state.cost)

    def simulate(self, state: PlanState, keep_dynamic_paths: bool = False) -> SimulationResult:
        """Simulates one month of astronaut movement through state."""
        distances, module_distances = self.distances_to_targets(state)
        graph = tube_graph(state.tubes)
        wanted_edges = self.wanted_edges(distances, graph)
        queues = self.initial_queues(distances)
        result = SimulationResult()
        fixed_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if not pod.dynamic]
        dynamic_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if pod.dynamic]
        service_counts = self.service_counts(state)
        service_graphs = {pod_id: tube_graph({edge: 1 for edge in pod.service_area}) for pod_id, pod in dynamic_pods}
        pod_positions = {pod_id: 0 for pod_id, _ in fixed_pods}
        dynamic_current = {pod_id: -1 for pod_id, _ in dynamic_pods}
        dynamic_paths = {pod_id: [] for pod_id, _ in dynamic_pods}
        module_arrivals = Counter()
        reservations = {}
        for day in range(MONTH_DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, module_arrivals, result)
            demand, priorities = self.prioritized_demand(queues, wanted_edges, distances, module_distances, module_arrivals, reservations)
            if not demand:
                break
            requests = self.pod_requests(fixed_pods, dynamic_pods, pod_positions, dynamic_current, demand, priorities, graph, service_counts,
                service_graphs)
            moves = self.allocate_tube_capacity(requests, state, demand, priorities, result, service_counts)
            self.board_and_launch(queues, distances, wanted_edges, state, moves, pod_positions, dynamic_current, dynamic_paths, result)
            self.settle(day + 1, queues, module_arrivals, result)
        if keep_dynamic_paths:
            result.dynamic_paths = {pod_id: normalize_month_path(path) for pod_id, path in dynamic_paths.items()}
        return result

    def distances_to_targets(self, state: PlanState) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
        """Calculates shortest distances from every building to demanded module types and individual modules."""
        demanded = {kind for pad in self.landing_pads() for kind in pad.demand}
        reverse_edges = {}
        for a, b in state.tubes:
            reverse_edges.setdefault(a, []).append((b, 1))
            reverse_edges.setdefault(b, []).append((a, 1))
        for a, b in state.teleports.items():
            reverse_edges.setdefault(b, []).append((a, 0))
        module_distances = {}
        for module in self.buildings.values():
            if module.kind not in demanded:
                continue
            module_distances[module.id] = {building_id: INF for building_id in self.buildings}
            queue = deque()
            module_distances[module.id][module.id] = 0
            queue.append(module.id)
            while queue:
                building_id = queue.popleft()
                for neighbor_id, cost in reverse_edges.get(building_id, []):
                    distance = module_distances[module.id][building_id] + cost
                    if distance < module_distances[module.id][neighbor_id]:
                        module_distances[module.id][neighbor_id] = distance
                        if cost:
                            queue.append(neighbor_id)
                        else:
                            queue.appendleft(neighbor_id)
        distances = {kind: {building_id: min(module_distances[module.id][building_id] for module in self.buildings.values() if module.kind == kind)
            for building_id in self.buildings} for kind in demanded}
        return distances, module_distances

    def initial_queues(self, distances: dict[int, dict[int, int]]) -> dict[int, list[Passenger]]:
        """Creates starting passenger queues using distances to drop impossible groups."""
        queues = {}
        for pad in self.landing_pads():
            passengers = [Passenger(pad.id, index, kind, pad.id * 1000 + index) for index, kind in enumerate(pad.order) if distances[kind][pad.id] < INF]
            if passengers:
                queues[pad.id] = passengers
        return queues

    def wanted_edges(self, distances: dict[int, dict[int, int]], graph: dict[int, list[int]]) -> dict[tuple[int, int], tuple[DirectedPair, ...]]:
        """Returns every distance-reducing directed tube edge for each building and astronaut type."""
        wanted = {}
        for kind, kind_distances in distances.items():
            for building_id, distance in kind_distances.items():
                options = [neighbor_id for neighbor_id in graph.get(building_id, []) if kind_distances[neighbor_id] < distance]
                wanted[building_id, kind] = tuple((building_id, neighbor_id) for neighbor_id in options)
        return wanted

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
                score = speed + diversity
                result.score += score
                result.speed += speed
                result.diversity += diversity
                result.delivered += 1
                pool = passenger.pad_id, passenger.kind
                result.speed_by_pool[pool] += speed
                result.delivered_by_pool[pool] += 1
                result.diversity_by_module[building_id] += diversity
                result.delivered_by_module[building_id] += 1
                if result.delivered_by_pool[pool] == self.buildings[passenger.pad_id].demand[passenger.kind]:
                    result.delivery_times[pool] = day
                module_arrivals[building_id] += 1
            if remaining:
                queues[building_id] = remaining
            else:
                del queues[building_id]

    def prioritized_demand(self, queues: dict[int, list[Passenger]], wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]],
            distances: dict[int, dict[int, int]], module_distances: dict[int, dict[int, int]], module_arrivals: Counter[int],
            reservations: dict[int, int]) -> tuple[Counter[DirectedPair], dict[DirectedPair, int]]:
        """Counts demand and resolves overlapping current batches into low, normal, or high priorities using diversity."""
        demand = Counter()
        batches = {}
        active = {}
        for building_id in sorted(queues):
            for passenger in sorted(queues[building_id], key=lambda item: item.id):
                active[passenger.id] = building_id, passenger
                for move in wanted_edges[building_id, passenger.kind]:
                    demand[move] += 1
                    batch = batches.setdefault(move, [])
                    if len(batch) < POD_CAPACITY:
                        batch.append(passenger)
        for passenger_id in set(reservations) - set(active):
            del reservations[passenger_id]
        inbound = Counter(module_arrivals)
        counted = set()
        for passenger_id, (building_id, passenger) in active.items():
            modules = self.nearest_modules(building_id, passenger.kind, distances, module_distances)
            if reservations.get(passenger_id) in modules:
                inbound[reservations[passenger_id]] += 1
                counted.add(passenger_id)
            elif len(modules) == 1:
                reservations[passenger_id] = modules[0]
                inbound[modules[0]] += 1
                counted.add(passenger_id)
            else:
                reservations.pop(passenger_id, None)
        priorities = {move: 1 for move in demand}
        for building_id in sorted(queues):
            remaining = {move for move in demand if move[0] == building_id}
            batch_ids = {move: {passenger.id for passenger in batches[move]} for move in remaining}
            while remaining:
                component = {min(remaining)}
                passenger_ids = set(batch_ids[next(iter(component))])
                while True:
                    added = {move for move in remaining - component if passenger_ids & batch_ids[move]}
                    if not added:
                        break
                    component.update(added)
                    passenger_ids.update(*(batch_ids[move] for move in added))
                remaining -= component
                if len(component) == 1:
                    continue
                for passenger_id in passenger_ids:
                    if passenger_id in counted:
                        inbound[reservations[passenger_id]] -= 1
                        counted.remove(passenger_id)
                    reservations.pop(passenger_id, None)
                scored = []
                for move in sorted(component):
                    gain, assignments = self.diversity_assignments(move, batches[move], inbound, distances, module_distances)
                    scored.append((gain, -move[1], move, assignments))
                _, _, preferred, assignments = max(scored)
                for move in component:
                    priorities[move] = 2 if move == preferred else 0
                reservations.update(assignments)
                inbound.update(assignments.values())
                counted.update(assignments)
        return demand, priorities

    def nearest_modules(self, building_id: int, kind: int, distances: dict[int, dict[int, int]],
            module_distances: dict[int, dict[int, int]]) -> list[int]:
        """Returns nearest kind modules to building_id under distances and module_distances."""
        return sorted(module.id for module in self.buildings.values()
            if module.kind == kind and module_distances[module.id][building_id] == distances[kind][building_id])

    def diversity_assignments(self, move: DirectedPair, passengers: list[Passenger], inbound: Counter[int],
            distances: dict[int, dict[int, int]], module_distances: dict[int, dict[int, int]]) -> tuple[int, dict[int, int]]:
        """Estimates diversity gain and best eventual module assignments for passengers taking move."""
        projected = Counter(inbound)
        assignments = {}
        gain = 0
        for passenger in passengers:
            modules = self.nearest_modules(move[1], passenger.kind, distances, module_distances)
            module_id = min(modules, key=lambda item: (-max(0, 50 - projected[item]), item))
            gain += max(0, 50 - projected[module_id])
            projected[module_id] += 1
            assignments[passenger.id] = module_id
        return gain, assignments

    def pod_requests(self, fixed_pods: list[tuple[int, PodPlan]], dynamic_pods: list[tuple[int, PodPlan]], pod_positions: dict[int, int],
            dynamic_current: dict[int, int], demand: Counter[DirectedPair], priorities: dict[DirectedPair, int], graph: dict[int, list[int]],
            service_counts: Counter[Pair], service_graphs: dict[int, dict[int, list[int]]]) -> dict[int, DirectedPair]:
        """Returns daily pod move requests, with dynamic pods choosing requests after fixed pods."""
        requests = {}
        for pod_id, pod in fixed_pods:
            index = pod_positions[pod_id]
            next_index = fixed_next_index(pod.path, index)
            if next_index == index:
                continue
            requests[pod_id] = (pod.path[index], pod.path[next_index])
        for pod_id, pod in dynamic_pods:
            move = self.dynamic_move(pod, dynamic_current[pod_id], demand, priorities, service_counts, graph, service_graphs[pod_id])
            if move != (-1, -1):
                requests[pod_id] = move
        return requests

    def dynamic_move(self, pod: PodPlan, current_id: int, demand: Counter[DirectedPair], priorities: dict[DirectedPair, int],
            service_counts: Counter[Pair], full_graph: dict[int, list[int]], graph: dict[int, list[int]]) -> DirectedPair:
        """Chooses one dynamic pod move according to service_area and demand."""
        area_nodes = set(graph)
        loads = {edge: count for edge, count in demand.items() if route_key(*edge) in pod.service_area}
        active_graph = graph
        if not loads:
            loads = dict(demand)
            active_graph = full_graph
            area_nodes = set(full_graph)
        if not loads:
            return -1, -1
        priority = max(priorities[edge] for edge in loads)
        loads = {edge: count for edge, count in loads.items() if priorities[edge] == priority}
        current = None if current_id == -1 else current_id
        if current is not None and current not in area_nodes:
            active_graph = full_graph
        best_edge = (-1, -1)
        best_key = (INF, INF, INF, INF, INF)
        for source_id, target_id in loads:
            distance = 0 if current is None else graph_distance(active_graph, current, source_id)
            if distance == INF:
                continue
            key = (-min(loads[source_id, target_id], POD_CAPACITY), distance, service_counts[route_key(source_id, target_id)], source_id, target_id)
            if key < best_key:
                best_key = key
                best_edge = source_id, target_id
        if best_edge == (-1, -1):
            return -1, -1
        source_id, target_id = best_edge
        if current is None:
            return source_id, target_id
        if current == source_id:
            return source_id, target_id
        next_id = next_step(active_graph, current, source_id)
        return current, next_id

    def allocate_tube_capacity(self, requests: dict[int, DirectedPair], state: PlanState, demand: Counter[DirectedPair],
            priorities: dict[DirectedPair, int], result: SimulationResult, service_counts: Counter[Pair]) -> dict[int, DirectedPair]:
        """Applies tube capacities, rerouting blocked dynamic pods to available demanded outgoing edges."""
        moves = {}
        by_tube = {}
        remaining_demand = Counter(demand)
        arbitrary_fallback_pods = set()
        for pod_id, move in requests.items():
            by_tube.setdefault(route_key(*move), []).append((pod_id, move))
        for edge, pods in by_tube.items():
            capacity = state.tubes[edge]
            if len(pods) <= capacity:
                for pod_id, move in sorted(pods):
                    moves[pod_id] = move
                    remaining_demand[move] = max(0, remaining_demand[move] - POD_CAPACITY)
                continue
            result.congestion_by_edge[edge] += 1
            selected = self.prioritized_capacity_moves(pods, capacity, state, remaining_demand)
            selected_ids = {pod_id for pod_id, _ in selected}
            id_winners = {pod_id for pod_id, _ in sorted(pods)[:capacity]}
            arbitrary_fallback_pods.update(pod_id for pod_id, _ in pods if pod_id in id_winners - selected_ids and state.pods[pod_id].dynamic)
            for pod_id, move in selected:
                moves[pod_id] = move
                remaining_demand[move] = max(0, remaining_demand[move] - POD_CAPACITY)
        used = Counter(route_key(*move) for move in moves.values())
        for pod_id in sorted(set(requests) - set(moves)):
            if not state.pods[pod_id].dynamic:
                continue
            move = self.capacity_fallback_move(pod_id, requests[pod_id], remaining_demand, priorities, state, used, service_counts,
                pod_id in arbitrary_fallback_pods)
            if move != (-1, -1):
                moves[pod_id] = move
                used[route_key(*move)] += 1
                remaining_demand[move] = max(0, remaining_demand[move] - POD_CAPACITY)
        return moves

    def prioritized_capacity_moves(self, pods: list[tuple[int, DirectedPair]], capacity: int, state: PlanState,
            remaining_demand: Counter[DirectedPair]) -> list[tuple[int, DirectedPair]]:
        """Chooses capacity winners, giving dynamic pods with deliverable passengers priority."""
        selected = []
        test_demand = Counter(remaining_demand)
        fixed = sorted((pod_id, move) for pod_id, move in pods if not state.pods[pod_id].dynamic)
        for pod_id, move in fixed[:capacity]:
            selected.append((pod_id, move))
            test_demand[move] = max(0, test_demand[move] - POD_CAPACITY)
        waiting = [(pod_id, move) for pod_id, move in pods if state.pods[pod_id].dynamic]
        while len(selected) < capacity and waiting:
            best = max(waiting, key=lambda item: (min(POD_CAPACITY, test_demand[item[1]]), -item[0]))
            selected.append(best)
            test_demand[best[1]] = max(0, test_demand[best[1]] - POD_CAPACITY)
            waiting.remove(best)
        return selected

    def capacity_fallback_move(self, pod_id: int, blocked_move: DirectedPair, demand: Counter[DirectedPair], priorities: dict[DirectedPair, int],
            state: PlanState, used: Counter[Pair], service_counts: Counter[Pair], allow_arbitrary: bool) -> DirectedPair:
        """Chooses an available demanded or arbitrary outgoing edge from the blocked move source."""
        source_id = blocked_move[0]
        candidates = []
        for move, count in demand.items():
            edge = route_key(*move)
            if count and move[0] == source_id and move != blocked_move and edge in state.tubes and used[edge] < state.tubes[edge]:
                candidates.append((-priorities[move], -min(count, POD_CAPACITY), service_counts[edge], move))
        in_service = [candidate for candidate in candidates if route_key(*candidate[3]) in state.pods[pod_id].service_area]
        if in_service:
            candidates = in_service
        if candidates:
            return min(candidates)[3]
        return self.arbitrary_capacity_move(pod_id, source_id, state, used) if allow_arbitrary else (-1, -1)

    def arbitrary_capacity_move(self, pod_id: int, source_id: int, state: PlanState, used: Counter[Pair]) -> DirectedPair:
        """Chooses the smallest available outgoing edge from pod_id service area, then the full map."""
        for target_id in sorted({node for edge in state.pods[pod_id].service_area if source_id in edge for node in edge if node != source_id}):
            edge = route_key(source_id, target_id)
            if used[edge] < state.tubes[edge]:
                return source_id, target_id
        for target_id in sorted({node for edge in state.tubes if source_id in edge for node in edge if node != source_id}):
            edge = route_key(source_id, target_id)
            if used[edge] < state.tubes[edge]:
                return source_id, target_id
        return -1, -1

    def board_and_launch(self, queues: dict[int, list[Passenger]], distances: dict[int, dict[int, int]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], state: PlanState, moves: dict[int, DirectedPair],
            pod_positions: dict[int, int], dynamic_current: dict[int, int], dynamic_paths: dict[int, list[int]], result: SimulationResult):
        """Boards passengers into moves, launches pods, and updates wait counters."""
        by_start = {}
        for pod_id, (source_id, target_id) in moves.items():
            by_start.setdefault(source_id, []).append((pod_id, target_id))
        for candidates in by_start.values():
            candidates.sort()
        seats = Counter({pod_id: POD_CAPACITY for pod_id in moves})
        onboard = {}
        wait_today = Counter()
        for building_id in sorted(list(queues)):
            candidates = by_start.get(building_id, [])
            remaining = []
            for passenger in sorted(queues[building_id], key=lambda item: item.id):
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
                    for wanted in wanted_edges[building_id, passenger.kind]:
                        wait_today[route_key(*wanted)] += 1
            if remaining:
                queues[building_id] = remaining
            else:
                del queues[building_id]
        used_capacity = Counter(route_key(*move) for move in moves.values())
        for edge, count in wait_today.items():
            result.wait_by_edge[edge] += count
            result.preventable_wait_by_edge[edge] += min(count, (state.tubes[edge] - used_capacity[edge]) * POD_CAPACITY)
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
            exact_hops: bool = False, via_nodes: tuple[int, ...] = ()) -> list[int]:
        """Searches state paths from start_id to targets up to hop_limit, optionally with exact_hops and a required via_nodes visit."""
        target_set = set(targets)
        via_set = set(via_nodes)
        edge_graph = self.build_candidate_edge_graph(state.tubes)
        start_key = start_id, 0, not via_set or start_id in via_set
        costs = {start_key: 0}
        parents = {}
        queue = deque([start_key])
        while queue:
            building_id, hops, visited_via = queue.popleft()
            if hops >= hop_limit:
                continue
            for neighbor_id, edge_cost in edge_graph.get(building_id, []):
                next_hops = hops + 1
                cost = costs[building_id, hops, visited_via] + edge_cost
                key = neighbor_id, next_hops, visited_via or neighbor_id in via_set
                if cost >= costs.get(key, INF):
                    continue
                costs[key] = cost
                parents[key] = building_id, hops, visited_via
                queue.append(key)
        best_key = None
        for target_id in target_set:
            for hops in range(1, hop_limit + 1):
                key = target_id, hops, True
                if exact_hops and hops != hop_limit or key not in costs:
                    continue
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
        print(f"month {self.month + 1}", file=sys.stderr)
        print(f"resources {self.resources}", file=sys.stderr)
        for building in sorted(self.buildings.values(), key=lambda item: item.id):
            if building.kind == 0:
                demand = ",".join(map(str, building.order)) if building.order else "none"
                print(f"landing {building.id} {building.x} {building.y} {demand}", file=sys.stderr)
            else:
                print(f"module {building.id} {building.kind} {building.x} {building.y}", file=sys.stderr)
        for a, b in sorted(self.tubes):
            print(f"tube {a} {b} {self.tubes[a, b]}", file=sys.stderr)
        for a, b in sorted(self.teleports.items()):
            print(f"teleport {a} {b}", file=sys.stderr)
        for pod_id in sorted(self.pods):
            path_text = ", ".join(map(str, self.pods[pod_id].path))
            area_text = ", ".join(f"{a}-{b}" for a, b in sorted(self.service_areas[pod_id]))
            print(f"pod id={pod_id}, service={{{area_text}}}, path=[{path_text}]", file=sys.stderr)

    def score_debug(self, label: str, result: SimulationResult, cost: int) -> str:
        """Formats score diagnostics for label, result, and cost."""
        demand = sum(sum(pad.demand.values()) for pad in self.landing_pads())
        stats = self.status_debug(result)
        if label == "before":
            landings = []
            for pad in self.landing_pads():
                demand_text = ", ".join(f"{kind}x{count}" for kind, count in sorted(pad.demand.items()))
                landings.append(f"landing {pad.id}: {demand_text}")
            return "\n".join((*landings, "", stats))
        return f"After: speed {result.speed}, diversity {result.diversity}, delivered {result.delivered}/{demand}, " \
            f"score: {result.score}, resources: {self.resources - cost}\n{stats}"

    def status_debug(self, result: SimulationResult) -> str:
        """Formats all speed-pool and diversity-module statistics in result."""
        return f"{self.pool_debug(result)}\n{self.diversity_debug(result)}"

    def pool_debug(self, result: SimulationResult) -> str:
        """Formats speed and delivery diagnostics for each astronaut pool."""
        lines = []
        for pool in self.speed_pools():
            pad_id, kind = pool
            max_speed = self.buildings[pad_id].demand[kind] * 50
            delivery_time = result.delivery_times[pool] if pool in result.delivery_times else "-"
            lines.append(f"speed pool {pool}: {result.speed_by_pool[pool]}/{max_speed}, delivery {delivery_time}")
        return "\n".join(lines)

    def diversity_debug(self, result: SimulationResult) -> str:
        """Formats diversity diagnostics for each demanded module."""
        lines = []
        for building in sorted(self.buildings.values(), key=lambda item: item.id):
            if building.kind <= 0:
                continue
            max_diversity = self.max_diversity(building.kind)
            if not max_diversity:
                continue
            line = f"diversity pool {building.id}: {result.diversity_by_module[building.id]}/{max_diversity}, "
            lines.append(f"{line}delivered {result.delivered_by_module[building.id]}")
        return "\n".join(lines)

    def max_diversity(self, kind: int) -> int:
        """Returns the maximum diversity score for module kind."""
        demand = sum(pad.demand[kind] for pad in self.landing_pads())
        return sum(max(0, 50 - index) for index in range(demand))


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


def parse_auto_area(text: str) -> set[Pair]:
    """Parses AUTO service area text into route keys."""
    area = set()
    for edge_text in text.removeprefix("AUTO(").removesuffix(")").replace(",", " ").split():
        a, b = map(int, edge_text.split("-"))
        area.add(route_key(a, b))
    return area


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
