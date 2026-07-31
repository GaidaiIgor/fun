from collections import Counter, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import inf, isqrt
from operator import attrgetter
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
OVERRIDE_MONTH = 15
OVERRIDE_COMMAND = "POD 3 3 5 3 6 3 5 3 4 1 4 1 4 3 5 3 6 3 5 3 5 3;POD 4 4 1 4 1 4 1 4 1 4 1 4 1 4 1 4 1 4 1 4 1 4;POD 5 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2"
# "POD 3 AUTO;POD 4 AUTO;POD 5 AUTO"
FULL_DEBUG = False
_G = {}
BY_ID = attrgetter("id")
Pair = tuple[int, int]
DirectedPair = tuple[int, int]
Pool = tuple[int, int]
PoolOwner = Pool | int
PathKey = tuple[int, ...]
LoadKey = tuple[Pool, int, PathKey]
def debug(text: str):
    if FULL_DEBUG:
        print(text, file=sys.stderr)
@dataclass(slots=True)
class Building:
    id: int
    kind: int
    x: int
    y: int
    demand: Counter[int] = field(default_factory=Counter)
    order: list[int] = field(default_factory=list)
@dataclass(slots=True)
class Passenger:
    pad_id: int
    kind: int
    id: int
@dataclass(slots=True, frozen=True)
class PathDemand:
    pool: Pool
    destination: int
    nodes: PathKey
    cap: int
    ambiguous: bool = False
    h: int = field(init=False, compare=False)
    def __post_init__(self):
        object.__setattr__(self, "h", hash((self.pool, self.destination, self.nodes, self.cap, self.ambiguous)))
    def __hash__(self) -> int:
        return self.h
@dataclass(slots=True)
class PodPlan:
    path: list[int] = field(default_factory=list)
    dynamic: bool = False
@dataclass(slots=True)
class Bundle:
    pool: PoolOwner
    tubes: tuple[Pair, ...] = ()
    teleport: Pair = (-1, -1)
    pod_specs: tuple[int, ...] = ()
    upgrades: tuple[Pair, ...] = ()
    label: str = "empty"
    path_edges: tuple[Pair, ...] = ()
    destination: int = -1
    path_length: int = 0
    path: tuple[int, ...] = ()
    debug_id: str = ""
    debug_chosen: str = ""
    @property
    def fingerprint(self) -> tuple:
        return self.tubes, self.teleport, self.pod_specs, self.upgrades
@dataclass(slots=True)
class PlanState:
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, PodPlan]
    actions: list[str] = field(default_factory=list)
    pod_slots: list[tuple[int, int]] = field(default_factory=list)
    ops: set[int] = field(default_factory=set)
    routes: dict[int, set[Pair]] = field(default_factory=dict)
    pairs: dict[int, tuple[Pool, int]] = field(default_factory=dict)
    new_tubes: set[Pair] = field(default_factory=set)
    cost: int = 0
@dataclass(slots=True)
class SimulationResult:
    score: int = 0
    speed_by_pool: Counter[Pool] = field(default_factory=Counter)
    delivered_by_pool: Counter[Pool] = field(default_factory=Counter)
    delivery_times: dict[Pool, int] = field(default_factory=dict)
    diversity_by_module: Counter[int] = field(default_factory=Counter)
    delivered_by_module: Counter[int] = field(default_factory=Counter)
    delivered_by_pool_module: Counter[tuple[Pool, int]] = field(default_factory=Counter)
    congestion_by_edge: Counter[Pair] = field(default_factory=Counter)
    dynamic_paths: dict[int, list[int]] = field(default_factory=dict)
    table: list[list[str]] = field(default_factory=list)
    reserved: str = "-"
@dataclass(slots=True)
class Candidate:
    bundle: Bundle
    pair: PoolOwner
    global_gain: int
    global_cost: int
    @property
    def efficiency(self) -> float:
        return self.global_gain / self.global_cost if self.global_cost > 0 else inf
class Planner:
    buildings: dict[int, Building]
    resources: int
    month: int
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, PodPlan]
    simulation_cache: dict[tuple, SimulationResult]
    def __init__(self):
        self.buildings = {}
        self.resources = 0
        self.month = 0
        self.tubes = {}
        self.teleports = {}
        self.pods = {}
        self.simulation_cache = {}
        self.fs_cache = {}
    def play(self):
        while True:
            try:
                self.read_month()
            except EOFError:
                return
            actions = self.choose_actions()
            print(";".join(actions) if actions else "WAIT")
            self.month += 1
    def read_month(self):
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
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            pod_id = values[0]
            self.pods[pod_id] = PodPlan(values[2:])
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            if values[0] == 0:
                self.buildings[values[1]] = Building(values[1], 0, values[2], values[3], Counter(values[5:]), values[5:])
            else:
                self.buildings[values[1]] = Building(values[1], values[0], values[2], values[3])
        self.print_debug_input()
    def choose_actions(self) -> list[str]:
        self.simulation_cache = {}
        self.fs_cache = {}
        self.cap_cache = {}
        _G.clear()
        if self.month + 1 == OVERRIDE_MONTH:
            return self.override_actions()
        selected = []
        current_state = self.replay_bundle_sequence(selected)
        current_result = self.score_state(current_state)
        before_score = current_result.score
        if FULL_DEBUG:
            debug("\n" + self.score_debug("before", current_result, current_state.cost))
        while True:
            best = self.best_candidate(selected, current_state, current_result, before_score)
            if best is None:
                break
            selected.append(best.bundle)
            current_state = self.replay_bundle_sequence(selected)
            current_result = self.score_state(current_state)
            if FULL_DEBUG:
                self.selected_debug(best, current_state, current_result, before_score)
                debug(f"\nIteration {len(selected) + 1}\n" + self.status_debug(current_result))
        final_state = self.replay_bundle_sequence(selected)
        final_result = self.score_state(final_state, True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths)
        if FULL_DEBUG:
            debug("\n" + self.table_debug(final_result, final_state))
            debug("\n" + self.score_debug("after", final_result, final_state.cost))
        action_order = {"TUBE": 0, "TELEPORT": 0, "UPGRADE": 1, "DESTROY": 2, "POD": 3}
        return sorted((action for action in final_state.actions if action), key=lambda action: action_order[action.split()[0]])
    def override_actions(self) -> list[str]:
        current_state = self.replay_bundle_sequence([])
        current_result = self.score_state(current_state)
        if FULL_DEBUG:
            debug("\n" + self.score_debug("override", current_result, current_state.cost))
        final_state = self.override_state(OVERRIDE_COMMAND)
        final_result = self.score_state(final_state, True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths)
        if FULL_DEBUG:
            debug("\n" + self.table_debug(final_result, final_state))
            debug(f"override month {self.month + 1}: {OVERRIDE_COMMAND}")
            debug("\n" + self.score_debug("after", final_result, final_state.cost))
        return [action for action in final_state.actions if action]
    def override_state(self, command: str) -> PlanState:
        state = self.replay_bundle_sequence([])
        if command.strip() == "WAIT":
            return state
        for action in (item.strip() for item in command.split(";")):
            if action:
                self.apply_override_action(state, action)
        return state
    def apply_override_action(self, state: PlanState, action: str):
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
            state.cost -= POD_REFUND
            state.actions.append(action)
        elif command == "POD":
            self.apply_override_pod(state, action)
        elif command != "WAIT":
            raise ValueError(f"unknown override action {command}")
    def apply_override_pod(self, state: PlanState, action: str):
        _, pod_text, route_text = action.split(maxsplit=2)
        pod_id = int(pod_text)
        if pod_id in state.pods:
            del state.pods[pod_id]
            state.cost -= POD_REFUND
            state.actions.append(f"DESTROY {pod_id}")
        if route_text.startswith("AUTO"):
            state.cost += POD_COST
            state.pods[pod_id] = PodPlan([], True)
            state.pod_slots.append((len(state.actions), pod_id))
            state.actions.append("")
            return
        path = [int(item) for item in route_text.split()]
        state.cost += POD_COST
        state.pods[pod_id] = PodPlan(path)
        state.actions.append(action)
    def best_candidate(self, selected: list[Bundle], current_state: PlanState, current_result: SimulationResult,
            before_score: int) -> Candidate:
        pools = []
        distances, _ = self.distances_to_targets(current_state)
        for pool in self.speed_pools():
            missing = self.buildings[pool[0]].demand[pool[1]] * 50 - current_result.speed_by_pool[pool]
            if missing > 0:
                modules = sorted(building.id for building in self.buildings.values()
                    if building.kind == pool[1] and self.speed_destination_eligible(pool, building.id, current_result))
                pools.append((missing, (0, pool[0], pool[1]), pool, [(module_id, pool, [module_id]) for module_id in modules]))
        for module in sorted(self.buildings.values(), key=lambda item: item.id):
            if module.kind <= 0:
                continue
            missing = self.perfect_diversity(module.kind) - current_result.diversity_by_module[module.id]
            if missing <= 0:
                continue
            groups = [pool for pool in self.speed_pools()
                if pool[1] == module.kind and self.diversity_group_eligible(pool, module.id, current_state, current_result, distances)]
            if groups:
                pools.append((missing, (1, module.id), module.id, [(group, group, [module.id]) for group in groups]))
        pools.sort(key=lambda item: (-item[0], item[1]))
        for _, _, owner, pairs in pools:
            debug(f"Considering {owner}:")
            best = None
            for pair, group, module_ids in pairs:
                debug(f"  Considering {pair}:")
                candidate = self.next_candidate(owner, pair, group, selected, current_state, current_result, before_score,
                    self.generate_bundles(owner, group, module_ids, selected, current_state, current_result, before_score))
                if candidate and (best is None or (candidate.efficiency, candidate.global_gain, -candidate.global_cost) >
                        (best.efficiency, best.global_gain, -best.global_cost)):
                    best = candidate
            if best:
                return best
        return None
    def speed_destination_eligible(self, pool: Pool, module_id: int, result: SimulationResult) -> bool:
        modules = [building for building in self.buildings.values() if building.kind == pool[1]]
        before = sum(result.diversity_by_module[module.id] for module in modules)
        populations = Counter({module.id: result.delivered_by_module[module.id] for module in modules})
        for (delivered_pool, delivered_module_id), count in result.delivered_by_pool_module.items():
            if delivered_pool == pool:
                populations[delivered_module_id] -= count
        populations[module_id] += self.buildings[pool[0]].demand[pool[1]]
        after = sum(sum(max(0, 50 - index) for index in range(populations[module.id])) for module in modules)
        return after >= before
    def diversity_group_eligible(self, group: Pool, module_id: int, state: PlanState, result: SimulationResult,
            distances: dict[int, dict[int, int]]) -> bool:
        if result.delivered_by_pool_module[group, module_id]:
            return False
        current_length = distances[group[1]][group[0]]
        if current_length <= MAX_TUBE_HOPS and self.cheapest_hop_path(group[0], [module_id], current_length, state):
            return True
        hop_limit = min(max(0, current_length - 1), MAX_TUBE_HOPS)
        return self.speed_destination_eligible(group, module_id, result) and \
            bool(hop_limit and self.cheapest_path_with_hop_limit(group[0], [module_id], hop_limit, state))
    def next_candidate(self, owner: PoolOwner, pair: PoolOwner, group: Pool, selected: list[Bundle], current_state: PlanState,
            current_result: SimulationResult, before_score: int, bundles: list[Bundle]) -> Candidate:
        best = None
        seen = set()
        current_pool_score = current_result.speed_by_pool[owner] if isinstance(owner, tuple) else current_result.diversity_by_module[owner]
        plans = []
        for bundle in bundles:
            if bundle.fingerprint == Bundle(owner).fingerprint and not bundle.path_edges:
                continue
            if bundle.path_edges and current_result.delivery_times.get(group, INF) == bundle.path_length:
                continue
            fingerprint = bundle.path, bundle.debug_id, bundle.fingerprint
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                state = self.replay_bundle_sequence([*selected, bundle])
            except ValueError:
                continue
            if current_state.tubes == state.tubes and current_state.teleports == state.teleports and \
                    current_state.routes == state.routes and current_state.pairs == state.pairs:
                continue
            action_text = self.state_delta_text(current_state, state) if FULL_DEBUG else ""
            plans.append((bundle, state, action_text, state.cost))
        path = ()
        for bundle, state, action_text, cost in plans:
            if bundle.path != path:
                path = bundle.path
                path_text = ", ".join(map(str, path))
                debug(f"    Considering path=[{path_text}]:")
            prefix = "-> " if bundle.debug_chosen else ""
            text = f"      {prefix}{bundle.debug_id}: action={action_text}, "
            if state.cost > self.resources:
                debug(f"{text}local gain=-, global gain=-, cost={cost}, efficiency=-")
                continue
            result = self.score_state(state)
            pool_score = result.speed_by_pool[owner] if isinstance(owner, tuple) else result.diversity_by_module[owner]
            local_gain = pool_score - current_pool_score
            global_gain = result.score - before_score
            checkpoint_delta = result.score - current_result.score
            efficiency = global_gain / cost if cost > 0 else inf
            debug(f"{text}local gain={local_gain}, global gain={global_gain}({checkpoint_delta:+d}), cost={cost}, "
                f"efficiency={efficiency:.3f}")
            if global_gain > 0 and result.score > current_result.score:
                candidate = Candidate(bundle, pair, global_gain, cost)
                if best is None or (candidate.efficiency, candidate.global_gain, -candidate.global_cost) > \
                        (best.efficiency, best.global_gain, -best.global_cost):
                    best = candidate
        return best
    def generate_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], selected: list[Bundle], state: PlanState,
            current_result: SimulationResult, before_score: int) -> list[Bundle]:
        bases = []
        pad_id = group[0]
        current_length = INF
        allow_shorter = True
        if isinstance(owner, int):
            distances, _ = self.distances_to_targets(state)
            current_length = distances[group[1]][pad_id]
            allow_shorter = self.speed_destination_eligible(group, module_ids[0], current_result)
        existing_path = self.shortest_existing_tube_path(pad_id, module_ids, state.tubes)
        if not existing_path:
            connections = self.connection_bundles(owner, group, module_ids, state)
            bases.extend(connections)
            route_length = connections[0].path_length if connections else 1
        else:
            route_length = len(existing_path) - 1
            bases.extend(self.path_bundles(owner, "existing", existing_path, state))
        bases.extend(self.shortest_route_bundles(owner, group, module_ids, route_length, state))
        if isinstance(owner, int) and current_length <= MAX_TUBE_HOPS:
            equal_path = self.cheapest_hop_path(pad_id, module_ids, current_length, state)
            if equal_path and tuple(equal_path) not in {bundle.path for bundle in bases}:
                bases.extend(self.path_bundles(owner, f"equal-{current_length}", equal_path, state))
        if isinstance(owner, int):
            bases = [bundle for bundle in bases
                if bundle.path_length == current_length or allow_shorter and bundle.path_length < current_length]
        bundles = []
        connections = [base for base in bases if base.label in ("connect", "connect-pod")]
        if connections:
            bundles.extend(self.connection_bundle_stack(owner, group, connections, selected, state, before_score))
        connection_ids = {id(base) for base in connections}
        seen = set()
        for base in bases:
            if id(base) in connection_ids:
                continue
            key = base.path, base.tubes, base.pod_specs
            if key in seen:
                continue
            seen.add(key)
            bundles.extend(self.path_bundle_stack(owner, group, base, selected, state, before_score))
        teleports = self.teleport_bundles(owner, group, module_ids, state)
        if isinstance(owner, int):
            teleports = [bundle for bundle in teleports
                if bundle.path_length == current_length or allow_shorter and bundle.path_length < current_length]
        bundles.extend(teleports)
        return bundles
    def connection_bundle_stack(self, owner: PoolOwner, group: Pool, bases: list[Bundle], selected: list[Bundle], state: PlanState,
            before_score: int) -> list[Bundle]:
        bundles = []
        options = []
        for base in bases:
            base.debug_id = "0c" if base.label == "connect-pod" else "0"
            metrics = self.bundle_metrics(base, selected, before_score)
            bundles.append(base)
            if metrics[3].cost <= self.resources:
                options.append((metrics[2], metrics[0], -metrics[1], base, metrics[3]))
        if not options:
            return bundles
        efficiency, _, _, parent, parent_state = max(options, key=lambda item: item[:3])
        parent.debug_chosen = parent.debug_id
        bundles.extend(self.throughput_bundles(owner, group, parent, parent_state, efficiency, selected, state, before_score, 1))
        return bundles
    def path_bundle_stack(self, owner: PoolOwner, group: Pool, base: Bundle, selected: list[Bundle], state: PlanState,
            before_score: int) -> list[Bundle]:
        if base.tubes:
            base.debug_id = "0c" if base.label == "connect-pod" else "0"
            bundles = [base]
            base_metrics = self.bundle_metrics(base, selected, before_score)
            if base_metrics[3].cost > self.resources:
                return bundles
            base.debug_chosen = base.debug_id
            bundles.extend(self.throughput_bundles(owner, group, base, base_metrics[3], base_metrics[2], selected, state, before_score, 1))
            return bundles
        parent = Bundle(owner, label=base.label, path_edges=base.path_edges, destination=base.destination,
            path_length=base.path_length, path=base.path)
        _, _, efficiency, parent_state = self.bundle_metrics(parent, selected, before_score)
        pod_seed = base if base.fingerprint != Bundle(owner).fingerprint else None
        return self.throughput_bundles(owner, group, parent, parent_state, efficiency, selected, state, before_score, 1, pod_seed)
    def throughput_bundles(self, owner: PoolOwner, group: Pool, parent: Bundle, parent_state: PlanState, parent_efficiency: float,
            selected: list[Bundle], state: PlanState, before_score: int, round_number: int, pod_seed: Bundle = None) -> list[Bundle]:
        bundles = []
        while parent_state.cost <= self.resources:
            result = self.cached_simulate(parent_state)
            if result.delivery_times.get(group, INF) == parent.path_length:
                break
            options = []
            if pod_seed:
                projected = self.replay_bundle_sequence([*selected, pod_seed])
            else:
                projected = self.replay_bundle_on_state(parent_state, Bundle(owner, pod_specs=(0,),
                    path_edges=parent.path_edges, destination=parent.destination, path=parent.path))
            pod_bundle = self.projection_bundle(owner, parent, state, projected, f"{parent.label}-pod")
            pod_bundle.debug_id = f"{round_number}p"
            pod_metrics = self.bundle_metrics(pod_bundle, selected, before_score)
            options.append((pod_bundle, pod_metrics))
            upgrade_edge = self.best_counter_edge(parent.path_edges, result.congestion_by_edge)
            if upgrade_edge != (-1, -1):
                projected = self.replay_bundle_on_state(parent_state,
                    Bundle(owner, upgrades=(upgrade_edge,), path_edges=parent.path_edges))
                upgrade_bundle = self.projection_bundle(owner, parent, state, projected, f"{parent.label}-upgrade")
                upgrade_bundle.debug_id = f"{round_number}u"
                options.append((upgrade_bundle, self.bundle_metrics(upgrade_bundle, selected, before_score)))
            combined_affordable = False
            if pod_metrics[3].cost <= self.resources:
                edge = self.best_counter_edge(parent.path_edges, self.cached_simulate(pod_metrics[3]).congestion_by_edge)
                if edge != (-1, -1):
                    upgrade_cost = tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * (parent_state.tubes[edge] + 1)
                    if parent_state.cost + upgrade_cost <= self.resources:
                        projected = self.replay_bundle_on_state(pod_metrics[3], Bundle(owner, upgrades=(edge,), path_edges=parent.path_edges))
                        combined = self.projection_bundle(owner, parent, state, projected, f"{parent.label}-pod-upgrade")
                        combined.debug_id = f"{round_number}b"
                        combined_metrics = self.bundle_metrics(combined, selected, before_score)
                        options.append((combined, combined_metrics))
                        combined_affordable = combined_metrics[3].cost <= self.resources
            bundles.extend(bundle for bundle, _ in options)
            affordable = [(metrics[2], metrics[0], -metrics[1], bundle, metrics[3]) for bundle, metrics in options
                if metrics[3].cost <= self.resources]
            if not affordable:
                break
            efficiency, _, _, next_parent, next_state = max(affordable, key=lambda item: item[:3])
            if efficiency <= parent_efficiency or not combined_affordable:
                break
            next_parent.debug_chosen = next_parent.debug_id
            parent, parent_state, parent_efficiency = next_parent, next_state, efficiency
            round_number += 1
            pod_seed = None
        return bundles
    def bundle_metrics(self, bundle: Bundle, selected: list[Bundle], before_score: int) -> tuple[int, int, float, PlanState]:
        projected = self.replay_bundle_sequence([*selected, bundle])
        cost = projected.cost
        if projected.cost > self.resources:
            return 0, cost, -inf, projected
        result = self.score_state(projected)
        gain = result.score - before_score
        return gain, cost, gain / cost if cost > 0 else inf if gain > 0 else 0, projected
    def projection_bundle(self, owner: PoolOwner, template: Bundle, before: PlanState, after: PlanState, label: str) -> Bundle:
        tubes = tuple(edge for edge in template.path_edges if edge in after.tubes and edge not in before.tubes)
        upgrades = tuple(edge for edge in sorted(after.tubes) for _ in range(after.tubes[edge] - before.tubes.get(edge, 1)))
        specs = [pod_id for pod_id in sorted(set(before.pods) & set(after.pods))
            if after.pods[pod_id].dynamic and (not before.pods[pod_id].dynamic or
                before.routes[pod_id] != after.routes[pod_id] or before.pairs[pod_id] != after.pairs[pod_id])]
        specs.extend(0 for _ in set(after.pods) - set(before.pods))
        return Bundle(owner, tubes=tubes, pod_specs=tuple(specs), upgrades=upgrades, label=label, path_edges=template.path_edges,
            destination=template.destination, path_length=template.path_length, path=template.path)
    def connection_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> list[Bundle]:
        path = self.cheapest_connecting_path(group[0], module_ids, state)
        bundles = self.path_bundles(owner, "connect", path, state)
        if bundles and state.pods and any(not pod_id for pod_id in bundles[0].pod_specs):
            connected = self.pod_connection_bundle(owner, group, module_ids, state)
            if connected:
                bundles.append(connected)
        return bundles
    def pod_connection_bundle(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> Bundle:
        best = None
        places = self.pod_locations(state)
        for pod_id in state.pods:
            locations = places[pod_id]
            routes = []
            path = self.cheapest_path_with_hop_limit(group[0], module_ids, MAX_TUBE_HOPS, state, via_nodes=tuple(locations))
            if path:
                routes.append((tuple(route_key(a, b) for a, b in zip(path, path[1:])), path[-1]))
            for module_id in module_ids:
                base_path = self.cheapest_connecting_path(group[0], [module_id], state)
                base_edges = tuple(route_key(a, b) for a, b in zip(base_path, base_path[1:]))
                for junction_id in base_path:
                    remaining_hops = MAX_TUBE_HOPS - len(base_edges)
                    connector = [junction_id] if junction_id in locations else \
                        self.cheapest_path_with_hop_limit(junction_id, list(locations), remaining_hops, state)
                    if not connector:
                        continue
                    edges = tuple(dict.fromkeys((*base_edges, *(route_key(a, b) for a, b in zip(connector, connector[1:])))))
                    if len(edges) <= MAX_TUBE_HOPS and self.can_add_tubes([edge for edge in edges if edge not in state.tubes], state.tubes):
                        routes.append((edges, module_id))
            for path_edges, module_id in routes:
                pair = group, module_id
                specs = (pod_id,) if pod_id not in state.ops or state.pairs[pod_id] == pair else ()
                tubes = tuple(edge for edge in path_edges if edge not in state.tubes)
                cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes)
                cost += REROUTE_COST if specs and pod_id not in state.ops else 0
                projected_tubes = dict(state.tubes)
                projected_tubes.update((edge, 1) for edge in path_edges)
                route = self.shortest_existing_tube_path(group[0], [module_id], projected_tubes)
                bundle = Bundle(owner, tubes=tubes, pod_specs=specs, label="connect-pod", path_edges=path_edges,
                    destination=module_id, path_length=len(route) - 1, path=tuple(route))
                source = self.buildings[group[0]]
                target = self.buildings[module_id]
                distance = (source.x - target.x) * (source.x - target.x) + (source.y - target.y) * (source.y - target.y)
                order = cost, distance, len(route), path_edges, pod_id
                if best is None or order < best[0]:
                    best = order, bundle
        return best[1] if best else None
    def shortest_route_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], route_length: int,
            state: PlanState) -> list[Bundle]:
        bundles = []
        for hop_count in range(route_length - 1, 0, -1):
            for module_id in module_ids:
                path = self.cheapest_hop_path(group[0], [module_id], hop_count, state)
                bundles.extend(self.path_bundles(owner, f"short-{hop_count}", path, state))
        return bundles
    def teleport_bundles(self, owner: PoolOwner, group: Pool, modules: list[int], state: PlanState) -> list[Bundle]:
        pad_id = group[0]
        used = self.teleport_used_buildings(state.teleports)
        if pad_id in used:
            return []
        bundles = [Bundle(owner, teleport=(pad_id, module_id), label=f"teleport-{module_id}", destination=module_id,
            path=(pad_id, module_id)) for module_id in sorted(modules, key=lambda item: tube_cost(self.buildings[pad_id], self.buildings[item]))
            if module_id not in used]
        for index, bundle in enumerate(bundles, 1):
            bundle.debug_id = "0t" if index == 1 else f"0t{index}"
        return bundles
    def path_bundles(self, owner: PoolOwner, label: str, path: list[int], state: PlanState) -> list[Bundle]:
        if not path:
            return []
        tubes = tuple(unique_new_tubes(path, state.tubes))
        path_edges = tuple(route_key(a, b) for a, b in zip(path, path[1:]))
        group = owner if isinstance(owner, tuple) else (path[0], self.buildings[owner].kind)
        pair = group, path[-1]
        pod_id = self.closest_pod(path[0], path_edges, state) if tubes else -1
        specs = (pod_id,) if pod_id >= 0 and (not pod_id or pod_id not in state.ops or state.pairs[pod_id] == pair) else ()
        return [Bundle(owner, tubes=tubes, pod_specs=specs, label=label, path_edges=path_edges,
            destination=path[-1], path_length=len(path) - 1, path=tuple(path))]
    def closest_pod(self, origin_id: int, path_edges: tuple[Pair, ...], state: PlanState) -> int:
        if not state.pods:
            return 0
        tubes = dict(state.tubes)
        tubes.update((edge, 1) for edge in path_edges)
        graph = tube_graph(tubes)
        options = []
        places = self.pod_locations(state)
        for pod_id in state.pods:
            nodes = places[pod_id]
            dist = min(graph_distance(graph, origin_id, node) for node in nodes)
            options.append((dist, pod_id not in state.ops, pod_id))
        best = min(options)
        return best[2] if best[0] < INF else 0
    def pod_locations(self, state: PlanState) -> dict[int, set[int]]:
        dynamic_paths = self.score_state(state, True).dynamic_paths if state.ops else {}
        return {pod_id: set(dynamic_paths.get(pod_id) or pod.path) for pod_id, pod in state.pods.items()}
    def best_counter_edge(self, path_edges: tuple[Pair, ...], counts: Counter[Pair]) -> Pair:
        candidates = [(counts[edge], edge) for edge in path_edges if counts[edge]]
        return max(candidates, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1] if candidates else (-1, -1)
    def replay_bundle_on_state(self, state: PlanState, bundle: Bundle) -> PlanState:
        pods = {pod_id: PodPlan(pod.path[:], pod.dynamic) for pod_id, pod in state.pods.items()}
        copied = PlanState(dict(state.tubes), dict(state.teleports), pods, list(state.actions), list(state.pod_slots),
            set(state.ops), {pod_id: set(edges) for pod_id, edges in state.routes.items()},
            dict(state.pairs), set(state.new_tubes), state.cost)
        self.apply_bundle(copied, bundle)
        return copied
    def replay_bundle_sequence(self, selected: list[Bundle]) -> PlanState:
        pods = {pod_id: PodPlan(pod.path[:]) for pod_id, pod in self.pods.items()}
        state = PlanState(dict(self.tubes), dict(self.teleports), pods)
        applied = []
        for bundle in selected:
            self.apply_bundle(state, bundle)
            applied.append(bundle)
            self.prune_uncommitted_infrastructure(state, applied)
        return state
    def prune_uncommitted_infrastructure(self, state: PlanState, selected: list[Bundle]):
        active = set()
        freed = set()
        routes = {}
        for bundle in selected:
            if bundle.path_edges or bundle.teleport != (-1, -1):
                edges = set(bundle.path_edges)
                group = bundle.pool if isinstance(bundle.pool, tuple) else (bundle.path[0], self.buildings[bundle.pool].kind)
                pair = group, bundle.destination
                if bundle.teleport != (-1, -1):
                    freed.update(routes.get(pair, ()))
                routes[pair] = edges
        for edges in routes.values():
            active.update(edges)
        for index, action in enumerate(state.actions):
            parts = action.split()
            if parts and parts[0] == "UPGRADE":
                edge = route_key(int(parts[1]), int(parts[2]))
                if edge not in active:
                    state.cost -= tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * state.tubes[edge]
                    state.tubes[edge] -= 1
                    state.actions[index] = ""
        for pod_id in list(state.ops):
            if state.routes[pod_id] != routes.get(state.pairs[pod_id]):
                self.remove_planned_pod(state, pod_id)
        for edge in sorted(state.new_tubes - active):
            remaining = dict(state.tubes)
            del remaining[edge]
            if edge in freed or graph_distance(tube_graph(remaining), *edge) < INF:
                self.remove_planned_tube(state, edge)
    def remove_planned_pod(self, state: PlanState, pod_id: int):
        if pod_id in self.pods:
            state.cost -= REROUTE_COST
            state.pods[pod_id] = PodPlan(self.pods[pod_id].path[:])
        else:
            state.cost -= POD_COST
            del state.pods[pod_id]
        state.ops.remove(pod_id)
        del state.routes[pod_id]
        del state.pairs[pod_id]
        state.pod_slots = [(index, placeholder_id) for index, placeholder_id in state.pod_slots if placeholder_id != pod_id]
        for index, action in enumerate(state.actions):
            parts = action.split()
            if parts and parts[0] == "DESTROY" and int(parts[1]) == pod_id:
                state.actions[index] = ""
    def remove_planned_tube(self, state: PlanState, edge: Pair):
        capacity = state.tubes[edge]
        state.cost -= tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * capacity * (capacity + 1) // 2
        del state.tubes[edge]
        state.new_tubes.remove(edge)
        for index, action in enumerate(state.actions):
            parts = action.split()
            if parts and parts[0] in ("TUBE", "UPGRADE") and route_key(int(parts[1]), int(parts[2])) == edge:
                state.actions[index] = ""
    def apply_bundle(self, state: PlanState, bundle: Bundle):
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
            state.new_tubes.add(key)
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
        for pod_id in bundle.pod_specs:
            group = bundle.pool if isinstance(bundle.pool, tuple) else (bundle.path[0], self.buildings[bundle.pool].kind)
            pair = group, bundle.destination
            if pod_id and pod_id in state.ops:
                if state.pairs[pod_id] != pair:
                    raise ValueError("pod owned by another route")
                if state.routes[pod_id] != set(bundle.path_edges):
                    self.remove_planned_pod(state, pod_id)
                    pod_id = pod_id if pod_id in self.pods else 0
            if pod_id:
                if pod_id not in state.pods:
                    raise ValueError("missing reroute pod")
                del state.pods[pod_id]
                if pod_id not in state.ops:
                    state.cost += REROUTE_COST
                    state.actions.append(f"DESTROY {pod_id}")
                    state.ops.add(pod_id)
                    state.pod_slots.append((len(state.actions), pod_id))
                    state.actions.append("")
            else:
                pod_id = self.next_pod_id(state.pods)
                state.cost += POD_COST
                state.ops.add(pod_id)
                state.pod_slots.append((len(state.actions), pod_id))
                state.actions.append("")
            state.routes[pod_id] = set(bundle.path_edges)
            state.pairs[pod_id] = pair
            state.pods[pod_id] = PodPlan([], True)
    def fill_dynamic_actions(self, state: PlanState, dynamic_paths: dict[int, list[int]]):
        for index, pod_id in state.pod_slots:
            path = dynamic_paths[pod_id]
            assert len(path) >= 2, f"dynamic pod {pod_id} produced an empty route"
            state.actions[index] = "POD {} {}".format(pod_id, " ".join(map(str, path)))
    def score_state(self, state: PlanState, keep_dynamic_paths: bool = False) -> SimulationResult:
        if not any(pod.dynamic for pod in state.pods.values()):
            return self.cached_simulate(state)
        dynamic_result = self.cached_simulate(state)
        paths = dynamic_result.dynamic_paths
        fixed_result = self.cached_simulate(self.fixed_dynamic_state(state, paths))
        if keep_dynamic_paths:
            fixed_result.dynamic_paths = paths
            fixed_result.table = dynamic_result.table
            fixed_result.reserved = dynamic_result.reserved
        return fixed_result
    def cached_simulate(self, state: PlanState) -> SimulationResult:
        keep_dynamic_paths = any(pod.dynamic for pod in state.pods.values())
        pods = tuple(sorted((pod_id, tuple(pod.path), pod.dynamic) for pod_id, pod in state.pods.items()))
        pairs = tuple(sorted(state.pairs.items())) if keep_dynamic_paths else ()
        key = tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), pods, pairs
        if key not in self.simulation_cache:
            self.simulation_cache[key] = self.simulate(state, keep_dynamic_paths)
        return self.simulation_cache[key]
    def fixed_dynamic_state(self, state: PlanState, dynamic_paths: dict[int, list[int]]) -> PlanState:
        pods = {pod_id: PodPlan(dynamic_paths[pod_id][:] if pod.dynamic else pod.path[:]) for pod_id, pod in state.pods.items()}
        return PlanState(state.tubes, state.teleports, pods)
    def simulate(self, state: PlanState, keep_dynamic_paths: bool = False) -> SimulationResult:
        network_key = tuple(sorted(state.tubes)), tuple(sorted(state.teleports.items()))
        if network_key not in self.fs_cache:
            distances, module_distances = self.distances_to_targets(state)
            graph = tube_graph(state.tubes)
            self.fs_cache[network_key] = distances, graph, tube_components(graph), self.wanted_edges(distances, graph), \
                self.path_demands(state, distances, module_distances), self.initial_queues(distances)
        distances, graph, components, wanted_edges, path_demands, initial = self.fs_cache[network_key]
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
        result = SimulationResult()
        self.capacities = tuple(sorted(state.tubes.items()))
        fixed_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if not pod.dynamic]
        dynamic_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if pod.dynamic]
        key = tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), tuple((pod_id, tuple(pod.path)) for pod_id, pod in fixed_pods)
        if dynamic_pods and key not in self.fs_cache:
            self.fs_cache[key] = self.fixed_assignment_schedule(state, distances, wanted_edges, path_demands, fixed_pods)
        fixed_schedule, summary = self.fs_cache[key] if dynamic_pods else ([], Counter())
        if keep_dynamic_paths:
            result.reserved = ", ".join("({})x{}".format("-".join(map(str, path)), count)
                for path, count in summary.items()) or "-"
        pod_positions = {pod_id: 0 for pod_id, _ in fixed_pods}
        dynamic_current = {pod_id: -1 for pod_id, _ in dynamic_pods}
        dynamic_pending = {pod_id: (-1, -1) for pod_id, _ in dynamic_pods}
        dynamic_paths = {pod_id: [] for pod_id, _ in dynamic_pods}
        arrivals = Counter()
        for day in range(MONTH_DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, arrivals, result)
            for passengers in queues.values():
                passengers.sort(key=BY_ID)
            self.supply = Counter((passenger.pad_id, passenger.kind, node_id) for node_id, passengers in queues.items() for passenger in passengers)
            active = []
            for path in path_demands:
                active.extend(self.active_path_demands(path, self.supply, wanted_edges, result))
            if not active:
                break
            if not dynamic_pods:
                requests = self.path_pod_requests(fixed_pods, [], pod_positions, {}, {}, {}, state, graph, result)
                moves = self.allocate_tube_capacity(requests, state, Counter(), result)
                self.board_and_launch(queues, distances, state, moves, pod_positions, {}, {})
                self.settle(day + 1, queues, arrivals, result)
                continue
            fixed_assignments, fixed_reservations, _ = self.fixed_load_assignments(fixed_pods, pod_positions, active, queues, wanted_edges,
                result)
            self.fixed_edges = tuple(sorted(tuple(sorted({route_key(a, b) for path in paths
                for a, b in zip(path.nodes, path.nodes[1:])})) for paths in fixed_assignments.values()))
            demand = self.edge_demand(queues, wanted_edges)
            reserved_loads = Counter()
            for _, reservations in fixed_schedule[day:]:
                reserved_loads.update(reservations)
            assignments, preferences, locations, priorities = self.dispatch_dynamic_paths(day, active, dynamic_pods, dynamic_pending,
                dynamic_current, fixed_reservations, reserved_loads, result, state, graph, components, queues, wanted_edges, demand)
            requests = self.path_pod_requests(fixed_pods, dynamic_pods, pod_positions, dynamic_current, dynamic_pending, assignments,
                state, graph, result, True)
            moves = self.allocate_tube_capacity(requests, state, demand, result, False)
            assignments, requests, moves = self.resolve_dispatch_congestion(assignments, preferences, requests, moves,
                fixed_pods, dynamic_pods, pod_positions, dynamic_current, locations, dynamic_pending, graph, result, state, demand,
                fixed_reservations, priorities)
            if keep_dynamic_paths:
                loads = ", ".join("({})x{}{}".format("-".join(map(str, path.nodes)), self.path_remaining(path, result),
                    "N" if priorities[path] == 0 else "L") for path in active)
                cells = []
                for pod_id, pod in sorted(state.pods.items()):
                    paths = fixed_assignments.get(pod_id, set()) if not pod.dynamic else {assignments[pod_id]} if pod_id in assignments else set()
                    path_text = "/".join("-".join(map(str, path.nodes)) for path in paths) or "-"
                    location = pod.path[pod_positions[pod_id]] if not pod.dynamic else dynamic_current[pod_id]
                    location = requests[pod_id][0] if location == -1 and pod_id in requests else location
                    cells.append("{} ({})".format(path_text, location if location != -1 else "-"))
                result.table.append([str(day + 1), loads, *cells])
            for pod_id, _ in dynamic_pods:
                if dynamic_pending[pod_id] != (-1, -1) or pod_id not in requests:
                    continue
                dynamic_pending[pod_id] = requests[pod_id]
                if dynamic_current[pod_id] == -1:
                    dynamic_current[pod_id] = requests[pod_id][0]
                    dynamic_paths[pod_id].append(requests[pod_id][0])
                dynamic_paths[pod_id].append(requests[pod_id][1])
            self.board_and_launch(queues, distances, state, moves, pod_positions, dynamic_current, dynamic_pending)
            self.settle(day + 1, queues, arrivals, result)
        if keep_dynamic_paths:
            result.dynamic_paths = {pod_id: normalize_month_path(path) for pod_id, path in dynamic_paths.items()}
        return result
    def fixed_assignment_schedule(self, state: PlanState, distances: dict[int, dict[int, int]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], path_demands: list[PathDemand],
            fixed_pods: list[tuple[int, PodPlan]]) -> tuple[list[tuple[dict[int, set[PathDemand]], Counter[LoadKey]]], Counter[PathKey]]:
        if not fixed_pods:
            return [({}, Counter()) for _ in range(MONTH_DAYS)], Counter()
        queues = self.initial_queues(distances)
        result = SimulationResult()
        positions = {pod_id: 0 for pod_id, _ in fixed_pods}
        arrivals = Counter()
        schedule = []
        summary = Counter()
        for day in range(MONTH_DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, arrivals, result)
            for passengers in queues.values():
                passengers.sort(key=BY_ID)
            supply = Counter((passenger.pad_id, passenger.kind, node_id) for node_id, passengers in queues.items() for passenger in passengers)
            active = []
            for path in path_demands:
                active.extend(self.active_path_demands(path, supply, wanted_edges, result))
            assignments, _, claimed = self.fixed_load_assignments(fixed_pods, positions, active, queues, wanted_edges, result)
            reservations = Counter()
            for path, count in claimed.items():
                reservations[path.pool, path.destination, path.nodes[:2]] += count
                summary[path.nodes] += count
            schedule.append((assignments, reservations))
            if not active:
                schedule.extend(({}, Counter()) for _ in range(day + 1, MONTH_DAYS))
                break
            requests = {}
            for pod_id, pod in fixed_pods:
                next_index = fixed_next_index(pod.path, positions[pod_id])
                if next_index != positions[pod_id]:
                    requests[pod_id] = pod.path[positions[pod_id]], pod.path[next_index]
            moves = self.allocate_tube_capacity(requests, state, self.edge_demand(queues, wanted_edges), result, False)
            self.board_and_launch(queues, distances, state, moves, positions, {}, {})
            self.settle(day + 1, queues, arrivals, result)
        return schedule, summary
    def fixed_load_assignments(self, fixed_pods: list[tuple[int, PodPlan]], pod_positions: dict[int, int],
            active: list[PathDemand], queues: dict[int, list[Passenger]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], result: SimulationResult) -> tuple:
        moves = {}
        for pod_id, pod in fixed_pods:
            next_index = fixed_next_index(pod.path, pod_positions[pod_id])
            if next_index != pod_positions[pod_id]:
                moves[pod_id] = pod.path[pod_positions[pod_id]], pod.path[next_index]
        paths = {}
        for path in active:
            paths.setdefault((path.pool, path.nodes[:2]), []).append(path)
        assignments = {}
        claimed = Counter()
        reservations = Counter()
        seats = {pod_id: POD_CAPACITY for pod_id in moves}
        by_source = {}
        for pod_id, move in moves.items():
            by_source.setdefault(move[0], []).append((pod_id, move[1]))
        for source_id, passengers in queues.items():
            candidates = sorted(by_source.get(source_id, []))
            if not candidates:
                continue
            for passenger in passengers:
                pod_id = next((candidate_id for candidate_id, target_id in candidates
                    if seats[candidate_id] and (source_id, target_id) in wanted_edges[source_id, passenger.kind]), 0)
                if not pod_id:
                    continue
                seats[pod_id] -= 1
                pool = passenger.pad_id, passenger.kind
                reservations[pool, source_id] += 1
                edge = source_id, moves[pod_id][1]
                options = [path for path in paths.get((pool, edge), []) if claimed[path] < self.path_remaining(path, result)]
                if options:
                    path = min(options, key=lambda item: (not item.ambiguous, item.destination, item.nodes))
                    claimed[path] += 1
                    assignments.setdefault(pod_id, set()).add(path)
        return assignments, reservations, claimed
    def active_path_demands(self, path: PathDemand, supply: Counter[tuple[int, int, int]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], result: SimulationResult) -> list[PathDemand]:
        remaining = self.path_remaining(path, result)
        demands = []
        delivered = result.delivered_by_pool_module[path.pool, path.destination]
        for index in range(len(path.nodes) - 2, -1, -1):
            edge = path.nodes[index], path.nodes[index + 1]
            count = supply[path.pool[0], path.pool[1], edge[0]] if edge in wanted_edges[edge[0], path.pool[1]] else 0
            count = min(count, remaining)
            if count:
                demands.append(PathDemand(path.pool, path.destination, path.nodes[index:], delivered + count, path.ambiguous and index == 0))
                remaining -= count
        return demands
    def path_demands(self, state: PlanState, distances: dict[int, dict[int, int]],
            module_distances: dict[int, dict[int, int]]) -> list[PathDemand]:
        groups = []
        inbound = Counter()
        for pool in self.speed_pools():
            options = [module.id for module in self.buildings.values() if module.kind == pool[1]
                and module_distances[module.id][pool[0]] == distances[pool[1]][pool[0]]]
            groups.append((pool, sorted(options)))
            if len(options) == 1:
                inbound[options[0]] += self.buildings[pool[0]].demand[pool[1]]
        demands = []
        for pool, options in groups:
            caps = Counter()
            count = self.buildings[pool[0]].demand[pool[1]]
            if len(options) == 1:
                caps[options[0]] = count
            else:
                for _ in range(count):
                    module_id = min(options, key=lambda item: (inbound[item], item))
                    caps[module_id] += 1
                    inbound[module_id] += 1
            for module_id, cap in caps.items():
                path = self.concrete_path(pool[0], module_id, state)
                for run in self.tube_path_runs(path, state.tubes):
                    demands.append(PathDemand(pool, module_id, run, cap, len(options) > 1))
        return demands
    def concrete_path(self, start_id: int, finish_id: int, state: PlanState) -> PathKey:
        edges = {}
        for a, b in state.tubes:
            edges.setdefault(a, []).append((b, 1))
            edges.setdefault(b, []).append((a, 1))
        for a, b in state.teleports.items():
            edges.setdefault(a, []).append((b, 0))
        queue = [(0, (start_id,), start_id)]
        best = {}
        while queue:
            cost, path, node = heappop(queue)
            if node in best and best[node] <= (cost, path):
                continue
            best[node] = cost, path
            if node == finish_id:
                return path
            for target_id, edge_cost in sorted(edges.get(node, [])):
                heappush(queue, (cost + edge_cost, (*path, target_id), target_id))
        return ()
    def tube_path_runs(self, path: PathKey, tubes: dict[Pair, int]) -> list[PathKey]:
        runs = []
        run = []
        for a, b in zip(path, path[1:]):
            if route_key(a, b) in tubes:
                if not run:
                    run.append(a)
                run.append(b)
            elif run:
                runs.append(tuple(run))
                run = []
        if run:
            runs.append(tuple(run))
        return runs
    def path_remaining(self, path: PathDemand, result: SimulationResult) -> int:
        remaining_pool = self.buildings[path.pool[0]].demand[path.pool[1]] - result.delivered_by_pool[path.pool]
        return min(max(0, path.cap - result.delivered_by_pool_module[path.pool, path.destination]), remaining_pool)
    def dispatch_dynamic_paths(self, day: int, active: list[PathDemand], dynamic_pods: list[tuple[int, PodPlan]],
            pending: dict[int, DirectedPair], current: dict[int, int], fixed_reservations: Counter[tuple[Pool, int]],
            reserved_loads: Counter[LoadKey], result: SimulationResult, state: PlanState, graph: dict[int, list[int]],
            components: dict[int, int], queues: dict[int, list[Passenger]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], demand: Counter[DirectedPair]) -> tuple:
        pod_ids = [pod_id for pod_id, _ in dynamic_pods]
        locations = dict(current)
        preferred_targets = {}
        for path in active:
            if path.ambiguous:
                preferred_targets.setdefault((path.pool, path.nodes[0]), set()).add(path.nodes[1])
        availability = {path: max(0, self.path_remaining(path, result) -
            reserved_loads[path.pool, path.destination, path.nodes[:2]]) for path in active}
        load_sizes = {path: min(POD_CAPACITY, self.path_remaining(path, result)) for path in active}
        normal = {path for path in active if availability[path] >= POD_CAPACITY}
        dispatchable = active
        priorities = {path: int(path not in normal) for path in active}
        initial_capacity = {path: self.path_capacity_excess(path, {}, state) for path in dispatchable}
        allowed = {path for path in dispatchable if self.path_batch_allowed(path, preferred_targets, queues, wanted_edges)}
        preferences = {}
        for pod_id in pod_ids:
            candidates = dispatchable if day == 0 else [path for path in dispatchable
                if graph_distance(graph, locations[pod_id], path.nodes[0]) < INF]
            candidates = [path for path in candidates if path in allowed]
            preferences[pod_id] = sorted(candidates, key=lambda path: (priorities[path], -min(POD_CAPACITY, availability[path]),
                *self.path_assignment_key(path, pod_id, initial_capacity[path], locations, result, graph, queues, wanted_edges,
                    demand)))
        assignments = {}
        counts = Counter()
        if day == 0:
            uncovered = {components[path.nodes[0]] for path in dispatchable}
            for pod_id in pod_ids:
                options = [path for path in preferences[pod_id] if not counts[path] and components[path.nodes[0]] in uncovered
                    and not initial_capacity[path]
                    and not self.dispatch_supply_exceeded(assignments | {pod_id: path}, locations, fixed_reservations, result)]
                if options:
                    assignments[pod_id] = options[0]
                    counts[options[0]] += 1
                    uncovered.remove(components[options[0].nodes[0]])
                if not uncovered:
                    break
        remaining = [pod_id for pod_id in pod_ids if pod_id not in assignments]
        unavailable = set()
        surplus = 0
        while remaining:
            capacity_blocked = set() if surplus else {path for path in dispatchable
                if self.path_capacity_excess(path, assignments, state)}
            supply_left = {(path.pool, path.nodes[0]): self.supply[path.pool[0], path.pool[1], path.nodes[0]] -
                fixed_reservations[path.pool, path.nodes[0]] for path in dispatchable}
            for assigned_id, path in assignments.items():
                if locations[assigned_id] in (-1, path.nodes[0]):
                    supply_left[path.pool, path.nodes[0]] -= load_sizes[path]
            proposals = {}
            for pod_id in remaining:
                options = [path for path in preferences[pod_id] if path not in unavailable and path not in capacity_blocked
                    and (surplus > 1 or locations[pod_id] not in (-1, path.nodes[0]) or
                        load_sizes[path] <= supply_left[path.pool, path.nodes[0]])]
                if options:
                    pair = state.pairs.get(pod_id)
                    paired = [path for path in options if (path.pool, path.destination) == pair] if surplus and day == 0 else []
                    if paired:
                        options = paired
                    priority = min(priorities[path] for path in options)
                    options = [path for path in options if priorities[path] == priority]
                    level = min(counts[path] for path in options)
                    options = [path for path in options if counts[path] == level]
                    if level:
                        length = max(len(path.nodes) for path in options)
                        options = [path for path in options if len(path.nodes) == length]
                if options:
                    proposals.setdefault(options[0], []).append(pod_id)
            if not proposals:
                if surplus == 2:
                    break
                surplus += 1
                unavailable.clear()
                continue
            for path, pod_options in proposals.items():
                first_edge = path.nodes[0], path.nodes[1]
                pod_id = min(pod_options, key=lambda item: (graph_distance(graph, locations[item], path.nodes[0]),
                    pending[item] != first_edge[::-1], pending[item] == (-1, -1) or pending[item][1] != path.nodes[0], item))
                if not surplus and self.path_capacity_excess(path, assignments, state) or surplus < 2 and \
                        self.dispatch_supply_exceeded(assignments | {pod_id: path}, locations, fixed_reservations, result):
                    unavailable.add(path)
                    continue
                assignments[pod_id] = path
                counts[path] += 1
                remaining.remove(pod_id)
        return assignments, preferences, locations, priorities
    def path_batch_allowed(self, path: PathDemand, preferred_targets: dict[tuple[Pool, int], set[int]],
            queues: dict[int, list[Passenger]], wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]]) -> bool:
        source_id, target_id = path.nodes[:2]
        preferred = nonpreferred = boarded = 0
        for passenger in queues.get(source_id, []):
            if (source_id, target_id) not in wanted_edges[source_id, passenger.kind]:
                continue
            targets = preferred_targets.get(((passenger.pad_id, passenger.kind), source_id))
            if targets:
                if target_id in targets:
                    preferred += 1
                else:
                    nonpreferred += 1
            boarded += 1
            if boarded == POD_CAPACITY:
                break
        return preferred >= nonpreferred
    def dispatch_supply_exceeded(self, assignments: dict[int, PathDemand], current: dict[int, int],
            fixed_reservations: Counter[tuple[Pool, int]], result: SimulationResult) -> bool:
        reserved = {}
        for pod_id, path in assignments.items():
            if current[pod_id] in (-1, path.nodes[0]):
                key = path.pool, path.nodes[0]
                reserved[key] = reserved.get(key, 0) + min(POD_CAPACITY, self.path_remaining(path, result))
        return any(count + fixed_reservations[pool, source] > self.supply[*pool, source]
            for (pool, source), count in reserved.items())
    def resolve_dispatch_congestion(self, assignments: dict[int, PathDemand], preferences: dict[int, list[PathDemand]],
            requests: dict[int, DirectedPair], moves: dict[int, DirectedPair], fixed_pods: list[tuple[int, PodPlan]],
            dynamic_pods: list[tuple[int, PodPlan]], pod_positions: dict[int, int], current: dict[int, int],
            locations: dict[int, int], pending: dict[int, DirectedPair], graph: dict[int, list[int]], result: SimulationResult,
            state: PlanState, demand: Counter[DirectedPair], fixed_reservations: Counter[tuple[Pool, int]],
            priorities: dict[PathDemand, int]) -> tuple:
        def reposition(values: dict[int, PathDemand]) -> tuple:
            distances = [[graph_distance(graph, locations[pod_id], path.nodes[0]) for pod_id, path in values.items()
                if locations[pod_id] != -1 and priorities[path] == priority] for priority in (0, 1)]
            return tuple(value for items in distances for value in (max(items, default=0), sum(items)))
        def evaluate(values: dict[int, PathDemand]) -> tuple[dict[int, DirectedPair], dict[int, DirectedPair]]:
            key = tuple(sorted(values.items()))
            if key not in evaluations:
                trial_requests = self.path_pod_requests(fixed_pods, dynamic_pods, pod_positions, current, pending, values,
                    state, graph, result)
                evaluations[key] = trial_requests, self.allocate_tube_capacity(trial_requests, state, demand, result, False)
            return evaluations[key]
        def balanced(values: dict[int, PathDemand]) -> bool:
            counts = Counter(values.values())
            key = frozenset(counts.items())
            if key not in balance_cache:
                invalid = any(counts[a] > counts[b] + 1 and not self.path_capacity_excess(b, values, state)
                    for a in loads for b in loads if priorities[a] == priorities[b]) or \
                    any(len(a.nodes) < len(b.nodes) and counts[a] > counts[b] and
                        not self.path_capacity_excess(b, values, state)
                        for a in loads for b in loads if priorities[a] == priorities[b]) or \
                    any(priorities[a] < priorities[b] and counts[b] and
                        not self.path_capacity_excess(a, values, state) for a in loads for b in loads)
                balance_cache[key] = not invalid
            return balance_cache[key]
        evaluations = {tuple(sorted(assignments.items())): (requests, moves)}
        balance_cache = {}
        dispatchable = set(preferences)
        loads = {path for paths in preferences.values() for path in paths}
        blocked = sum(pod_id in dispatchable and pod_id not in moves for pod_id in requests)
        original = dict(assignments)
        while blocked:
            best = None
            trials = [(pod_id, path, -1) for pod_id, paths in preferences.items() for path in paths
                if assignments.get(pod_id) != path]
            trials.extend((pod_id, assignments[other_id], other_id) for pod_id in assignments for other_id in assignments
                if pod_id < other_id and assignments[other_id] in preferences[pod_id] and assignments[pod_id] in preferences[other_id])
            for pod_id, path, other_id in trials:
                previous = assignments.get(pod_id)
                trial = dict(assignments)
                trial[pod_id] = path
                if other_id >= 0:
                    trial[other_id] = previous
                if self.dispatch_supply_exceeded(trial, locations, fixed_reservations, result):
                    continue
                other_assignments = dict(trial)
                del other_assignments[pod_id]
                if self.path_capacity_excess(path, other_assignments, state):
                    continue
                if other_id >= 0:
                    del other_assignments[other_id]
                    if self.path_capacity_excess(previous, other_assignments | {pod_id: path}, state):
                        continue
                if not balanced(trial):
                    continue
                trial_requests, trial_moves = evaluate(trial)
                trial_blocked = sum(dynamic_id in dispatchable and dynamic_id not in trial_moves for dynamic_id in trial_requests)
                loaded = sum(min(POD_CAPACITY, demand[move]) for dynamic_id, move in trial_moves.items() if dynamic_id in dispatchable)
                changes = sum(trial.get(dynamic_id) != original.get(dynamic_id) for dynamic_id in dispatchable)
                rank = sum(preferences[dynamic_id].index(trial[dynamic_id]) for dynamic_id in dispatchable if dynamic_id in trial)
                candidate = trial_blocked, reposition(trial), changes, rank, -loaded, pod_id, other_id, trial, trial_requests, trial_moves
                if best is None or candidate[:7] < best[:7]:
                    best = candidate
            if best is None or best[0] >= blocked:
                break
            blocked, _, _, _, _, _, _, assignments, requests, moves = best
        while True:
            target = reposition(assignments)
            trials = []
            for pod_id in assignments:
                for other_id in assignments:
                    if pod_id >= other_id or assignments[other_id] not in preferences[pod_id] or \
                            assignments[pod_id] not in preferences[other_id]:
                        continue
                    trial = dict(assignments)
                    trial[pod_id], trial[other_id] = trial[other_id], trial[pod_id]
                    distance = reposition(trial)
                    if distance < target:
                        trials.append((*distance, pod_id, other_id, trial))
            changed = False
            loaded = sum(min(POD_CAPACITY, demand[move]) for pod_id, move in moves.items() if pod_id in dispatchable)
            for *_, trial in sorted(trials):
                trial_requests, trial_moves = evaluate(trial)
                trial_blocked = sum(pod_id in dispatchable and pod_id not in trial_moves for pod_id in trial_requests)
                trial_loaded = sum(min(POD_CAPACITY, demand[move]) for pod_id, move in trial_moves.items() if pod_id in dispatchable)
                if trial_blocked <= blocked and trial_loaded >= loaded:
                    assignments, requests, moves, blocked, changed = trial, trial_requests, trial_moves, trial_blocked, True
                    break
            if not changed:
                break
        return assignments, requests, moves
    def path_assignment_key(self, path: PathDemand, pod_id: int, capacity_blocked: bool, current: dict[int, int],
            result: SimulationResult, graph: dict[int, list[int]], queues: dict[int, list[Passenger]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], demand: Counter[DirectedPair]) -> tuple:
        distance = 0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])
        blocked = capacity_blocked or \
            self.path_blocks_loaded_pod(path, pod_id, current, graph, queues, wanted_edges, demand, result)
        return blocked, not path.ambiguous, self.path_remaining(path, result) < POD_CAPACITY, distance, len(path.nodes) - 1, \
            result.delivered_by_module[path.destination]
    def path_blocks_loaded_pod(self, path: PathDemand, pod_id: int, current: dict[int, int], graph: dict[int, list[int]],
            queues: dict[int, list[Passenger]], wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]],
            demand: Counter[DirectedPair], result: SimulationResult) -> bool:
        source_id = current[pod_id]
        if source_id == -1:
            return False
        if source_id not in path.nodes:
            target_id = next_step(graph, source_id, path.nodes[0])
        else:
            position = path.nodes.index(source_id)
            target_id = path.nodes[position - 1] if position == len(path.nodes) - 1 else path.nodes[position + 1]
            if position and not self.path_segment_ready(path, source_id, target_id, queues, wanted_edges, result):
                target_id = path.nodes[position - 1]
        return not demand[source_id, target_id] and demand[target_id, source_id] and \
            any(other_id != pod_id and node_id == target_id for other_id, node_id in current.items())
    def path_capacity_excess(self, candidate: PathDemand, assignments: dict[int, PathDemand], state: PlanState) -> bool:
        paths = sorted([*(path.nodes for path in assignments.values()), candidate.nodes])
        fixed = self.fixed_edges
        key = self.capacities, tuple(paths), fixed
        if key not in self.cap_cache:
            used = Counter(edge for edges in fixed for edge in edges)
            edges = [[route_key(a, b) for a, b in zip(path, path[1:])] for path in paths]
            required = {edge for path_edges in edges for edge in path_edges}
            slots = {edge: [(edge, index) for index in range(max(0, state.tubes[edge] - used[edge]))] for edge in required}
            owners = {}
            def place(worker: int, seen: set[tuple[Pair, int]]) -> bool:
                for edge in edges[worker]:
                    for slot in slots[edge]:
                        if slot in seen:
                            continue
                        seen.add(slot)
                        if slot not in owners or place(owners[slot], seen):
                            owners[slot] = worker
                            return True
                return False
            self.cap_cache[key] = any(not place(worker, set()) for worker in range(len(paths)))
        return self.cap_cache[key]
    def path_pod_requests(self, fixed_pods: list[tuple[int, PodPlan]], dynamic_pods: list[tuple[int, PodPlan]],
            pod_positions: dict[int, int], current: dict[int, int], pending: dict[int, DirectedPair], assignments: dict[int, PathDemand],
            state: PlanState, graph: dict[int, list[int]], result: SimulationResult, track: bool = False) -> dict[int, DirectedPair]:
        def place(pod_id: int, seen: set[tuple[Pair, int]]) -> bool:
            for occupied in (False, True):
                for move in options[pod_id]:
                    for slot in slots[route_key(*move)]:
                        owner = owners.get(slot)
                        if (owner is not None) != occupied or slot in seen:
                            continue
                        seen.add(slot)
                        if owner is None or place(owner, seen):
                            owners[slot] = pod_id
                            selected[pod_id] = move
                            return True
            return False
        requests = {}
        for pod_id, pod in fixed_pods:
            index = pod_positions[pod_id]
            next_index = fixed_next_index(pod.path, index)
            if next_index != index:
                requests[pod_id] = pod.path[index], pod.path[next_index]
        used = Counter(route_key(*move) for move in requests.values())
        options = {}
        reverse = {}
        claims = Counter()
        for pod_id, _ in dynamic_pods:
            if pending[pod_id] != (-1, -1):
                options[pod_id] = [pending[pod_id]]
                reverse[pod_id] = set()
                if pod_id in assignments:
                    path = assignments[pod_id].nodes
                    if pending[pod_id] in zip(path, path[1:]):
                        claims[assignments[pod_id].pool, *pending[pod_id]] += 1
                continue
            if pod_id not in assignments:
                continue
            demand = assignments[pod_id]
            path = demand.nodes
            if current[pod_id] == -1:
                key = demand.pool, path[0], path[1]
                waiting = min(self.supply[demand.pool[0], demand.pool[1], path[0]], self.path_remaining(demand, result))
                moves = [(path[index], path[index - 1]) for index in range(1, len(path))]
                reverse[pod_id] = set(moves)
                if claims[key] * POD_CAPACITY < waiting:
                    moves.insert(0, (path[0], path[1]))
                    claims[key] += 1
                options[pod_id] = list(dict.fromkeys(moves))
                continue
            if current[pod_id] not in path:
                options[pod_id] = [(current[pod_id], next_step(graph, current[pod_id], path[0]))]
                reverse[pod_id] = set(options[pod_id])
                continue
            position = path.index(current[pod_id])
            upstream = (current[pod_id], path[position - 1]) if position else None
            downstream = (current[pod_id], path[position + 1]) if position < len(path) - 1 else None
            loaded = False
            if downstream:
                waiting = min(self.supply[demand.pool[0], demand.pool[1], current[pod_id]], self.path_remaining(demand, result))
                covered = sum(min(POD_CAPACITY, self.path_remaining(other, result)) for other_id, other in assignments.items()
                    if other_id != pod_id and other.pool == demand.pool and other.destination == demand.destination
                    and other.nodes == path[position:])
                key = demand.pool, *downstream
                loaded = claims[key] * POD_CAPACITY < max(0, waiting - covered)
                claims[key] += loaded
            options[pod_id] = [move for move in ((downstream, upstream) if loaded else (upstream, downstream)) if move]
            reverse[pod_id] = {upstream} if upstream else set()
        required = {route_key(*move) for moves in options.values() for move in moves}
        slots = {edge: [(edge, index) for index in range(max(0, state.tubes[edge] - used[edge]))] for edge in required}
        if track:
            preferred = Counter(route_key(*move) for move in requests.values())
            preferred.update(route_key(*moves[0]) for moves in options.values())
            for edge, count in preferred.items():
                if count > state.tubes[edge]:
                    result.congestion_by_edge[edge] += 1
        owners = {}
        selected = {}
        for pod_id in sorted(options):
            place(pod_id, set())
        pods = sorted(selected)
        for index, pod_id in enumerate(pods):
            for other_id in pods[index + 1:]:
                if route_key(*options[pod_id][0]) != route_key(*options[other_id][0]):
                    continue
                occupied = Counter(used)
                occupied.update(route_key(*move) for worker_id, move in selected.items() if worker_id not in (pod_id, other_id))
                choices = []
                for rank, move in enumerate(options[pod_id]):
                    for other_rank, other_move in enumerate(options[other_id]):
                        trial = occupied + Counter((route_key(*move), route_key(*other_move)))
                        if all(count <= state.tubes[edge] for edge, count in trial.items()):
                            upstream = int(move in reverse[pod_id]) + int(other_move in reverse[other_id])
                            choices.append(((rank + other_rank, -upstream, rank, other_rank), move, other_move))
                if choices:
                    _, selected[pod_id], selected[other_id] = min(choices)
        requests.update(selected)
        for pod_id, moves in options.items():
            requests.setdefault(pod_id, moves[0])
        return requests
    def path_segment_ready(self, path: PathDemand, source_id: int, target_id: int, queues: dict[int, list[Passenger]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], result: SimulationResult) -> bool:
        edge = source_id, target_id
        eligible = sum(passenger.pad_id == path.pool[0] and passenger.kind == path.pool[1] and
            edge in wanted_edges[source_id, passenger.kind] for passenger in queues.get(source_id, []))
        return eligible >= min(POD_CAPACITY, self.path_remaining(path, result))
    def edge_demand(self, queues: dict[int, list[Passenger]], wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]]) -> Counter[DirectedPair]:
        demand = Counter()
        for building_id, passengers in queues.items():
            for kind, count in Counter(passenger.kind for passenger in passengers).items():
                for edge in wanted_edges[building_id, kind]:
                    demand[edge] += count
        return demand
    def distances_to_targets(self, state: PlanState) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
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
        queues = {}
        for pad in self.landing_pads():
            passengers = [Passenger(pad.id, kind, pad.id * 1000 + index) for index, kind in enumerate(pad.order) if distances[kind][pad.id] < INF]
            if passengers:
                queues[pad.id] = passengers
        return queues
    def wanted_edges(self, distances: dict[int, dict[int, int]], graph: dict[int, list[int]]) -> dict[tuple[int, int], tuple[DirectedPair, ...]]:
        wanted = {}
        for kind, kind_distances in distances.items():
            for building_id, distance in kind_distances.items():
                options = [neighbor_id for neighbor_id in graph.get(building_id, []) if kind_distances[neighbor_id] < distance]
                wanted[building_id, kind] = tuple((building_id, neighbor_id) for neighbor_id in options)
        return wanted
    def teleport_phase(self, queues: dict[int, list[Passenger]], distances: dict[int, dict[int, int]], teleports: dict[int, int]):
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
    def settle(self, day: int, queues: dict[int, list[Passenger]], arrivals: Counter[int], result: SimulationResult):
        for building_id in sorted(queues):
            building = self.buildings[building_id]
            if building.kind <= 0:
                continue
            remaining = []
            for passenger in queues[building_id]:
                if passenger.kind != building.kind:
                    remaining.append(passenger)
                    continue
                speed = max(0, 50 - day)
                diversity = max(0, 50 - arrivals[building_id])
                score = speed + diversity
                result.score += score
                pool = passenger.pad_id, passenger.kind
                result.speed_by_pool[pool] += speed
                result.delivered_by_pool[pool] += 1
                result.diversity_by_module[building_id] += diversity
                result.delivered_by_module[building_id] += 1
                result.delivered_by_pool_module[pool, building_id] += 1
                if result.delivered_by_pool[pool] == self.buildings[passenger.pad_id].demand[passenger.kind]:
                    result.delivery_times[pool] = day
                arrivals[building_id] += 1
            if remaining:
                queues[building_id] = remaining
            else:
                del queues[building_id]
    def allocate_tube_capacity(self, requests: dict[int, DirectedPair], state: PlanState, demand: Counter[DirectedPair],
            result: SimulationResult, count_congestion: bool = True) -> dict[int, DirectedPair]:
        moves = {}
        by_tube = {}
        for pod_id, move in requests.items():
            by_tube.setdefault(route_key(*move), []).append((pod_id, move))
        for edge, pods in by_tube.items():
            capacity = state.tubes[edge]
            fixed = sorted((pod_id, move) for pod_id, move in pods if not state.pods[pod_id].dynamic)
            dynamic = sorted(((pod_id, move) for pod_id, move in pods if state.pods[pod_id].dynamic),
                key=lambda item: (-min(POD_CAPACITY, demand[item[1]]), item[0]))
            selected = [*fixed, *dynamic][:capacity]
            if count_congestion and len(pods) > capacity:
                result.congestion_by_edge[edge] += 1
            for pod_id, move in selected:
                moves[pod_id] = move
        return moves
    def board_and_launch(self, queues: dict[int, list[Passenger]], distances: dict[int, dict[int, int]], state: PlanState,
            moves: dict[int, DirectedPair], pod_positions: dict[int, int], dynamic_current: dict[int, int],
            dynamic_pending: dict[int, DirectedPair]):
        by_start = {}
        for pod_id, (source_id, target_id) in moves.items():
            by_start.setdefault(source_id, []).append((pod_id, target_id))
        for candidates in by_start.values():
            candidates.sort()
        seats = {pod_id: POD_CAPACITY for pod_id in moves}
        onboard = {}
        for building_id in sorted(queues):
            candidates = by_start.get(building_id, [])
            if not candidates:
                continue
            remaining = []
            for passenger in queues[building_id]:
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
            if remaining:
                queues[building_id] = remaining
            else:
                del queues[building_id]
        for pod_id, (_, target_id) in moves.items():
            if pod_id in pod_positions:
                pod_positions[pod_id] = fixed_next_index(state.pods[pod_id].path, pod_positions[pod_id])
            else:
                dynamic_current[pod_id] = target_id
                dynamic_pending[pod_id] = (-1, -1)
            if onboard.get(pod_id):
                queues.setdefault(target_id, []).extend(onboard[pod_id])
    def shortest_existing_tube_path(self, start_id: int, targets: list[int], tubes: dict[Pair, int]) -> list[int]:
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
        return self.cheapest_path_with_hop_limit(start_id, targets, MAX_TUBE_HOPS, state)
    def cheapest_hop_path(self, start_id: int, targets: list[int], hop_count: int, state: PlanState) -> list[int]:
        path = self.cheapest_path_with_hop_limit(start_id, targets, hop_count, state, exact_hops=True)
        return path
    def cheapest_path_with_hop_limit(self, start_id: int, targets: list[int], hop_limit: int, state: PlanState,
            exact_hops: bool = False, via_nodes: tuple[int, ...] = ()) -> list[int]:
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
        pools = []
        for pad in self.landing_pads():
            pools.extend((pad.id, kind) for kind in sorted(pad.demand))
        return pools
    def landing_pads(self) -> list[Building]:
        return [building for building in sorted(self.buildings.values(), key=lambda item: item.id) if building.kind == 0]
    def tube_degrees(self, tubes: dict[Pair, int]) -> Counter[int]:
        degrees = Counter()
        for a, b in tubes:
            degrees[a] += 1
            degrees[b] += 1
        return degrees
    def teleport_used_buildings(self, teleports: dict[int, int]) -> set[int]:
        used = set(teleports)
        used.update(teleports.values())
        return used
    def next_pod_id(self, pods: dict[int, PodPlan]) -> int:
        for pod_id in range(1, MAX_PODS + 1):
            if pod_id not in pods:
                return pod_id
        raise RuntimeError("No pod identifiers remain")
    def print_debug_input(self):
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
            print(f"pod id={pod_id}, path=[{path_text}]", file=sys.stderr)
    def perfect_diversity(self, kind: int) -> int:
        demand = sum(pad.demand[kind] for pad in self.landing_pads())
        module_count = sum(building.kind == kind for building in self.buildings.values())
        balanced_population = (demand + module_count - 1) // module_count
        return sum(max(0, 50 - index) for index in range(balanced_population))
def route_key(a: int, b: int) -> Pair:
    return (a, b) if a < b else (b, a)
def tube_cost(a: Building, b: Building) -> int:
    return isqrt(100 * ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)))
def orientation(a: Building, b: Building, c: Building) -> int:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
def point_on_segment(point: Building, a: Building, b: Building) -> bool:
    return orientation(a, b, point) == 0 and min(a.x, b.x) <= point.x <= max(a.x, b.x) and min(a.y, b.y) <= point.y <= max(a.y, b.y)
def segments_intersect(a: Building, b: Building, c: Building, d: Building) -> bool:
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
    result = []
    seen = set()
    for a, b in zip(path, path[1:]):
        edge = route_key(a, b)
        if edge not in tubes and edge not in seen:
            result.append(edge)
            seen.add(edge)
    return result
def tube_graph(tubes: dict[Pair, int]) -> dict[int, list[int]]:
    graph = {}
    for a, b in tubes:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return {node: sorted(neighbors) for node, neighbors in graph.items()}
def tube_components(graph: dict[int, list[int]]) -> dict[int, int]:
    components = {}
    for origin_id in sorted(graph):
        if origin_id in components:
            continue
        components[origin_id] = origin_id
        queue = deque([origin_id])
        while queue:
            building_id = queue.popleft()
            for neighbor_id in graph[building_id]:
                if neighbor_id not in components:
                    components[neighbor_id] = origin_id
                    queue.append(neighbor_id)
    return components
def graph_distance(graph: dict[int, list[int]], start_id: int, finish_id: int) -> int:
    if start_id == finish_id:
        return 0
    d = _G.setdefault(id(graph), (graph, {}))[1]
    key = start_id, finish_id
    if key in d:
        return d[key]
    queue = deque([(start_id, 0)])
    seen = {start_id}
    while queue:
        building_id, distance = queue.popleft()
        for neighbor_id in graph.get(building_id, []):
            if neighbor_id == finish_id:
                return d.setdefault(key, distance + 1)
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                queue.append((neighbor_id, distance + 1))
    return d.setdefault(key, INF)
def next_step(graph: dict[int, list[int]], start_id: int, finish_id: int) -> int:
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
    path = [finish_id]
    while path[-1] != start_id:
        path.append(parent[path[-1]])
    path.reverse()
    return path
def normalize_month_path(path: list[int]) -> list[int]:
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
    if len(path) < 2:
        return index
    if index < len(path) - 1:
        return index + 1
    if path[0] == path[-1]:
        return 1
    return index
if __name__ == "__main__":
    Planner().play()
