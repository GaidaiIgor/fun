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
OVERRIDE_MONTH = -1
OVERRIDE_COMMAND = "POD 3 3 6 3 6 3 6 3 4 1 4 3 6 3 6 3 6 3 6 3 6 3;POD 4 3 5 3 5 3 5 3 2 0 2 3 5 3 5 3 5 3 5 3 5 3;POD 5 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2"
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
LayoutKey = tuple[tuple[Pair, ...], tuple[tuple[int, int], ...]]
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
    new_tubes: set[Pair] = field(default_factory=set)
    cost: int = 0
@dataclass(slots=True)
class LayoutBranch:
    layouts: tuple[Bundle, ...]
    state: PlanState
    efficiency: float
    parent_score: int
@dataclass(slots=True)
class PlanOption:
    bundle: Bundle
    layouts: tuple[Bundle, ...]
    layout_key: LayoutKey
    state: PlanState
    parent_score: int
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
    layouts: tuple[Bundle, ...]
    layout_key: LayoutKey
    state: PlanState
    parent_score: int
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
    layout_cache: dict[LayoutKey, LayoutBranch]
    turn_start_result: SimulationResult
    def __init__(self):
        self.buildings = {}
        self.resources = 0
        self.month = 0
        self.tubes = {}
        self.teleports = {}
        self.pods = {}
        self.simulation_cache = {}
        self.fs_cache = {}
        self.layout_cache = {}
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
        layouts = ()
        current_state = self.replay_bundle_sequence(layouts)
        current_result = self.score_state(current_state)
        self.turn_start_result = current_result
        before_score = current_result.score
        current_key = self.layout_key(current_state)
        self.layout_cache = {current_key: LayoutBranch(layouts, current_state, -inf, -1)}
        if FULL_DEBUG:
            debug("\n" + self.score_debug("before", current_result, current_state.cost))
        iteration = 1
        while True:
            best = self.best_candidate(layouts, current_state, current_result, before_score)
            if best is None:
                break
            layouts, current_key, current_state = best.layouts, best.layout_key, best.state
            current_result = self.score_state(current_state)
            self.layout_cache[current_key] = LayoutBranch(layouts, current_state, best.efficiency, best.parent_score)
            if FULL_DEBUG:
                self.selected_debug(best, current_state, current_result, before_score)
                iteration += 1
                debug(f"\nIteration {iteration}\n" + self.status_debug(current_result))
        final_state = current_state
        final_result = self.score_state(final_state, True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths)
        if FULL_DEBUG:
            debug("\n" + self.table_debug(final_result, final_state))
            debug("\n" + self.score_debug("after", final_result, final_state.cost))
        action_order = {"TUBE": 0, "TELEPORT": 0, "UPGRADE": 1, "DESTROY": 2, "POD": 3}
        return sorted((action for action in final_state.actions if action), key=lambda action: action_order[action.split()[0]])
    def override_actions(self) -> list[str]:
        current_state = self.replay_bundle_sequence(())
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
        state = self.replay_bundle_sequence(())
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
    def best_candidate(self, layouts: tuple[Bundle, ...], current_state: PlanState, current_result: SimulationResult,
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
                options = self.generate_options(owner, group, module_ids, layouts, current_state, current_result, before_score)
                candidate = self.next_candidate(owner, pair, group, current_state, current_result, before_score, options)
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
    def next_candidate(self, owner: PoolOwner, pair: PoolOwner, group: Pool, current_state: PlanState,
            current_result: SimulationResult, before_score: int, options: list[PlanOption]) -> Candidate:
        best = None
        seen_states = set()
        plans = []
        for option in options:
            bundle, state = option.bundle, option.state
            if bundle.fingerprint == Bundle(owner).fingerprint and not bundle.path_edges:
                continue
            if bundle.path_edges and current_result.delivery_times.get(group, INF) == bundle.path_length:
                continue
            state_key = tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), \
                tuple(sorted((pod_id, tuple(pod.path), pod.dynamic) for pod_id, pod in state.pods.items()))
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            if current_state.tubes == state.tubes and current_state.teleports == state.teleports and current_state.pods == state.pods:
                continue
            action_text = self.state_action_text(state) if FULL_DEBUG else ""
            plans.append((option, action_text))
        branch = None
        for option, action_text in plans:
            bundle, state = option.bundle, option.state
            next_branch = bundle.teleport != (-1, -1), bundle.path
            if next_branch != branch:
                branch = next_branch
                path_text = ", ".join(map(str, bundle.path))
                mode = "teleport" if next_branch[0] else "path"
                debug(f"    Considering {mode}=[{path_text}]:")
            prefix = "-> " if bundle.debug_chosen else ""
            text = f"      {prefix}{bundle.debug_id}: action={action_text}, "
            if state.cost > self.resources:
                debug(f"{text}local gain=-, global gain=-, cost={state.cost}, efficiency=-")
                continue
            result = self.score_state(state)
            pool_score = result.speed_by_pool[owner] if isinstance(owner, tuple) else result.diversity_by_module[owner]
            start_pool_score = self.turn_start_result.speed_by_pool[owner] if isinstance(owner, tuple) else \
                self.turn_start_result.diversity_by_module[owner]
            local_gain = pool_score - start_pool_score
            global_gain = result.score - before_score
            checkpoint = "" if option.parent_score < 0 else f"({result.score - option.parent_score:+d})"
            efficiency = global_gain / state.cost if state.cost > 0 else inf
            debug(f"{text}local gain={local_gain}, global gain={global_gain}{checkpoint}, cost={state.cost}, "
                f"efficiency={efficiency:.3f}")
            if global_gain > 0 and result.score > current_result.score:
                candidate = Candidate(bundle, pair, global_gain, state.cost, option.layouts, option.layout_key, state,
                    option.parent_score)
                if best is None or (candidate.efficiency, candidate.global_gain, -candidate.global_cost) > \
                        (best.efficiency, best.global_gain, -best.global_cost):
                    best = candidate
        return best
    def generate_options(self, owner: PoolOwner, group: Pool, module_ids: list[int], layouts: tuple[Bundle, ...], state: PlanState,
            current_result: SimulationResult, before_score: int) -> list[PlanOption]:
        bases = []
        pad_id = group[0]
        distances, _ = self.distances_to_targets(state)
        current_length = distances[group[1]][pad_id]
        allow_shorter = True
        if isinstance(owner, int):
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
        else:
            bases = [bundle for bundle in bases if bundle.path_length <= current_length]
        options = []
        connections = [base for base in bases if base.label == "connect"]
        if connections:
            options.extend(self.connection_option_stack(owner, group, connections, layouts, before_score))
        connection_ids = {id(base) for base in connections}
        seen = set()
        for base in bases:
            if id(base) in connection_ids:
                continue
            key = base.path, base.tubes, base.pod_specs
            if key in seen:
                continue
            seen.add(key)
            options.extend(self.path_option_stack(owner, group, base, layouts, before_score))
        teleports = self.teleport_bundles(owner, group, module_ids, state)
        if isinstance(owner, int):
            teleports = [bundle for bundle in teleports
                if bundle.path_length == current_length or allow_shorter and bundle.path_length < current_length]
        options.extend(self.base_option(bundle, layouts, before_score)[0] for bundle in teleports)
        return options
    def connection_option_stack(self, owner: PoolOwner, group: Pool, bases: list[Bundle], layouts: tuple[Bundle, ...],
            before_score: int) -> list[PlanOption]:
        result = []
        options = []
        for base in bases:
            option, parent_efficiency = self.base_option(base, layouts, before_score)
            metrics = self.option_metrics(option, before_score)
            result.append(option)
            if option.state.cost <= self.resources:
                options.append((metrics[2], metrics[0], -metrics[1], option, parent_efficiency))
        if not options:
            return result
        efficiency, _, _, parent, parent_efficiency = max(options, key=lambda item: item[:3])
        parent.bundle.debug_chosen = parent.bundle.debug_id
        result.extend(self.throughput_options(owner, group, parent, max(efficiency, parent_efficiency), before_score))
        return result
    def path_option_stack(self, owner: PoolOwner, group: Pool, base: Bundle, layouts: tuple[Bundle, ...],
            before_score: int) -> list[PlanOption]:
        parent, parent_efficiency = self.base_option(base, layouts, before_score)
        result = [parent]
        if parent.state.cost > self.resources:
            return result
        if base.tubes:
            parent.bundle.debug_chosen = parent.bundle.debug_id
            parent_efficiency = max(parent_efficiency, self.option_metrics(parent, before_score)[2])
        else:
            parent_efficiency = -inf
        result.extend(self.throughput_options(owner, group, parent, parent_efficiency, before_score))
        return result
    def throughput_options(self, owner: PoolOwner, group: Pool, parent: PlanOption, parent_efficiency: float,
            before_score: int) -> list[PlanOption]:
        result = []
        while parent.state.cost <= self.resources:
            simulation = self.cached_simulate(parent.state)
            if simulation.delivery_times.get(group, INF) == parent.bundle.path_length:
                break
            options = []
            parent_score = self.score_state(parent.state).score
            pod_bundle = Bundle(owner, pod_specs=(0,), label=f"{parent.bundle.label}-pod", path_edges=parent.bundle.path_edges,
                destination=parent.bundle.destination, path_length=parent.bundle.path_length, path=parent.bundle.path)
            pod_option = PlanOption(pod_bundle, parent.layouts, parent.layout_key,
                self.replay_bundle_on_state(parent.state, pod_bundle), parent_score)
            pod_bundle.debug_id = self.bundle_debug_id(pod_option.state)
            pod_metrics = self.option_metrics(pod_option, before_score)
            if len(parent.state.pods) < sum(parent.state.tubes.values()):
                options.append((pod_option, pod_metrics))
            upgrade_edge = self.best_counter_edge(parent.bundle.path_edges, simulation.congestion_by_edge)
            if upgrade_edge != (-1, -1):
                upgrade_bundle = Bundle(owner, upgrades=(upgrade_edge,), label=f"{parent.bundle.label}-upgrade",
                    path_edges=parent.bundle.path_edges, destination=parent.bundle.destination, path_length=parent.bundle.path_length,
                    path=parent.bundle.path)
                upgrade_option = PlanOption(upgrade_bundle, parent.layouts, parent.layout_key,
                    self.replay_bundle_on_state(parent.state, upgrade_bundle), parent_score)
                upgrade_bundle.debug_id = self.bundle_debug_id(upgrade_option.state)
                options.append((upgrade_option, self.option_metrics(upgrade_option, before_score)))
            combined_affordable = pod_option.state.cost <= self.resources
            if combined_affordable:
                edge = self.best_counter_edge(parent.bundle.path_edges, self.cached_simulate(pod_option.state).congestion_by_edge)
                if edge != (-1, -1):
                    combined_affordable = False
                    upgrade_cost = tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * (parent.state.tubes[edge] + 1)
                    if parent.state.cost + POD_COST + upgrade_cost <= self.resources:
                        combined = Bundle(owner, pod_specs=(0,), upgrades=(edge,), label=f"{parent.bundle.label}-pod-upgrade",
                            path_edges=parent.bundle.path_edges, destination=parent.bundle.destination,
                            path_length=parent.bundle.path_length, path=parent.bundle.path)
                        combined_option = PlanOption(combined, parent.layouts, parent.layout_key,
                            self.replay_bundle_on_state(parent.state, combined), parent_score)
                        combined.debug_id = self.bundle_debug_id(combined_option.state)
                        combined_metrics = self.option_metrics(combined_option, before_score)
                        options.append((combined_option, combined_metrics))
                        combined_affordable = combined_option.state.cost <= self.resources
            result.extend(option for option, _ in options)
            affordable = [(metrics[2], metrics[0], -metrics[1], option) for option, metrics in options
                if option.state.cost <= self.resources]
            if not affordable:
                break
            efficiency, _, _, next_parent = max(affordable, key=lambda item: item[:3])
            if efficiency <= parent_efficiency or not combined_affordable:
                break
            next_parent.bundle.debug_chosen = next_parent.bundle.debug_id
            parent, parent_efficiency = next_parent, efficiency
        return result
    def option_metrics(self, option: PlanOption, before_score: int) -> tuple[int, int, float]:
        cost = option.state.cost
        if cost > self.resources:
            return 0, cost, -inf
        gain = self.score_state(option.state).score - before_score
        return gain, cost, gain / cost if cost > 0 else inf if gain > 0 else 0
    def base_option(self, base: Bundle, layouts: tuple[Bundle, ...], before_score: int) -> tuple[PlanOption, float]:
        next_layouts = layouts
        if base.tubes or base.teleport != (-1, -1):
            layout = Bundle(base.pool, tubes=base.tubes, teleport=base.teleport, label=base.label, path_edges=base.path_edges,
                destination=base.destination, path_length=base.path_length, path=base.path)
            next_layouts = (*layouts, layout)
        layout_state = self.replay_bundle_sequence(next_layouts)
        key = self.layout_key(layout_state)
        if key in self.layout_cache:
            branch = self.layout_cache[key]
            state = self.copy_state(branch.state)
            base.debug_id = self.bundle_debug_id(state)
            return PlanOption(base, next_layouts, key, state, branch.parent_score), branch.efficiency
        state = layout_state
        parent_score = -1
        if base.tubes or not state.pods and self.has_tube_loads(state):
            pod_id = self.closest_pod(base.path[0], base.path_edges, state) if base.tubes else 0
            base.pod_specs = (pod_id,)
            parent_score = self.score_state(layout_state).score
            state = self.replay_bundle_on_state(state, Bundle(base.pool, pod_specs=(pod_id,)))
        base.debug_id = self.bundle_debug_id(state)
        option = PlanOption(base, next_layouts, key, state, parent_score)
        efficiency = self.option_metrics(option, before_score)[2]
        self.layout_cache[key] = LayoutBranch(next_layouts, self.copy_state(state), efficiency, parent_score)
        return option, efficiency
    def bundle_debug_id(self, state: PlanState) -> str:
        upgrades = sum(capacity - self.tubes.get(edge, 1) for edge, capacity in state.tubes.items())
        return f"{len(state.ops)}p{upgrades}u"
    def has_tube_loads(self, state: PlanState) -> bool:
        distances, module_distances = self.distances_to_targets(state)
        return bool(self.path_demands(state, distances, module_distances))
    def layout_key(self, state: PlanState) -> LayoutKey:
        return tuple(sorted(state.tubes)), tuple(sorted(state.teleports.items()))
    def connection_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState) -> list[Bundle]:
        path = self.cheapest_connecting_path(group[0], module_ids, state)
        network_nodes = {node for edge in state.tubes for node in edge}
        if not network_nodes or network_nodes.intersection(path):
            return self.path_bundles(owner, "connect", path, state)
        connected = self.network_connection_bundle(owner, group, module_ids, state, network_nodes)
        return [connected] if connected else []
    def network_connection_bundle(self, owner: PoolOwner, group: Pool, module_ids: list[int], state: PlanState,
            network_nodes: set[int]) -> Bundle:
        best = None
        routes = []
        path = self.cheapest_path_with_hop_limit(group[0], module_ids, MAX_TUBE_HOPS, state, via_nodes=tuple(network_nodes))
        if path:
            routes.append((tuple(route_key(a, b) for a, b in zip(path, path[1:])), path[-1]))
        for module_id in module_ids:
            base_path = self.cheapest_connecting_path(group[0], [module_id], state)
            base_edges = tuple(route_key(a, b) for a, b in zip(base_path, base_path[1:]))
            for junction_id in base_path:
                remaining_hops = MAX_TUBE_HOPS - len(base_edges)
                connector = [junction_id] if junction_id in network_nodes else \
                    self.cheapest_path_with_hop_limit(junction_id, list(network_nodes), remaining_hops, state)
                if not connector:
                    continue
                edges = tuple(dict.fromkeys((*base_edges, *(route_key(a, b) for a, b in zip(connector, connector[1:])))))
                if len(edges) <= MAX_TUBE_HOPS and self.can_add_tubes([edge for edge in edges if edge not in state.tubes], state.tubes):
                    routes.append((edges, module_id))
        for path_edges, module_id in routes:
            tubes = tuple(edge for edge in path_edges if edge not in state.tubes)
            cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes)
            projected_tubes = dict(state.tubes)
            projected_tubes.update((edge, 1) for edge in path_edges)
            route = self.shortest_existing_tube_path(group[0], [module_id], projected_tubes)
            bundle = Bundle(owner, tubes=tubes, label="connect", path_edges=path_edges,
                destination=module_id, path_length=len(route) - 1, path=tuple(route))
            source = self.buildings[group[0]]
            target = self.buildings[module_id]
            distance = (source.x - target.x) * (source.x - target.x) + (source.y - target.y) * (source.y - target.y)
            order = cost, distance, len(route), path_edges
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
        return bundles
    def path_bundles(self, owner: PoolOwner, label: str, path: list[int], state: PlanState) -> list[Bundle]:
        if not path:
            return []
        path_edges = self.connected_path_edges(path, state)
        if not path_edges:
            return []
        tubes = tuple(edge for edge in path_edges if edge not in state.tubes)
        return [Bundle(owner, tubes=tubes, label=label, path_edges=path_edges,
            destination=path[-1], path_length=len(path) - 1, path=tuple(path))]
    def connected_path_edges(self, path: list[int], state: PlanState) -> tuple[Pair, ...]:
        base_edges = tuple(route_key(a, b) for a, b in zip(path, path[1:]))
        network_nodes = {node for edge in state.tubes for node in edge}
        if not network_nodes or network_nodes.intersection(path):
            return base_edges
        remaining_hops = MAX_TUBE_HOPS - len(base_edges)
        best = None
        for junction_id in path:
            connector = self.cheapest_path_with_hop_limit(junction_id, list(network_nodes), remaining_hops, state)
            if not connector:
                continue
            edges = tuple(dict.fromkeys((*base_edges, *(route_key(a, b) for a, b in zip(connector, connector[1:])))))
            tubes = [edge for edge in edges if edge not in state.tubes]
            if len(edges) > MAX_TUBE_HOPS or not self.can_add_tubes(tubes, state.tubes):
                continue
            cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes)
            order = cost, len(edges), edges
            if best is None or order < best[0]:
                best = order, edges
        return best[1] if best else ()
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
        copied = self.copy_state(state)
        self.apply_bundle(copied, bundle)
        return copied
    def copy_state(self, state: PlanState) -> PlanState:
        pods = {pod_id: PodPlan(pod.path[:], pod.dynamic) for pod_id, pod in state.pods.items()}
        return PlanState(dict(state.tubes), dict(state.teleports), pods, list(state.actions), list(state.pod_slots),
            set(state.ops), set(state.new_tubes), state.cost)
    def replay_bundle_sequence(self, selected: tuple[Bundle, ...]) -> PlanState:
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
        for edge in sorted(state.new_tubes - active):
            remaining = dict(state.tubes)
            del remaining[edge]
            if edge in freed or graph_distance(tube_graph(remaining), *edge) < INF:
                self.remove_planned_tube(state, edge)
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
            if pod_id and pod_id in state.ops:
                continue
            if pod_id:
                if pod_id not in state.pods:
                    raise ValueError("missing reroute pod")
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
        key = tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), pods
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
            self.fs_cache[network_key] = distances, graph, self.wanted_edges(distances, graph), \
                self.path_demands(state, distances, module_distances), self.initial_queues(distances)
        distances, graph, wanted_edges, path_demands, initial = self.fs_cache[network_key]
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
                requests = self.path_pod_requests(fixed_pods, [], pod_positions, {}, {}, {}, graph, result)
                moves = self.allocate_tube_capacity(requests, state, Counter(), result)
                self.board_and_launch(queues, distances, state, moves, pod_positions, {}, {})
                self.settle(day + 1, queues, arrivals, result)
                continue
            fixed_assignments, _, _ = self.fixed_load_assignments(fixed_pods, pod_positions, active, queues, wanted_edges, result)
            self.fixed_edges = tuple(sorted(tuple(sorted({route_key(a, b) for path in paths
                for a, b in zip(path.nodes, path.nodes[1:])})) for paths in fixed_assignments.values()))
            demand = self.edge_demand(queues, wanted_edges)
            reserved_loads = Counter()
            for _, reservations in fixed_schedule[day:]:
                reserved_loads.update(reservations)
            assignments, preferences = self.dispatch_dynamic_paths(active, dynamic_pods, dynamic_current, reserved_loads,
                result, state, graph)
            assignments, requests, moves = self.resolve_dispatch_congestion(assignments, preferences, fixed_pods, dynamic_pods,
                pod_positions, dynamic_current, dynamic_pending, graph, result, state, demand)
            if keep_dynamic_paths:
                loads = ", ".join("({})x{}{}".format("-".join(map(str, path.nodes)), self.path_remaining(path, result),
                    "H" if path.ambiguous else "N") for path in active)
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
    def dispatch_dynamic_paths(self, active: list[PathDemand], dynamic_pods: list[tuple[int, PodPlan]], current: dict[int, int],
            reserved_loads: Counter[LoadKey], result: SimulationResult, state: PlanState,
            graph: dict[int, list[int]]) -> tuple[dict[int, PathDemand], dict[int, list[PathDemand]]]:
        boarding = {path: min(POD_CAPACITY, max(0, self.path_remaining(path, result) -
            reserved_loads[path.pool, path.destination, path.nodes[:2]])) for path in active}
        assignments = {}
        preferences = {}
        counts = Counter()
        for pod_id, _ in dynamic_pods:
            options = [path for path in active if current[pod_id] == -1 or graph_distance(graph, current[pod_id], path.nodes[0]) < INF]
            preferences[pod_id] = sorted(options, key=lambda path: self.path_assignment_key(
                path, pod_id, assignments, counts, boarding, current, result, state, graph))
            if preferences[pod_id]:
                assignments[pod_id] = preferences[pod_id][0]
                counts[assignments[pod_id]] += 1
        return assignments, preferences
    def resolve_dispatch_congestion(self, assignments: dict[int, PathDemand], preferences: dict[int, list[PathDemand]],
            fixed_pods: list[tuple[int, PodPlan]], dynamic_pods: list[tuple[int, PodPlan]], pod_positions: dict[int, int],
            current: dict[int, int], pending: dict[int, DirectedPair], graph: dict[int, list[int]], result: SimulationResult,
            state: PlanState, demand: Counter[DirectedPair]) -> tuple:
        def requests_for(values: dict[int, PathDemand]) -> dict[int, DirectedPair]:
            return self.path_pod_requests(fixed_pods, dynamic_pods, pod_positions, current, pending, values, graph, result)
        def conflicts(values: dict[int, DirectedPair]) -> Counter[Pair]:
            counts = Counter(route_key(*move) for move in values.values())
            return Counter({edge: count - state.tubes[edge] for edge, count in counts.items() if count > state.tubes[edge]})
        def improves(trial: Counter[Pair], original: Counter[Pair], edge: Pair) -> bool:
            return trial[edge] < original[edge] and all(count <= original[other] for other, count in trial.items() if other != edge)
        requests = requests_for(assignments)
        congestion = conflicts(requests)
        result.congestion_by_edge.update(congestion.keys())
        unresolved = set()
        while remaining := sorted(set(congestion) - unresolved):
            edge = remaining[0]
            pods = sorted(pod_id for pod_id in assignments if route_key(*requests[pod_id]) == edge)
            changed = False
            for index, pod_id in enumerate(pods):
                for other_id in pods[index + 1:]:
                    if requests[pod_id] != requests[other_id][::-1] or assignments[pod_id] == assignments[other_id]:
                        continue
                    trial = dict(assignments)
                    trial[pod_id], trial[other_id] = trial[other_id], trial[pod_id]
                    trial_requests = requests_for(trial)
                    trial_congestion = conflicts(trial_requests)
                    if improves(trial_congestion, congestion, edge):
                        assignments, requests, congestion, changed = trial, trial_requests, trial_congestion, True
                        break
                if changed:
                    break
            if not changed and pods:
                pod_id = pods[0]
                start = preferences[pod_id].index(assignments[pod_id]) + 1
                for path in preferences[pod_id][start:]:
                    trial = dict(assignments)
                    trial[pod_id] = path
                    trial_requests = requests_for(trial)
                    trial_congestion = conflicts(trial_requests)
                    target = route_key(*trial_requests[pod_id])
                    if not trial_congestion[target] and improves(trial_congestion, congestion, edge):
                        assignments, requests, congestion, changed = trial, trial_requests, trial_congestion, True
                        break
            if changed:
                unresolved.intersection_update(congestion)
            else:
                unresolved.add(edge)
        return assignments, requests, self.allocate_tube_capacity(requests, state, demand, result, False)
    def path_assignment_key(self, path: PathDemand, pod_id: int, assignments: dict[int, PathDemand], counts: Counter[PathDemand],
            boarding: dict[PathDemand, int], current: dict[int, int], result: SimulationResult, state: PlanState,
            graph: dict[int, list[int]]) -> tuple:
        distance = 0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])
        return self.path_capacity_excess(path, assignments, state), not path.ambiguous, counts[path], -boarding[path], distance, \
            len(path.nodes) - 1, result.delivered_by_module[path.destination], -self.path_remaining(path, result), path.pool, \
            path.destination, path.nodes
    def path_capacity_excess(self, candidate: PathDemand, assignments: dict[int, PathDemand], state: PlanState) -> bool:
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
        paths = sorted([*(path.nodes for path in assignments.values()), candidate.nodes])
        fixed = self.fixed_edges
        key = self.capacities, tuple(paths), fixed
        if key not in self.cap_cache:
            used = Counter(edge for edges in fixed for edge in edges)
            edges = [[route_key(a, b) for a, b in zip(path, path[1:])] for path in paths]
            required = {edge for path_edges in edges for edge in path_edges}
            slots = {edge: [(edge, index) for index in range(max(0, state.tubes[edge] - used[edge]))] for edge in required}
            owners = {}
            self.cap_cache[key] = any(not place(worker, set()) for worker in range(len(paths)))
        return self.cap_cache[key]
    def path_pod_requests(self, fixed_pods: list[tuple[int, PodPlan]], dynamic_pods: list[tuple[int, PodPlan]],
            pod_positions: dict[int, int], current: dict[int, int], pending: dict[int, DirectedPair], assignments: dict[int, PathDemand],
            graph: dict[int, list[int]], result: SimulationResult) -> dict[int, DirectedPair]:
        requests = {}
        for pod_id, pod in fixed_pods:
            index = pod_positions[pod_id]
            next_index = fixed_next_index(pod.path, index)
            if next_index != index:
                requests[pod_id] = pod.path[index], pod.path[next_index]
        options = {}
        claims = Counter()
        for pod_id, _ in dynamic_pods:
            if pending[pod_id] != (-1, -1):
                options[pod_id] = [pending[pod_id]]
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
                if claims[key] * POD_CAPACITY < waiting:
                    moves.insert(0, (path[0], path[1]))
                    claims[key] += 1
                options[pod_id] = list(dict.fromkeys(moves))
                continue
            if current[pod_id] not in path:
                options[pod_id] = [(current[pod_id], next_step(graph, current[pod_id], path[0]))]
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
        requests.update((pod_id, moves[0]) for pod_id, moves in options.items())
        return requests
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
