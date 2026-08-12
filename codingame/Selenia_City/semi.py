from collections import Counter, deque
from dataclasses import dataclass, field, replace
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
INF = 10 ** 9
OVERRIDE_MONTH = 15
OVERRIDE_COMMAND = "WAIT"
FULL_DEBUG = False
_G = {}
BY_ID = attrgetter("id")
Pair = tuple[int, int]
DirectedPair = tuple[int, int]
Pool = tuple[int, int]
PoolOwner = Pool | int
PathKey = tuple[int, ...]
LayoutKey = tuple[tuple[Pair, ...], tuple[tuple[int, int], ...]]
Feature = tuple[str, Pair | int, int, float, bool]
def debug(text: str):
    if FULL_DEBUG:
        print(text, file=sys.stderr)
def score_efficiency(gain: int, cost: int) -> float:
    """Calculates efficiency for gain and cost, treating beneficial free changes as infinite."""
    return gain / cost if cost > 0 else inf if gain > 0 else 0
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
    priority: int = field(default=0, compare=False)
    reserved: bool = False
    h: int = field(init=False, compare=False)
    def __post_init__(self):
        object.__setattr__(self, "h", hash((self.pool, self.destination, self.nodes, self.cap, self.reserved)))
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
    round_number: int = 0
    prune_unused: bool = False
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
    features: list[Feature] = field(default_factory=list)
@dataclass(slots=True)
class LayoutBranch:
    state: PlanState
@dataclass(slots=True)
class PlanOption:
    bundle: Bundle
    layouts: tuple[Bundle, ...]
    layout_key: LayoutKey
    state: PlanState
    round_score: int
    round_cost: int
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
    congestion_by_day: dict[int, Counter[Pair]] = field(default_factory=dict)
    dynamic_paths: dict[int, list[int]] = field(default_factory=dict)
    initial_table: list[list[str]] = field(default_factory=list)
    table: list[list[str]] = field(default_factory=list)
    reserved: str = "-"
@dataclass(slots=True)
class Candidate:
    bundle: Bundle
    pair: PoolOwner
    marginal_gain: int
    marginal_cost: int
    round_gain: int
    round_cost: int
    layouts: tuple[Bundle, ...]
    layout_key: LayoutKey
    state: PlanState
    @property
    def efficiency(self) -> float:
        return score_efficiency(self.marginal_gain, self.marginal_cost)
class Planner:
    buildings: dict[int, Building]
    resources: int
    month: int
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, PodPlan]
    simulation_cache: dict[tuple, SimulationResult]
    path_cache: dict[tuple, dict[int, list[int]]]
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
        self.path_cache = {}
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
        self.path_cache = {}
        self.fs_cache = {}
        _G.clear()
        if self.month + 1 == OVERRIDE_MONTH:
            return self.override_actions()
        layouts = ()
        current_state = self.replay_bundle_sequence(layouts)
        current_result = self.score_state(current_state)
        self.turn_start_result = current_result
        before_score = current_result.score
        current_key = self.layout_key(current_state)
        self.layout_cache = {current_key: LayoutBranch(current_state)}
        if FULL_DEBUG:
            debug("\n" + self.score_debug("before", current_result, current_state.cost))
        iteration = 1
        while True:
            best = self.best_candidate(layouts, current_state, current_result, before_score)
            if best is None:
                break
            layouts, current_key, current_state = best.layouts, best.layout_key, best.state
            current_state.features = [(kind, key, cost, best.efficiency, False) if pending else feature
                for feature in current_state.features for kind, key, cost, _, pending in (feature,)]
            current_result = self.score_state(current_state)
            self.layout_cache = {current_key: LayoutBranch(current_state)}
            if FULL_DEBUG:
                self.selected_debug(best, current_state, current_result, before_score)
                iteration += 1
                debug(f"\nIteration {iteration}\n" + self.status_debug(current_result))
        debug("\nCleanup:")
        for feature in list(current_state.features):
            kind, key, _, _, _ = feature
            if kind not in ("upgrade", "reroute"):
                continue
            trial = self.copy_state(current_state)
            self.cancel_feature(trial, feature)
            trial_result = self.score_state(trial)
            action = f"DROP UPGRADE {key[0]} {key[1]}" if kind == "upgrade" else f"REVERT POD {key}"
            keep = trial_result.score >= current_result.score
            debug("  {}: score={}, {}".format(action, trial_result.score, "kept" if keep else "rejected"))
            if keep:
                current_state, current_result = trial, trial_result
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
        parts = action.split()
        pod_id = int(parts[1])
        if pod_id in state.pods:
            del state.pods[pod_id]
            state.cost -= POD_REFUND
            state.actions.append(f"DESTROY {pod_id}")
        if len(parts) == 2:
            state.cost += POD_COST
            state.pods[pod_id] = PodPlan([], True)
            state.pod_slots.append((len(state.actions), pod_id))
            state.actions.append("")
            return
        path = [int(item) for item in parts[2:]]
        state.cost += POD_COST
        state.pods[pod_id] = PodPlan(path)
        state.actions.append(action)
    def best_candidate(self, layouts: tuple[Bundle, ...], current_state: PlanState, current_result: SimulationResult,
            before_score: int) -> Candidate:
        pools = []
        distances, module_distances = self.distances_to_targets(current_state)
        for pool in self.speed_pools():
            missing = self.buildings[pool[0]].demand[pool[1]] * 50 - current_result.speed_by_pool[pool]
            if missing > 0:
                pairs = []
                for module in sorted(self.buildings.values(), key=BY_ID):
                    if module.kind != pool[1]:
                        continue
                    eligibility = self.layout_path_eligibility(pool, module.id, current_state, current_result, distances,
                        module_distances, False)
                    if any(eligibility):
                        pairs.append((module.id, pool, [module.id], eligibility))
                pools.append((missing, (0, pool[0], pool[1]), pool, pairs))
        for module in sorted(self.buildings.values(), key=lambda item: item.id):
            if module.kind <= 0:
                continue
            missing = self.perfect_diversity(module.kind) - current_result.diversity_by_module[module.id]
            if missing <= 0:
                continue
            pairs = []
            for group in self.speed_pools():
                if group[1] != module.kind:
                    continue
                eligibility = self.layout_path_eligibility(group, module.id, current_state, current_result, distances,
                    module_distances, True)
                if any(eligibility):
                    pairs.append((group, group, [module.id], eligibility))
            pools.append((missing, (1, module.id), module.id, pairs))
        pools.sort(key=lambda item: (-item[0], item[1]))
        if not pools:
            return None
        for _, _, owner, pairs in pools:
            debug(f"Considering {owner}:")
            best = None
            affordable = False
            for pair, group, module_ids, eligibility in pairs:
                debug(f"  Considering {pair}:")
                options = self.generate_options(owner, group, module_ids, layouts, current_state, current_result, eligibility)
                candidate, pair_affordable = self.next_candidate(owner, pair, current_state, current_result, before_score, options)
                affordable |= pair_affordable
                if candidate and (best is None or (candidate.efficiency, candidate.marginal_gain, -candidate.marginal_cost) >
                        (best.efficiency, best.marginal_gain, -best.marginal_cost)):
                    best = candidate
            if best or affordable:
                return best
        return None
    def layout_path_eligibility(self, group: Pool, module_id: int, state: PlanState, result: SimulationResult,
            distances: dict[int, dict[int, int]], module_distances: dict[int, dict[int, int]], diversity: bool) \
            -> tuple[bool, bool]:
        current_length = distances[group[1]][group[0]]
        active = current_length < INF and module_distances[module_id][group[0]] == current_length
        equal_gain, shorter_gain = self.diversity_reroute_eligibility(group, module_id, result, module_distances) \
            if diversity else (active, True)
        equal = equal_gain and current_length < INF and bool(self.cheapest_hop_path(group[0], [module_id], current_length, state))
        hop_limit = len(self.buildings) - 1 if current_length >= INF else min(current_length - 1, len(self.buildings) - 1)
        shorter = shorter_gain and hop_limit > 0 and bool(self.cheapest_path_with_hop_limit(group[0], [module_id], hop_limit, state))
        used = self.teleport_used_buildings(state.teleports)
        shorter |= shorter_gain and current_length > 0 and group[0] not in used and module_id not in used
        return equal, shorter
    def diversity_reroute_eligibility(self, group: Pool, module_id: int, result: SimulationResult,
            module_distances: dict[int, dict[int, int]]) -> tuple[bool, bool]:
        modules = sorted((building for building in self.buildings.values() if building.kind == group[1]), key=BY_ID)
        current_length = min(module_distances[module.id][group[0]] for module in modules)
        active = [module.id for module in modules if current_length < INF and module_distances[module.id][group[0]] == current_length]
        if not active:
            return False, False
        populations = Counter({module.id: result.delivered_by_module[module.id] for module in modules})
        for (delivered_group, delivered_module_id), count in result.delivered_by_pool_module.items():
            if delivered_group == group:
                populations[delivered_module_id] -= count
        allocations = Counter()
        for index in range(self.buildings[group[0]].demand[group[1]]):
            allocations[active[index % len(active)]] += 1
        populations.update(allocations)
        losses = []
        for source_id, count in allocations.items():
            if source_id != module_id:
                losses.extend(max(0, 51 - populations[source_id] + offset) for offset in range(count))
        losses.sort()
        delta = 0
        partial = False
        for moved, loss in enumerate(losses, 1):
            delta += max(0, 51 - populations[module_id] - moved) - loss
            partial |= moved < len(losses) and delta > 0
        return partial, bool(losses and delta > 0)
    def next_candidate(self, owner: PoolOwner, pair: PoolOwner, current_state: PlanState, current_result: SimulationResult,
            before_score: int, options: list[PlanOption]) -> tuple[Candidate, bool]:
        best = None
        seen_states = set()
        plans = []
        for option in options:
            bundle, state = option.bundle, option.state
            if bundle.fingerprint == Bundle(owner).fingerprint and not bundle.path_edges:
                continue
            state_key = tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), \
                tuple(sorted((pod_id, tuple(pod.path), pod.dynamic) for pod_id, pod in state.pods.items())), tuple(state.features)
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            if current_state.tubes == state.tubes and current_state.teleports == state.teleports and current_state.pods == state.pods:
                continue
            action_text = self.state_action_text(state, current_state) if FULL_DEBUG else ""
            plans.append((option, action_text))
        branch = None
        round_number = -1
        for option, action_text in plans:
            bundle, state = option.bundle, option.state
            next_branch = bundle.teleport != (-1, -1), bundle.path
            if next_branch != branch:
                branch = next_branch
                path_text = ", ".join(map(str, bundle.path))
                mode = "teleport" if next_branch[0] else "path"
                debug(f"    Considering {mode}=[{path_text}]:")
                round_number = -1
            if not next_branch[0] and bundle.round_number != round_number:
                round_number = bundle.round_number
                debug(f"      Round {round_number}:")
            prefix = "-> " if bundle.debug_chosen else ""
            indent = "      " if next_branch[0] else "        "
            text = f"{indent}{prefix}{bundle.debug_id}: action={action_text}, "
            costs = state.cost, state.cost - current_state.cost, state.cost - option.round_cost
            cost_text = "/".join(f"{cost:+d}" for cost in costs)
            if state.cost > self.resources:
                debug(f"{text}global gains=-/-/-, costs={cost_text}, efficiencies=-/-/-")
                continue
            result = self.score_state(state)
            gains = result.score - before_score, result.score - current_result.score, result.score - option.round_score
            efficiencies = tuple(score_efficiency(gain, cost) for gain, cost in zip(gains, costs))
            gain_text = "/".join(f"{gain:+d}" for gain in gains)
            efficiency_text = "/".join(f"{efficiency:+.3f}" for efficiency in efficiencies)
            debug(f"{text}global gains={gain_text}, costs={cost_text}, efficiencies={efficiency_text}")
            if gains[0] > 0 and gains[1] > 0:
                candidate = Candidate(bundle, pair, gains[1], costs[1], gains[2], costs[2], option.layouts, option.layout_key, state)
                if best is None or (candidate.efficiency, candidate.marginal_gain, -candidate.marginal_cost) > \
                        (best.efficiency, best.marginal_gain, -best.marginal_cost):
                    best = candidate
        return best, any(option.state.cost <= self.resources for option, _ in plans)
    def generate_options(self, owner: PoolOwner, group: Pool, module_ids: list[int], layouts: tuple[Bundle, ...], state: PlanState,
            current_result: SimulationResult, eligibility: tuple[bool, bool]) -> list[PlanOption]:
        bases = []
        pad_id = group[0]
        distances, _ = self.distances_to_targets(state)
        current_length = distances[group[1]][pad_id]
        allow_equal, allow_shorter = eligibility
        if allow_equal:
            equal_path = self.cheapest_hop_path(pad_id, module_ids, current_length, state)
            bases.extend(self.path_bundles(owner, f"equal-{current_length}", equal_path, state))
        if allow_shorter:
            hop_limit = len(self.buildings) - 1 if current_length >= INF else min(current_length - 1, len(self.buildings) - 1)
            if current_length >= INF:
                path = self.cheapest_path_with_hop_limit(pad_id, module_ids, hop_limit, state)
                bases.extend(self.path_bundles(owner, "cheapest", path, state))
            else:
                bases.extend(self.shortest_route_bundles(owner, group, module_ids, hop_limit, state))
        for base in bases:
            base.prune_unused = base.path_length < current_length
        options = []
        seen = set()
        for base in bases:
            key = base.path, base.tubes, base.pod_specs
            if key in seen:
                continue
            seen.add(key)
            options.extend(self.path_option_stack(owner, group, base, layouts, state, current_result.score, state.cost))
        if allow_shorter:
            for bundle in self.teleport_bundles(owner, group, module_ids, state):
                bundle.prune_unused = True
                options.extend(option for option, _ in self.base_options(bundle, layouts, state, current_result.score, state.cost))
        return options
    def path_option_stack(self, owner: PoolOwner, group: Pool, base: Bundle, layouts: tuple[Bundle, ...],
            inherited_state: PlanState, checkpoint_score: int, checkpoint_cost: int) -> list[PlanOption]:
        base_options = self.base_options(base, layouts, inherited_state, checkpoint_score, checkpoint_cost)
        result = [option for option, _ in base_options]
        affordable = []
        for option, efficiency in base_options:
            if option.state.cost <= self.resources:
                gain, cost, _ = self.option_metrics(option)
                affordable.append((efficiency, gain, -cost, option))
        if not affordable:
            return result
        parent_efficiency, _, _, parent = max(affordable, key=lambda item: item[:3])
        parent.bundle.debug_chosen = parent.bundle.debug_id
        result.extend(self.throughput_options(owner, group, parent, parent_efficiency))
        return result
    def throughput_options(self, owner: PoolOwner, group: Pool, parent: PlanOption,
            parent_efficiency: float) -> list[PlanOption]:
        result = []
        round_number = 1
        allow_reroute = True
        while parent.state.cost <= self.resources:
            simulation = self.cached_simulate(parent.state)
            parent_score = self.score_state(parent.state).score
            options = []
            reroute_id = self.closest_fixed_pod(group[0], parent.state) if allow_reroute else None
            if reroute_id is not None:
                reroute_bundle = Bundle(owner, pod_specs=(reroute_id,), label=f"{parent.bundle.label}-reroute",
                    path_edges=parent.bundle.path_edges, destination=parent.bundle.destination,
                    path_length=parent.bundle.path_length, path=parent.bundle.path, round_number=round_number)
                reroute_option = PlanOption(reroute_bundle, parent.layouts, parent.layout_key,
                    self.replay_bundle_on_state(parent.state, reroute_bundle), parent_score, parent.state.cost)
                reroute_bundle.debug_id = self.bundle_debug_id(reroute_option.state)
                options.append((reroute_option, self.option_metrics(reroute_option)))
            pod_bundle = Bundle(owner, pod_specs=(0,), label=f"{parent.bundle.label}-pod", path_edges=parent.bundle.path_edges,
                destination=parent.bundle.destination, path_length=parent.bundle.path_length, path=parent.bundle.path,
                round_number=round_number)
            pod_option = PlanOption(pod_bundle, parent.layouts, parent.layout_key,
                self.replay_bundle_on_state(parent.state, pod_bundle), parent_score, parent.state.cost)
            pod_bundle.debug_id = self.bundle_debug_id(pod_option.state)
            pod_metrics = self.option_metrics(pod_option)
            pod_added = len(pod_option.state.pods) > len(parent.state.pods)
            if pod_added and len(parent.state.pods) < sum(parent.state.tubes.values()):
                options.append((pod_option, pod_metrics))
            upgrade_edge = self.best_counter_edge(parent.bundle.path_edges, simulation.congestion_by_edge)
            if upgrade_edge != (-1, -1):
                upgrade_bundle = Bundle(owner, upgrades=(upgrade_edge,), label=f"{parent.bundle.label}-upgrade",
                    path_edges=parent.bundle.path_edges, destination=parent.bundle.destination, path_length=parent.bundle.path_length,
                    path=parent.bundle.path, round_number=round_number)
                upgrade_option = PlanOption(upgrade_bundle, parent.layouts, parent.layout_key,
                    self.replay_bundle_on_state(parent.state, upgrade_bundle), parent_score, parent.state.cost)
                upgrade_bundle.debug_id = self.bundle_debug_id(upgrade_option.state)
                options.append((upgrade_option, self.option_metrics(upgrade_option)))
            combined_edge = (-1, -1)
            if pod_added and pod_option.state.cost <= self.resources:
                combined_edge = self.best_counter_edge(parent.bundle.path_edges,
                    self.cached_simulate(pod_option.state).congestion_by_edge)
            if combined_edge != (-1, -1):
                combined = Bundle(owner, pod_specs=(0,), upgrades=(combined_edge,), label=f"{parent.bundle.label}-pod-upgrade",
                    path_edges=parent.bundle.path_edges, destination=parent.bundle.destination,
                    path_length=parent.bundle.path_length, path=parent.bundle.path, round_number=round_number)
                combined_option = PlanOption(combined, parent.layouts, parent.layout_key,
                    self.replay_bundle_on_state(parent.state, combined), parent_score, parent.state.cost)
                combined.debug_id = self.bundle_debug_id(combined_option.state)
                if len(combined_option.state.pods) > len(parent.state.pods):
                    options.append((combined_option, self.option_metrics(combined_option)))
            result.extend(option for option, _ in options)
            affordable = [(metrics[2], metrics[0], -metrics[1], option) for option, metrics in options
                if option.state.cost <= self.resources]
            if not affordable:
                break
            efficiency, _, _, next_parent = max(affordable, key=lambda item: item[:3])
            next_parent.bundle.debug_chosen = next_parent.bundle.debug_id
            if efficiency < parent_efficiency:
                break
            if len(next_parent.state.pods) > len(parent.state.pods):
                allow_reroute = False
            parent, parent_efficiency = next_parent, efficiency
            round_number += 1
        return result
    def option_metrics(self, option: PlanOption) -> tuple[int, int, float]:
        cost = option.state.cost - option.round_cost
        if option.state.cost > self.resources:
            return 0, cost, -inf
        gain = self.score_state(option.state).score - option.round_score
        return gain, cost, score_efficiency(gain, cost)
    def base_options(self, base: Bundle, layouts: tuple[Bundle, ...], inherited_state: PlanState,
            checkpoint_score: int, checkpoint_cost: int) -> list[tuple[PlanOption, float]]:
        next_layouts = layouts
        if base.tubes or base.teleport != (-1, -1) or base.prune_unused:
            layout = Bundle(base.pool, tubes=base.tubes, teleport=base.teleport, label=base.label, path_edges=base.path_edges,
                destination=base.destination, path_length=base.path_length, path=base.path, prune_unused=base.prune_unused)
            next_layouts = (*layouts, layout)
        layout_state = self.replay_bundle_sequence(next_layouts)
        key = self.layout_key(layout_state)
        if key in self.layout_cache:
            state = self.copy_state(self.layout_cache[key].state)
        else:
            state = self.inherit_operations(layout_state, inherited_state, base.pool)
            self.layout_cache[key] = LayoutBranch(self.copy_state(state))
        self.cancel_features(state, False)
        bundle = replace(base)
        bundle.debug_id = self.bundle_debug_id(state)
        option = PlanOption(bundle, next_layouts, key, state, checkpoint_score, checkpoint_cost)
        return [(option, self.option_metrics(option)[2])]
    def inherit_operations(self, state: PlanState, inherited_state: PlanState, owner: PoolOwner) -> PlanState:
        features = []
        for feature in inherited_state.features:
            kind, key, _, _, _ = feature
            if kind == "upgrade":
                if key not in state.tubes:
                    continue
                self.apply_bundle(state, Bundle(owner, upgrades=(key,)))
            else:
                self.add_dynamic_pod(state, key, kind == "reroute", False)
            features.append(feature)
        state.features = features
        return state
    def bundle_debug_id(self, state: PlanState) -> str:
        upgrades = sum(capacity - self.tubes.get(edge, 1) for edge, capacity in state.tubes.items())
        reroutes = sum(pod_id in self.pods for pod_id in state.ops)
        return f"{len(state.ops) - reroutes}p{upgrades}u{reroutes}r"
    def layout_key(self, state: PlanState) -> LayoutKey:
        return tuple(sorted(state.tubes)), tuple(sorted(state.teleports.items()))
    def shortest_route_bundles(self, owner: PoolOwner, group: Pool, module_ids: list[int], hop_limit: int,
            state: PlanState) -> list[Bundle]:
        bundles = []
        paths = self.cheapest_paths_by_hop(group[0], module_ids, hop_limit, state)
        for hop_count in range(hop_limit, 0, -1):
            bundles.extend(self.path_bundles(owner, f"short-{hop_count}", paths.get(hop_count, []), state))
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
        max_edges = len(self.buildings) - 1
        remaining_hops = max_edges - len(base_edges)
        best = None
        for junction_id in path:
            connector = self.cheapest_path_with_hop_limit(junction_id, list(network_nodes), remaining_hops, state)
            if not connector:
                continue
            edges = tuple(dict.fromkeys((*base_edges, *(route_key(a, b) for a, b in zip(connector, connector[1:])))))
            tubes = [edge for edge in edges if edge not in state.tubes]
            if len(edges) > max_edges or not self.can_add_tubes(tubes, state.tubes):
                continue
            cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes)
            order = cost, len(edges), edges
            if best is None or order < best[0]:
                best = order, edges
        return best[1] if best else ()
    def closest_fixed_pod(self, origin_id: int, state: PlanState) -> int:
        graph = tube_graph(state.tubes)
        options = []
        for pod_id, pod in state.pods.items():
            if pod_id in state.ops:
                continue
            index = 0
            distance = 0
            for _ in range(MONTH_DAYS):
                distance += graph_distance(graph, pod.path[index], origin_id)
                index = fixed_next_index(pod.path, index)
            options.append((distance, pod_id))
        return min(options)[1] if options else None
    def best_counter_edge(self, path_edges: tuple[Pair, ...], counts: Counter[Pair]) -> Pair:
        candidates = [(counts[edge], edge) for edge in path_edges if counts[edge]]
        return max(candidates, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1] if candidates else (-1, -1)
    def replay_bundle_on_state(self, state: PlanState, bundle: Bundle) -> PlanState:
        copied = self.copy_state(state)
        self.apply_bundle(copied, bundle)
        self.cancel_features(copied, True)
        return copied
    def copy_state(self, state: PlanState) -> PlanState:
        pods = {pod_id: PodPlan(pod.path[:], pod.dynamic) for pod_id, pod in state.pods.items()}
        return PlanState(dict(state.tubes), dict(state.teleports), pods, list(state.actions), list(state.pod_slots),
            set(state.ops), set(state.new_tubes), state.cost, list(state.features))
    def cancel_features(self, state: PlanState, upgrades_only: bool):
        """Cancels prior features in state until affordable; upgrades_only restricts eligible kinds."""
        eligible = [feature for feature in state.features if not feature[4] and (not upgrades_only or feature[0] == "upgrade")]
        for feature in sorted(eligible, key=lambda item: (item[3], -item[2], item[0], str(item[1]))):
            if state.cost <= self.resources:
                break
            self.cancel_feature(state, feature)
    def cancel_feature(self, state: PlanState, feature: Feature):
        """Cancels feature from state and refunds its effective current cost."""
        kind, key, _, _, _ = feature
        if kind == "upgrade":
            edge = key
            capacity = state.tubes[edge]
            state.cost -= tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * capacity
            state.tubes[edge] -= 1
            action = f"UPGRADE {edge[0]} {edge[1]}"
            index = max(index for index, current in enumerate(state.actions) if current == action)
            state.actions[index] = ""
        else:
            pod_id = key
            slot = max(index for index, (_, current_id) in enumerate(state.pod_slots) if current_id == pod_id)
            action_index, _ = state.pod_slots.pop(slot)
            state.actions[action_index] = ""
            state.ops.remove(pod_id)
            if kind == "pod":
                state.cost -= POD_COST
                del state.pods[pod_id]
            else:
                state.cost -= REROUTE_COST
                state.pods[pod_id] = PodPlan(self.pods[pod_id].path[:])
                action = f"DESTROY {pod_id}"
                index = max(index for index, current in enumerate(state.actions) if current == action)
                state.actions[index] = ""
        state.features.remove(feature)
    def replay_bundle_sequence(self, selected: tuple[Bundle, ...]) -> PlanState:
        pods = {pod_id: PodPlan(pod.path[:]) for pod_id, pod in self.pods.items()}
        state = PlanState(dict(self.tubes), dict(self.teleports), pods)
        for bundle in selected:
            self.apply_bundle(state, bundle)
            if bundle.prune_unused:
                self.prune_uncommitted_infrastructure(state)
        return state
    def prune_uncommitted_infrastructure(self, state: PlanState):
        used = set()
        if state.new_tubes:
            distances, module_distances = self.distances_to_targets(state)
            demands = self.path_demands(state, distances, module_distances)
            for demand in demands:
                used.update(route_key(a, b) for a, b in zip(demand.nodes, demand.nodes[1:]))
            origins = sorted({demand.nodes[0] for demand in demands})
            graph = tube_graph(state.tubes)
            for index, source_id in enumerate(origins):
                for target_id in origins[index + 1:]:
                    if graph_distance(graph, source_id, target_id) >= INF:
                        continue
                    current_id = source_id
                    while current_id != target_id:
                        next_id = next_step(graph, current_id, target_id)
                        used.add(route_key(current_id, next_id))
                        current_id = next_id
        for edge in sorted(state.new_tubes - used):
            remaining = dict(state.tubes)
            del remaining[edge]
            graph = tube_graph(remaining)
            if edge[0] not in graph or edge[1] not in graph or graph_distance(graph, *edge) < INF:
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
            cost = tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * state.tubes[edge]
            state.cost += cost
            state.actions.append(f"UPGRADE {edge[0]} {edge[1]}")
            state.features.append(("upgrade", edge, cost, 0, True))
        for pod_id in bundle.pod_specs:
            if pod_id and pod_id in state.ops:
                continue
            if pod_id:
                if pod_id not in state.pods:
                    raise ValueError("missing reroute pod")
            else:
                pod_id = self.next_pod_id(state.pods)
            self.add_dynamic_pod(state, pod_id, pod_id in self.pods)
    def add_dynamic_pod(self, state: PlanState, pod_id: int, reroute: bool, track: bool = True):
        """Adds pod_id to dynamic state; reroute controls replacement cost and track records cancellation provenance."""
        cost = REROUTE_COST if reroute else POD_COST
        state.cost += cost
        if reroute:
            state.actions.append(f"DESTROY {pod_id}")
        state.ops.add(pod_id)
        state.pod_slots.append((len(state.actions), pod_id))
        state.actions.append("")
        state.pods[pod_id] = PodPlan([], True)
        if track:
            state.features.append(("reroute" if reroute else "pod", pod_id, cost, 0, True))
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
            fixed_result = replace(fixed_result, congestion_by_edge=dynamic_result.congestion_by_edge,
                congestion_by_day=dynamic_result.congestion_by_day, dynamic_paths=paths, table=dynamic_result.table,
                initial_table=dynamic_result.initial_table, reserved=dynamic_result.reserved)
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
            self.fs_cache[network_key] = distances, module_distances, graph, self.wanted_edges(distances, graph), \
                self.initial_queues(distances), {}, {}
        distances, module_distances, graph, wanted_edges, initial, route_cache, choice_cache = self.fs_cache[network_key]
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
        result = SimulationResult()
        fixed_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if not pod.dynamic]
        dynamic_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if pod.dynamic]
        fixed_usage = Counter(edge for _, pod in fixed_pods for edge in {route_key(a, b) for a, b in zip(pod.path, pod.path[1:])})
        shared_capacities = tuple((edge, state.tubes[edge]) for edge, count in sorted(fixed_usage.items()) if count > 1)
        key = network_key, shared_capacities, tuple((pod_id, tuple(pod.path)) for pod_id, pod in fixed_pods)
        if dynamic_pods and key not in self.fs_cache:
            self.fs_cache[key] = self.fixed_assignment_schedule(state, distances, module_distances, wanted_edges, fixed_pods,
                route_cache, choice_cache)
        fixed_reservations, summary = self.fs_cache[key] if dynamic_pods else ([], Counter())
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
            reserved_passengers = fixed_reservations[day] if dynamic_pods else set()
            active, passenger_priorities = self.daily_loads(state, queues, module_distances, result, reserved_passengers,
                route_cache, choice_cache)
            if not active:
                break
            if not dynamic_pods:
                requests = self.path_pod_requests(fixed_pods, [], pod_positions, {}, {}, {}, {}, graph)
                moves = self.allocate_tube_capacity(requests, state, result, day)
                self.board_and_launch(queues, distances, state, moves, pod_positions, {}, {})
                self.settle(day + 1, queues, arrivals, result)
                continue
            fixed_assignments, _, _ = self.fixed_load_assignments(fixed_pods, pod_positions, active, queues, wanted_edges)
            occupied_edges = Counter()
            for pod_id, pod in fixed_pods:
                index = pod_positions[pod_id]
                next_index = fixed_next_index(pod.path, index)
                if pod_id in fixed_assignments and next_index != index:
                    occupied_edges[route_key(pod.path[index], pod.path[next_index])] += 1
            assignments, preferences = self.dispatch_dynamic_paths(active, dynamic_pods, dynamic_current, queues,
                wanted_edges, passenger_priorities, reserved_passengers, result.delivered_by_module, state, graph, occupied_edges)
            if FULL_DEBUG:
                initial_assignments = dict(assignments)
                initial_requests = self.path_pod_requests(fixed_pods, dynamic_pods, pod_positions, dynamic_current, dynamic_pending,
                    initial_assignments, fixed_assignments, graph)
            assignments, requests, moves = self.resolve_dispatch_congestion(assignments, preferences, fixed_pods, dynamic_pods,
                fixed_assignments, pod_positions, dynamic_current, dynamic_pending, graph, result, state, day)
            if FULL_DEBUG:
                loads = ", ".join("({})x{}{}".format("-".join(map(str, path.nodes)), path.cap,
                    "H" if path.priority > 0 else "L" if path.priority < 0 else "N") for path in active)
                for table, day_assignments, day_requests in ((result.initial_table, initial_assignments, initial_requests),
                        (result.table, assignments, requests)):
                    cells = []
                    for pod_id, pod in sorted(state.pods.items()):
                        paths = fixed_assignments.get(pod_id, set()) if not pod.dynamic else \
                            {day_assignments[pod_id]} if pod_id in day_assignments else set()
                        path_text = "/".join("-".join(map(str, path.nodes)) for path in paths) or "-"
                        location = pod.path[pod_positions[pod_id]] if not pod.dynamic else dynamic_current[pod_id]
                        location = day_requests[pod_id][0] if location == -1 and pod_id in day_requests else location
                        cells.append("{} ({})".format(path_text, location if location != -1 else "-"))
                    table.append([str(day + 1), loads, *cells])
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
            module_distances: dict[int, dict[int, int]], wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]],
            fixed_pods: list[tuple[int, PodPlan]], route_cache: dict[tuple[int, int], PathKey],
            choice_cache: dict[tuple[int, int], tuple[int, ...]]) -> tuple:
        if not fixed_pods:
            return [set() for _ in range(MONTH_DAYS)], Counter()
        initial = self.initial_queues(distances)
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
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
            active, _ = self.daily_loads(state, queues, module_distances, result, set(), route_cache, choice_cache)
            _, reservations, claimed = self.fixed_load_assignments(fixed_pods, positions, active, queues, wanted_edges)
            schedule.append((reservations, claimed))
            if not active:
                schedule.extend((set(), {}) for _ in range(day + 1, MONTH_DAYS))
                break
            requests = {}
            for pod_id, pod in fixed_pods:
                next_index = fixed_next_index(pod.path, positions[pod_id])
                if next_index != positions[pod_id]:
                    requests[pod_id] = pod.path[positions[pod_id]], pod.path[next_index]
            moves = self.allocate_tube_capacity(requests, state, result, day, False)
            self.board_and_launch(queues, distances, state, moves, positions, {}, {})
            self.settle(day + 1, queues, arrivals, result)
        delivered = {passenger.id for passengers in initial.values() for passenger in passengers} \
            - {passenger.id for passengers in queues.values() for passenger in passengers}
        future = set()
        filtered = [set() for _ in range(MONTH_DAYS)]
        for day in range(MONTH_DAYS - 1, -1, -1):
            reservations, claimed = schedule[day]
            future.update(reservations & delivered)
            filtered[day] = set(future)
            for path, passenger_ids in claimed.items():
                summary[path.nodes] += len(passenger_ids & delivered)
        return filtered, summary
    def fixed_load_assignments(self, fixed_pods: list[tuple[int, PodPlan]], pod_positions: dict[int, int],
            active: list[PathDemand], queues: dict[int, list[Passenger]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]]) -> tuple:
        moves = {}
        for pod_id, pod in fixed_pods:
            next_index = fixed_next_index(pod.path, pod_positions[pod_id])
            if next_index != pod_positions[pod_id]:
                moves[pod_id] = pod.path[pod_positions[pod_id]], pod.path[next_index]
        paths = {}
        for path in active:
            paths.setdefault((path.pool, path.nodes[:2]), []).append(path)
        assignments = {}
        claimed = {}
        reservations = set()
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
                reservations.add(passenger.id)
                edge = source_id, moves[pod_id][1]
                options = [path for path in paths.get((pool, edge), []) if len(claimed.get(path, ())) < path.cap]
                if options:
                    path = min(options, key=lambda item: (-item.priority, item.destination, item.nodes))
                    claimed.setdefault(path, set()).add(passenger.id)
                    assignments.setdefault(pod_id, set()).add(path)
        return assignments, reservations, claimed
    def daily_loads(self, state: PlanState, queues: dict[int, list[Passenger]], module_distances: dict[int, dict[int, int]],
            result: SimulationResult, reserved: set[int], route_cache: dict[tuple[int, int], PathKey],
            choice_cache: dict[tuple[int, int], tuple[int, ...]]) \
            -> tuple[list[PathDemand], dict[tuple[Pool, int, int], int]]:
        counts = Counter(((passenger.pad_id, passenger.kind), node_id)
            for node_id, passengers in queues.items() for passenger in passengers)
        reserved_counts = Counter(((passenger.pad_id, passenger.kind), node_id)
            for node_id, passengers in queues.items() for passenger in passengers if passenger.id in reserved)
        inbound = Counter(result.delivered_by_module)
        choices = {}
        ambiguous = []
        for (pool, node_id), count in sorted(counts.items()):
            key = pool[1], node_id
            if key not in choice_cache:
                modules = [building.id for building in self.buildings.values() if building.kind == pool[1]]
                distance = min(module_distances[module_id][node_id] for module_id in modules)
                choice_cache[key] = tuple(sorted(module_id for module_id in modules if module_distances[module_id][node_id] == distance))
            options = choice_cache[key]
            choices[pool, node_id] = options
            if len(options) == 1:
                inbound[options[0]] += count
            else:
                ambiguous.append((pool, node_id, count, options))
        planned = Counter()
        for pool, node_id, count, options in ambiguous:
            for _ in range(count):
                module_id = min(options, key=lambda item: (inbound[item], item))
                planned[pool, node_id, module_id] += 1
                inbound[module_id] += 1
        loads = []
        own_priorities = {}
        for (pool, node_id), count in sorted(counts.items()):
            options = choices[pool, node_id]
            for destination in options:
                key = node_id, destination
                if key not in route_cache:
                    path = self.concrete_path(node_id, destination, state)
                    route_cache[key] = next((item for item in self.tube_path_runs(path, state) if item[0] == node_id), ())
                run = route_cache[key]
                if not run:
                    continue
                for is_reserved, load_count in ((False, count - reserved_counts[pool, node_id]), (True, reserved_counts[pool, node_id])):
                    if not load_count:
                        continue
                    priority = -1 if is_reserved else 0 if len(options) == 1 else 1 if planned[pool, node_id, destination] else -1
                    loads.append(PathDemand(pool, destination, run, load_count, priority, is_reserved))
                    key = pool, run[0], run[1]
                    own_priorities[key] = max(own_priorities.get(key, -1), priority)
        direction_priorities = {}
        for (_, source_id, target_id), priority in own_priorities.items():
            direction_priorities[source_id, target_id] = max(direction_priorities.get((source_id, target_id), -1), priority)
        prioritized = [path if path.reserved or path.priority == direction_priorities[path.nodes[:2]] else
            replace(path, priority=direction_priorities[path.nodes[:2]]) for path in loads]
        return prioritized, own_priorities
    def path_demands(self, state: PlanState, distances: dict[int, dict[int, int]],
            module_distances: dict[int, dict[int, int]]) -> list[PathDemand]:
        demands = []
        for pool in self.speed_pools():
            options = [module.id for module in self.buildings.values() if module.kind == pool[1]
                and module_distances[module.id][pool[0]] == distances[pool[1]][pool[0]]]
            count = self.buildings[pool[0]].demand[pool[1]]
            for module_id in options:
                path = self.concrete_path(pool[0], module_id, state)
                for run in self.tube_path_runs(path, state):
                    demands.append(PathDemand(pool, module_id, run, count))
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
    def tube_path_runs(self, path: PathKey, state: PlanState) -> list[PathKey]:
        runs = []
        run = []
        for a, b in zip(path, path[1:]):
            if state.teleports.get(a) == b:
                if run:
                    runs.append(tuple(run))
                    run = []
            elif route_key(a, b) in state.tubes:
                if not run:
                    run.append(a)
                run.append(b)
            elif run:
                runs.append(tuple(run))
                run = []
        if run:
            runs.append(tuple(run))
        return runs
    def dispatch_dynamic_paths(self, active: list[PathDemand], dynamic_pods: list[tuple[int, PodPlan]], current: dict[int, int],
            queues: dict[int, list[Passenger]], wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]],
            passenger_priorities: dict[tuple[Pool, int, int], int], reserved_passengers: set[int], delivered: Counter[int],
            state: PlanState, graph: dict[int, list[int]], occupied_edges: Counter[Pair]) \
            -> tuple[dict[int, PathDemand], dict[int, list[PathDemand]]]:
        assignments = {}
        preferences = {}
        fixed_reserved = set(reserved_passengers)
        inspected = set(reserved_passengers)
        for pod_id, _ in dynamic_pods:
            options = [path for path in active if current[pod_id] == -1 or graph_distance(graph, current[pod_id], path.nodes[0]) < INF]
            batches = {}
            inspection_batches = {}
            for path in options:
                if path.nodes[:2] not in batches:
                    batches[path.nodes[:2]] = self.boarding_batch(path.nodes[:2], queues, wanted_edges, fixed_reserved)
                    inspection_batches[path.nodes[:2]] = self.boarding_batch(path.nodes[:2], queues, wanted_edges, inspected)
            evaluated = []
            for path in options:
                priority = path.priority
                if priority > 0:
                    balance = sum(passenger_priorities.get(((passenger.pad_id, passenger.kind), *path.nodes[:2]), 0)
                        for passenger in inspection_batches[path.nodes[:2]])
                    if balance < 0:
                        priority = -1
                evaluated.append(path if priority == path.priority else replace(path, priority=priority))
            preferences[pod_id] = sorted(evaluated,
                key=lambda path: self.path_assignment_key(path, pod_id, len(batches[path.nodes[:2]]), current, delivered, graph))
            if preferences[pod_id]:
                assignments[pod_id] = preferences[pod_id][0]
                inspected.update(passenger.id for passenger in inspection_batches[assignments[pod_id].nodes[:2]])
        self.fix_load_assignments(assignments, preferences, current, graph, state, occupied_edges)
        return assignments, preferences
    def boarding_batch(self, edge: DirectedPair, queues: dict[int, list[Passenger]],
            wanted_edges: dict[tuple[int, int], tuple[DirectedPair, ...]], reserved: set[int]) -> list[Passenger]:
        source_id, _ = edge
        return [passenger for passenger in queues.get(source_id, [])
            if passenger.id not in reserved and edge in wanted_edges[source_id, passenger.kind]][:POD_CAPACITY]
    def resolve_dispatch_congestion(self, assignments: dict[int, PathDemand], preferences: dict[int, list[PathDemand]],
            fixed_pods: list[tuple[int, PodPlan]], dynamic_pods: list[tuple[int, PodPlan]],
            fixed_assignments: dict[int, set[PathDemand]], pod_positions: dict[int, int], current: dict[int, int],
            pending: dict[int, DirectedPair], graph: dict[int, list[int]], result: SimulationResult, state: PlanState, day: int) -> tuple:
        def requests_for(values: dict[int, PathDemand]) -> dict[int, DirectedPair]:
            return self.path_pod_requests(fixed_pods, dynamic_pods, pod_positions, current, pending, values, fixed_assignments, graph)
        def conflicts(values: dict[int, DirectedPair]) -> Counter[Pair]:
            counts = Counter(route_key(*move) for move in values.values())
            return Counter({edge: count - state.tubes[edge] for edge, count in counts.items() if count > state.tubes[edge]})
        def improves(trial: Counter[Pair], original: Counter[Pair], edge: Pair) -> bool:
            return trial[edge] < original[edge]
        requests = requests_for(assignments)
        congestion = conflicts(requests)
        fixed_ids = {pod_id for pod_id, _ in fixed_pods}
        counted = {edge for edge in congestion if any(route_key(*move) == edge and (pod_id in assignments or pod_id in fixed_ids)
            for pod_id, move in requests.items())}
        result.congestion_by_edge.update(counted)
        if counted:
            result.congestion_by_day.setdefault(day, Counter()).update(counted)
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
            if not changed:
                for pod_id in pods:
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
                        break
            if changed:
                unresolved.intersection_update(congestion)
            else:
                unresolved.add(edge)
        return assignments, requests, self.allocate_tube_capacity(requests, state, result, day, False)
    def path_assignment_key(self, path: PathDemand, pod_id: int, boarding: int, current: dict[int, int],
            delivered: Counter[int], graph: dict[int, list[int]]) -> tuple:
        distance = 0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])
        return -path.priority, -boarding, distance, len(path.nodes) - 1, delivered[path.destination], \
            -path.cap, path.pool, path.destination, path.nodes
    def fix_load_assignments(self, assignments: dict[int, PathDemand], preferences: dict[int, list[PathDemand]],
            current: dict[int, int], graph: dict[int, list[int]], state: PlanState, occupied_edges: Counter[Pair]):
        def passenger_capacity(path: PathDemand) -> int:
            return (path.cap + POD_CAPACITY - 1) // POD_CAPACITY
        def overflow(values: Counter[PathDemand]) -> PathDemand:
            def assign(path: PathDemand, seen: set[tuple[Pair, int]]) -> bool:
                for edge in path_edges[path]:
                    for slot in range(occupied_edges[edge], state.tubes[edge]):
                        key = edge, slot
                        if key in seen:
                            continue
                        seen.add(key)
                        if key not in matches or assign(matches[key], seen):
                            matches[key] = path
                            return True
                return False
            matches = {}
            for path in sorted(paths, key=lambda item: (-item.priority, item.pool, item.destination, item.nodes)):
                if values[path] > passenger_capacity(path):
                    return path
                for _ in range(values[path]):
                    if not assign(path, set()):
                        return path
            return None
        def can_add(path: PathDemand) -> bool:
            trial = counts.copy()
            trial[path] += 1
            return overflow(trial) is None
        indices = {pod_id: 0 for pod_id in assignments}
        counts = Counter(assignments.values())
        owners = {}
        for pod_id, path in assignments.items():
            owners.setdefault(path, set()).add(pod_id)
        paths = sorted({path for pod_paths in preferences.values() for path in pod_paths},
            key=lambda item: (item.pool, item.destination, item.nodes))
        path_edges = {path: tuple(route_key(a, b) for a, b in zip(path.nodes, path.nodes[1:])) for path in paths}
        distances = {(pod_id, path.nodes[0]): 0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])
            for pod_id in assignments for path in paths}
        while True:
            exceeded = overflow(counts)
            for path in paths:
                pods = [pod_id for pod_id in owners.get(path, ())]
                if not pods:
                    continue
                uneven = exceeded is None and any(candidate.priority == path.priority and counts[candidate] < counts[path] - 1
                    and can_add(candidate) for candidate in paths)
                if path != exceeded and not uneven:
                    continue
                pod_id = max(pods, key=lambda item: (distances[item, path.nodes[0]],
                    -distances[item, preferences[item][indices[item] + 1].nodes[0]] if indices[item] + 1 < len(preferences[item]) else -INF, item))
                owners[path].remove(pod_id)
                counts[path] -= 1
                indices[pod_id] += 1
                if indices[pod_id] == len(preferences[pod_id]):
                    del assignments[pod_id]
                else:
                    assignments[pod_id] = preferences[pod_id][indices[pod_id]]
                    owners.setdefault(assignments[pod_id], set()).add(pod_id)
                    counts[assignments[pod_id]] += 1
                break
            else:
                return
    def path_pod_requests(self, fixed_pods: list[tuple[int, PodPlan]], dynamic_pods: list[tuple[int, PodPlan]],
            pod_positions: dict[int, int], current: dict[int, int], pending: dict[int, DirectedPair], assignments: dict[int, PathDemand],
            fixed_assignments: dict[int, set[PathDemand]], graph: dict[int, list[int]]) -> dict[int, DirectedPair]:
        requests = {}
        for pod_id, pod in fixed_pods:
            index = pod_positions[pod_id]
            next_index = fixed_next_index(pod.path, index)
            if next_index != index:
                requests[pod_id] = pod.path[index], pod.path[next_index]
        options = {}
        assigned_paths = [*assignments.values(), *(path for paths in fixed_assignments.values() for path in paths)]
        assigned_edges = {route_key(a, b) for path in assigned_paths for a, b in zip(path.nodes, path.nodes[1:])}
        all_edges = [(a, b) for a in sorted(graph) for b in graph[a] if a < b]
        for pod_id, _ in dynamic_pods:
            if pending[pod_id] != (-1, -1):
                options[pod_id] = [pending[pod_id]]
                continue
            if pod_id not in assignments:
                moves = all_edges if current[pod_id] == -1 else [(current[pod_id], neighbor_id) for neighbor_id in graph[current[pod_id]]]
                options[pod_id] = [min(moves, key=lambda move: (route_key(*move) in assigned_edges, move))]
                continue
            path = assignments[pod_id].nodes
            if current[pod_id] == -1:
                options[pod_id] = [path[:2]]
                continue
            target_id = path[1] if current[pod_id] == path[0] else next_step(graph, current[pod_id], path[0])
            options[pod_id] = [(current[pod_id], target_id)]
        requests.update((pod_id, moves[0]) for pod_id, moves in options.items())
        return requests
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
        speed = max(0, 50 - day)
        for building_id in sorted(queues):
            building = self.buildings[building_id]
            if building.kind <= 0:
                continue
            remaining = []
            for passenger in queues[building_id]:
                if passenger.kind != building.kind:
                    remaining.append(passenger)
                    continue
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
    def allocate_tube_capacity(self, requests: dict[int, DirectedPair], state: PlanState, result: SimulationResult, day: int,
            count_congestion: bool = True) -> dict[int, DirectedPair]:
        moves = {}
        by_tube = {}
        for pod_id, move in requests.items():
            by_tube.setdefault(route_key(*move), []).append((pod_id, move))
        for edge, pods in by_tube.items():
            capacity = state.tubes[edge]
            selected = sorted(pods)[:capacity]
            if count_congestion and len(pods) > capacity:
                result.congestion_by_edge[edge] += 1
                result.congestion_by_day.setdefault(day, Counter())[edge] += 1
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
    def cheapest_hop_path(self, start_id: int, targets: list[int], hop_count: int, state: PlanState) -> list[int]:
        return self.cheapest_paths_by_hop(start_id, targets, hop_count, state).get(hop_count, [])
    def cheapest_path_with_hop_limit(self, start_id: int, targets: list[int], hop_limit: int,
            state: PlanState) -> list[int]:
        paths = self.cheapest_paths_by_hop(start_id, targets, hop_limit, state)
        return min(paths.values(), key=lambda path: (sum(0 if route_key(a, b) in state.tubes else
            tube_cost(self.buildings[a], self.buildings[b]) for a, b in zip(path, path[1:])), path), default=[])
    def cheapest_paths_by_hop(self, start_id: int, targets: list[int], hop_limit: int,
            state: PlanState) -> dict[int, list[int]]:
        cache_key = tuple(sorted(state.tubes)), start_id, tuple(sorted(targets)), hop_limit
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        target_set = set(targets)
        edge_graph = self.build_candidate_edge_graph(state.tubes)
        bit = {building_id: 1 << index for index, building_id in enumerate(sorted(self.buildings))}
        states = {start_id: (0, (start_id,), bit[start_id])}
        result = {}
        for hops in range(1, hop_limit + 1):
            next_states = {}
            for building_id, (cost, path, mask) in states.items():
                if building_id in target_set:
                    continue
                for neighbor_id, edge_cost in edge_graph.get(building_id, []):
                    if mask & bit[neighbor_id]:
                        continue
                    candidate = cost + edge_cost, (*path, neighbor_id), mask | bit[neighbor_id]
                    if neighbor_id not in next_states or candidate[:2] < next_states[neighbor_id][:2]:
                        next_states[neighbor_id] = candidate
            states = next_states
            candidates = [(cost, tube_cost(self.buildings[start_id], self.buildings[target_id]), path)
                for target_id, (cost, path, _) in states.items() if target_id in target_set]
            if candidates:
                path = list(min(candidates)[2])
                if self.can_add_tubes(unique_new_tubes(path, state.tubes), state.tubes):
                    result[hops] = path
        self.path_cache[cache_key] = result
        return result
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
