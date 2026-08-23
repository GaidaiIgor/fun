from collections import Counter, deque
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from math import inf, isqrt
from operator import attrgetter
import sys
DAYS, MAX_DEGREE, MAX_PODS, POD_SIZE = 20, 5, 500, 10
POD_COST, POD_REFUND, REROUTE_COST, TELEPORT_COST = 1000, 750, 250, 5000
INF = 10 ** 9
OVERRIDE_MONTH = 10
OVERRIDE_COMMAND = "TUBE 2 7;TUBE 4 8;POD 1;POD 2;POD 3;POD 4;POD 5;POD 6"
FULL_DEBUG = False
_G = {}
BY_ID = attrgetter("id")
Pair = tuple[int, int]
Pool = tuple[int, int]
Owner = Pool | int
Path = tuple[int, ...]
Layout = tuple[tuple[Pair, ...], tuple[tuple[int, int], ...]]
Feature = tuple[str, Pair | int, int, float, bool]
def debug(text):
    if FULL_DEBUG:
        print(text, file=sys.stderr)
def score_efficiency(gain, cost):
    return gain / cost if cost > 0 else inf if gain > 0 else 0
@dataclass(slots=True)
class Node:
    id: int
    kind: int
    x: int
    y: int
    demand: Counter[int] = field(default_factory=Counter)
    order: list[int] = field(default_factory=list)
@dataclass(slots=True)
class Person:
    pad_id: int
    kind: int
    id: int
@dataclass(slots=True, frozen=True)
class Load:
    pool: Pool
    destination: int
    nodes: Path
    cap: int
    priority: int = field(default=0, compare=False)
    reserved: bool = False
    h: int = field(init=False, compare=False)
    def __post_init__(self):
        object.__setattr__(self, "h", hash((self.pool, self.destination, self.nodes, self.cap, self.reserved)))
    def __hash__(self):
        return self.h
def assignment_text(paths, pod_id, requests, moves,
        carrying):
    return "/".join(("   " if pod_id not in requests else "E! " if pod_id not in moves else
        "-> " if requests[pod_id] not in zip(path.nodes, path.nodes[1:]) else ".. " if pod_id in carrying else "   ") +
        "-".join(map(str, path.nodes)) for path in paths) or "   -"
@dataclass(slots=True)
class PodPlan:
    path: list[int] = field(default_factory=list)
    dynamic: bool = False
    assignments: list[Path] = field(default_factory=list)
@dataclass(slots=True)
class Bundle:
    pool: Owner
    tubes: tuple[Pair, ...] = ()
    teleport: Pair = (-1, -1)
    pod_specs: tuple[int, ...] = ()
    upgrades: tuple[Pair, ...] = ()
    label: str = "empty"
    path_edges: tuple[Pair, ...] = ()
    destination: int = -1
    hops: int = 0
    path: tuple[int, ...] = ()
    debug_id: str = ""
    chosen: str = ""
    rnd: int = 0
    prune: bool = False
@dataclass(slots=True)
class State:
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
class Option:
    bundle: Bundle
    layouts: tuple[Bundle, ...]
    layout_key: Layout
    state: State
    round_score: int
    round_cost: int
@dataclass(slots=True)
class Result:
    score: int = 0
    speed_by_pool: Counter[Pool] = field(default_factory=Counter)
    delivered_by_pool: Counter[Pool] = field(default_factory=Counter)
    delivery_times: dict[Pool, int] = field(default_factory=dict)
    diversity_by_module: Counter[int] = field(default_factory=Counter)
    delivered_by_module: Counter[int] = field(default_factory=Counter)
    delivered_by_pool_module: Counter[tuple[Pool, int]] = field(default_factory=Counter)
    congestion_by_edge: Counter[Pair] = field(default_factory=Counter)
    congestion_by_day: dict[int, Counter[Pair]] = field(default_factory=dict)
    carrying_days: Counter[int] = field(default_factory=Counter)
    dynamic_paths: dict[int, list[int]] = field(default_factory=dict)
    pod_assignments: dict[int, list[Path]] = field(default_factory=dict)
    initial_table: list[list[str]] = field(default_factory=list)
    capacity_table: list[list[str]] = field(default_factory=list)
    table: list[list[str]] = field(default_factory=list)
    reserved: str = "-"
@dataclass(slots=True)
class Candidate:
    bundle: Bundle
    pair: Owner
    marginal_gain: int
    marginal_cost: int
    round_gain: int
    round_cost: int
    layouts: tuple[Bundle, ...]
    layout_key: Layout
    state: State
    @property
    def efficiency(self):
        return score_efficiency(self.marginal_gain, self.marginal_cost)
class Planner:
    buildings: dict[int, Node]
    resources: int
    month: int
    tubes: dict[Pair, int]
    teleports: dict[int, int]
    pods: dict[int, PodPlan]
    simulation_cache: dict[tuple, Result]
    path_cache: dict[tuple, dict[int, list[int]]]
    layout_cache: dict[Layout, State]
    turn_start_result: Result
    def __init__(self):
        self.buildings = {}
        self.resources = 0
        self.month = 0
        self.tubes = {}
        self.teleports = {}
        self.pods = {}
        self.pod_assignments = {}
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
        assignments = self.pod_assignments
        self.pods = {}
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            pod_id = values[0]
            self.pods[pod_id] = PodPlan(values[2:], False, assignments[pod_id][:])
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            if values[0] == 0:
                self.buildings[values[1]] = Node(values[1], 0, values[2], values[3], Counter(values[5:]), values[5:])
            else:
                self.buildings[values[1]] = Node(values[1], values[0], values[2], values[3])
        self.print_debug_input()
    def choose_actions(self):
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
        self.layout_cache = {current_key: current_state}
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
            self.layout_cache = {current_key: current_state}
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
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths, final_result.pod_assignments)
        self.pod_assignments = {pod_id: pod.assignments[:] for pod_id, pod in final_state.pods.items()}
        if FULL_DEBUG:
            debug("\n" + self.table_debug(final_result, final_state))
            debug("\n" + self.score_debug("after", final_result, final_state.cost))
        action_order = {"TUBE": 0, "TELEPORT": 0, "UPGRADE": 1, "DESTROY": 2, "POD": 3}
        return sorted((action for action in final_state.actions if action), key=lambda action: action_order[action.split()[0]])
    def override_actions(self):
        current_state = self.replay_bundle_sequence(())
        current_result = self.score_state(current_state)
        if FULL_DEBUG:
            debug("\n" + self.score_debug("override", current_result, current_state.cost))
        final_state = self.override_state(OVERRIDE_COMMAND)
        final_result = self.score_state(final_state, True)
        self.fill_dynamic_actions(final_state, final_result.dynamic_paths, final_result.pod_assignments)
        self.pod_assignments = {pod_id: pod.assignments[:] for pod_id, pod in final_state.pods.items()}
        if FULL_DEBUG:
            debug("\n" + self.table_debug(final_result, final_state))
            debug(f"override month {self.month + 1}: {OVERRIDE_COMMAND}")
            debug("\n" + self.score_debug("after", final_result, final_state.cost))
        return [action for action in final_state.actions if action]
    def override_state(self, command):
        state = self.replay_bundle_sequence(())
        if command.strip() == "WAIT":
            return state
        for action in (item.strip() for item in command.split(";")):
            if action:
                self.apply_override_action(state, action)
        return state
    def apply_override_action(self, state, action):
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
    def apply_override_pod(self, state, action):
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
        state.pods[pod_id] = PodPlan(path, False, [()] * DAYS)
        state.actions.append(action)
    def best_candidate(self, layouts, current_state, current_result,
            before_score):
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
            for pair, group, module_ids, eligibility in pairs:
                debug(f"  Considering {pair}:")
                options = self.generate_options(owner, group, module_ids, layouts, current_state, current_result, eligibility)
                candidate = self.next_candidate(owner, pair, current_state, current_result, before_score, options)
                if candidate and (best is None or (candidate.efficiency, candidate.marginal_gain, -candidate.marginal_cost) >
                        (best.efficiency, best.marginal_gain, -best.marginal_cost)):
                    best = candidate
            if best:
                return best
        return None
    def layout_path_eligibility(self, group, module_id, state, result,
            distances, module_distances, diversity):
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
    def diversity_reroute_eligibility(self, group, module_id, result,
            module_distances):
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
    def next_candidate(self, owner, pair, current_state, current_result,
            before_score, options):
        best = None
        seen_states = set()
        plans = []
        for option in options:
            bundle, state = option.bundle, option.state
            if not bundle.tubes and bundle.teleport == (-1, -1) and not bundle.pod_specs and not bundle.upgrades and not bundle.path_edges:
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
        rnd = -1
        for option, action_text in plans:
            bundle, state = option.bundle, option.state
            next_branch = bundle.teleport != (-1, -1), bundle.path
            if next_branch != branch:
                branch = next_branch
                path_text = ", ".join(map(str, bundle.path))
                mode = "teleport" if next_branch[0] else "path"
                debug(f"    Considering {mode}=[{path_text}]:")
                rnd = -1
            if not next_branch[0] and bundle.rnd != rnd:
                rnd = bundle.rnd
                debug(f"      Round {rnd}:")
            prefix = "-> " if bundle.chosen else ""
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
        return best
    def generate_options(self, owner, group, module_ids, layouts, state,
            current_result, eligibility):
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
                if len(path) > 2:
                    bases.extend(self.shortest_route_bundles(owner, group, module_ids, len(path) - 2, state))
            else:
                bases.extend(self.shortest_route_bundles(owner, group, module_ids, hop_limit, state))
        for base in bases:
            base.prune = base.hops < current_length
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
                bundle.prune = True
                options.extend(option for option, _ in self.base_options(bundle, layouts, state, current_result.score, state.cost))
        return options
    def path_option_stack(self, owner, group, base, layouts,
            inherited_state, checkpoint_score, checkpoint_cost):
        base_options = self.base_options(base, layouts, inherited_state, checkpoint_score, checkpoint_cost)
        result = [option for option, _ in base_options]
        affordable = []
        for option, efficiency in base_options:
            if option.state.cost <= self.resources:
                gain, cost, _ = self.option_metrics(option)
                affordable.append((efficiency, gain, -cost, option))
        if not affordable:
            return result
        parent_eff, _, _, parent = max(affordable, key=lambda item: item[:3])
        parent.bundle.chosen = parent.bundle.debug_id
        result.extend(self.throughput_options(owner, group, parent, parent_eff))
        return result
    def throughput_options(self, owner, group, parent,
            parent_eff):
        result = []
        rnd = 1
        allow_reroute = True
        iteration_score = parent.round_score
        iteration_cost = parent.round_cost
        positive = self.score_state(parent.state).score > iteration_score
        iteration_efficiency = parent_eff
        while parent.state.cost <= self.resources:
            simulation = self.cached_simulate(parent.state)
            parent_score = self.score_state(parent.state).score
            options = []
            reroute_id = self.best_reroute_pod(group[0], parent.state) if allow_reroute else None
            if reroute_id is not None:
                reroute_bundle = Bundle(owner, pod_specs=(reroute_id,), label=f"{parent.bundle.label}-reroute",
                    path_edges=parent.bundle.path_edges, destination=parent.bundle.destination,
                    hops=parent.bundle.hops, path=parent.bundle.path, rnd=rnd)
                reroute_option = Option(reroute_bundle, parent.layouts, parent.layout_key,
                    self.replay_bundle_on_state(parent.state, reroute_bundle), parent_score, parent.state.cost)
                reroute_bundle.debug_id = self.bundle_debug_id(reroute_option.state)
                options.append((reroute_option, self.option_metrics(reroute_option)))
            pod_bundle = Bundle(owner, pod_specs=(0,), label=f"{parent.bundle.label}-pod", path_edges=parent.bundle.path_edges,
                destination=parent.bundle.destination, hops=parent.bundle.hops, path=parent.bundle.path, rnd=rnd)
            pod_option = Option(pod_bundle, parent.layouts, parent.layout_key,
                self.replay_bundle_on_state(parent.state, pod_bundle), parent_score, parent.state.cost)
            pod_bundle.debug_id = self.bundle_debug_id(pod_option.state)
            pod_metrics = self.option_metrics(pod_option)
            pod_added = len(pod_option.state.pods) > len(parent.state.pods)
            if pod_added and len(parent.state.pods) < sum(parent.state.tubes.values()):
                options.append((pod_option, pod_metrics))
            upgrade_edge = self.best_counter_edge(parent.bundle.path_edges, simulation.congestion_by_edge)
            if upgrade_edge != (-1, -1):
                upgrade_bundle = Bundle(owner, upgrades=(upgrade_edge,), label=f"{parent.bundle.label}-upgrade",
                    path_edges=parent.bundle.path_edges, destination=parent.bundle.destination, hops=parent.bundle.hops,
                    path=parent.bundle.path, rnd=rnd)
                upgrade_option = Option(upgrade_bundle, parent.layouts, parent.layout_key,
                    self.replay_bundle_on_state(parent.state, upgrade_bundle), parent_score, parent.state.cost)
                upgrade_bundle.debug_id = self.bundle_debug_id(upgrade_option.state)
                options.append((upgrade_option, self.option_metrics(upgrade_option)))
            result.extend(option for option, _ in options)
            affordable = [(metrics[2], metrics[0], -metrics[1], option) for option, metrics in options
                if option.state.cost <= self.resources]
            if not affordable:
                break
            if all(item[0] == 0 for item in affordable):
                upgrade = next((item for item in affordable if item[3].bundle.upgrades), None)
            else:
                upgrade = None
            efficiency, _, _, next_parent = upgrade or max(affordable, key=lambda item: item[:3])
            next_parent.bundle.chosen = next_parent.bundle.debug_id
            positive |= any(option.round_score + round_gain > iteration_score
                for _, round_gain, _, option in affordable)
            next_efficiency = score_efficiency(self.score_state(next_parent.state).score - iteration_score,
                next_parent.state.cost - iteration_cost)
            if positive and efficiency < parent_eff and next_efficiency < iteration_efficiency:
                break
            if len(next_parent.state.pods) > len(parent.state.pods):
                allow_reroute = False
            parent, parent_eff, iteration_efficiency = next_parent, efficiency, next_efficiency
            rnd += 1
        return result
    def option_metrics(self, option):
        cost = option.state.cost - option.round_cost
        if option.state.cost > self.resources:
            return 0, cost, -inf
        gain = self.score_state(option.state).score - option.round_score
        return gain, cost, score_efficiency(gain, cost)
    def base_options(self, base, layouts, inherited_state,
            checkpoint_score, checkpoint_cost):
        next_layouts = layouts
        if base.tubes or base.teleport != (-1, -1) or base.prune:
            layout = Bundle(base.pool, tubes=base.tubes, teleport=base.teleport, label=base.label, path_edges=base.path_edges,
                destination=base.destination, hops=base.hops, path=base.path, prune=base.prune)
            next_layouts = (*layouts, layout)
        layout_state = self.replay_bundle_sequence(next_layouts)
        key = self.layout_key(layout_state)
        if key in self.layout_cache:
            state = self.copy_state(self.layout_cache[key])
        else:
            state = self.inherit_operations(layout_state, inherited_state, base.pool)
            self.layout_cache[key] = self.copy_state(state)
        self.cancel_features(state, False)
        bundle = replace(base)
        bundle.debug_id = self.bundle_debug_id(state)
        option = Option(bundle, next_layouts, key, state, checkpoint_score, checkpoint_cost)
        return [(option, self.option_metrics(option)[2])]
    def inherit_operations(self, state, inherited_state, owner):
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
    def bundle_debug_id(self, state):
        upgrades = sum(capacity - self.tubes.get(edge, 1) for edge, capacity in state.tubes.items())
        reroutes = sum(pod_id in self.pods for pod_id in state.ops)
        return f"{len(state.ops) - reroutes}p{upgrades}u{reroutes}r"
    def layout_key(self, state):
        return tuple(sorted(state.tubes)), tuple(sorted(state.teleports.items()))
    def shortest_route_bundles(self, owner, group, module_ids, hop_limit,
            state):
        bundles = []
        paths = self.cheapest_paths_by_hop(group[0], module_ids, hop_limit, state)
        for hop_count in range(hop_limit, 0, -1):
            bundles.extend(self.path_bundles(owner, f"short-{hop_count}", paths.get(hop_count, []), state))
        return bundles
    def teleport_bundles(self, owner, group, modules, state):
        pad_id = group[0]
        used = self.teleport_used_buildings(state.teleports)
        if pad_id in used:
            return []
        bundles = [Bundle(owner, teleport=(pad_id, module_id), label=f"teleport-{module_id}", destination=module_id,
            path=(pad_id, module_id)) for module_id in sorted(modules, key=lambda item: tube_cost(self.buildings[pad_id], self.buildings[item]))
            if module_id not in used]
        return bundles
    def path_bundles(self, owner, label, path, state):
        if not path:
            return []
        path_edges = self.connected_path_edges(path, state)
        if not path_edges:
            return []
        tubes = tuple(edge for edge in path_edges if edge not in state.tubes)
        return [Bundle(owner, tubes=tubes, label=label, path_edges=path_edges,
            destination=path[-1], hops=len(path) - 1, path=tuple(path))]
    def connected_path_edges(self, path, state):
        base_edges = edges_of(path)
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
            edges = tuple(dict.fromkeys((*base_edges, *edges_of(connector))))
            tubes = [edge for edge in edges if edge not in state.tubes]
            if len(edges) > max_edges or not self.can_add_tubes(tubes, state.tubes):
                continue
            cost = sum(tube_cost(self.buildings[a], self.buildings[b]) for a, b in tubes)
            order = cost, len(edges), edges
            if best is None or order < best[0]:
                best = order, edges
        return best[1] if best else ()
    def best_reroute_pod(self, origin_id, state):
        graph = tube_graph(state.tubes)
        carrying_days = self.cached_simulate(state).carrying_days
        options = []
        for pod_id, pod in state.pods.items():
            if pod_id in state.ops:
                continue
            index = 0
            distance = 0
            for assignment in pod.assignments:
                distance += graph_distance(graph, pod.path[index], origin_id)
                index = fixed_next_index(pod.path, index)
            options.append((carrying_days[pod_id], distance, pod_id))
        return min(options)[2] if options else None
    def best_counter_edge(self, path_edges, counts):
        candidates = [(counts[edge], edge) for edge in path_edges if counts[edge]]
        return max(candidates, key=lambda item: (item[0], -item[1][0], -item[1][1]))[1] if candidates else (-1, -1)
    def replay_bundle_on_state(self, state, bundle):
        copied = self.copy_state(state)
        self.apply_bundle(copied, bundle)
        self.cancel_features(copied, True)
        return copied
    def copy_state(self, state):
        pods = {pod_id: PodPlan(pod.path[:], pod.dynamic, pod.assignments[:]) for pod_id, pod in state.pods.items()}
        return State(dict(state.tubes), dict(state.teleports), pods, list(state.actions), list(state.pod_slots),
            set(state.ops), set(state.new_tubes), state.cost, list(state.features))
    def cancel_features(self, state, upgrades_only):
        eligible = [feature for feature in state.features if not feature[4] and (not upgrades_only or feature[0] == "upgrade")]
        for feature in sorted(eligible, key=lambda item: (item[3], -item[2], item[0], str(item[1]))):
            if state.cost <= self.resources:
                break
            self.cancel_feature(state, feature)
    def cancel_feature(self, state, feature):
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
                pod = self.pods[pod_id]
                state.pods[pod_id] = PodPlan(pod.path[:], False, pod.assignments[:])
                action = f"DESTROY {pod_id}"
                index = max(index for index, current in enumerate(state.actions) if current == action)
                state.actions[index] = ""
        state.features.remove(feature)
    def replay_bundle_sequence(self, selected):
        pods = {pod_id: PodPlan(pod.path[:], False, pod.assignments[:]) for pod_id, pod in self.pods.items()}
        state = State(dict(self.tubes), dict(self.teleports), pods)
        for bundle in selected:
            self.apply_bundle(state, bundle)
            if bundle.prune:
                self.prune_uncommitted_infrastructure(state)
        return state
    def prune_uncommitted_infrastructure(self, state):
        used = set()
        if state.new_tubes:
            distances, module_distances = self.distances_to_targets(state)
            demands = self.path_demands(state, distances, module_distances)
            for demand in demands:
                used.update(edges_of(demand.nodes))
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
    def remove_planned_tube(self, state, edge):
        capacity = state.tubes[edge]
        state.cost -= tube_cost(self.buildings[edge[0]], self.buildings[edge[1]]) * capacity * (capacity + 1) // 2
        del state.tubes[edge]
        state.new_tubes.remove(edge)
        for index, action in enumerate(state.actions):
            parts = action.split()
            if parts and parts[0] in ("TUBE", "UPGRADE") and route_key(int(parts[1]), int(parts[2])) == edge:
                state.actions[index] = ""
    def apply_bundle(self, state, bundle):
        degrees = self.tube_degrees(state.tubes)
        for a, b in bundle.tubes:
            key = route_key(a, b)
            if key in state.tubes:
                continue
            if not self.can_build_tube(a, b, state.tubes):
                raise ValueError("invalid tube")
            if degrees[a] >= MAX_DEGREE or degrees[b] >= MAX_DEGREE:
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
    def add_dynamic_pod(self, state, pod_id, reroute, track = True):
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
    def fill_dynamic_actions(self, state, dynamic_paths,
            pod_assignments):
        for index, pod_id in state.pod_slots:
            path = dynamic_paths[pod_id]
            assert len(path) >= 2, f"dynamic pod {pod_id} produced an empty route"
            state.actions[index] = "POD {} {}".format(pod_id, " ".join(map(str, path)))
            state.pods[pod_id].path = path
        for pod_id, pod in state.pods.items():
            pod.assignments = pod_assignments[pod_id]
    def score_state(self, state, keep_dynamic_paths = False):
        if not any(pod.dynamic for pod in state.pods.values()):
            return self.simulate(state, True) if keep_dynamic_paths else self.cached_simulate(state)
        dynamic_result = self.cached_simulate(state)
        paths = dynamic_result.dynamic_paths
        assignments = dynamic_result.pod_assignments
        if not keep_dynamic_paths and not FULL_DEBUG:
            return dynamic_result
        fixed_result = self.cached_simulate(self.fixed_dynamic_state(state, paths, assignments))
        if keep_dynamic_paths:
            fixed_result = replace(fixed_result, congestion_by_edge=dynamic_result.congestion_by_edge,
                congestion_by_day=dynamic_result.congestion_by_day, dynamic_paths=paths, pod_assignments=assignments,
                table=dynamic_result.table, initial_table=dynamic_result.initial_table, capacity_table=dynamic_result.capacity_table,
                reserved=dynamic_result.reserved)
        return fixed_result
    def cached_simulate(self, state):
        keep_dynamic_paths = any(pod.dynamic for pod in state.pods.values())
        pods = tuple(sorted((pod_id, (), True, ()) if pod.dynamic else
            (pod_id, tuple(pod.path), False, tuple(pod.assignments) if keep_dynamic_paths else ()) for pod_id, pod in state.pods.items()))
        key = tuple(sorted(state.tubes.items())), tuple(sorted(state.teleports.items())), pods
        if key not in self.simulation_cache:
            self.simulation_cache[key] = self.simulate(state, keep_dynamic_paths)
        return self.simulation_cache[key]
    def fixed_dynamic_state(self, state, dynamic_paths,
            pod_assignments):
        pods = {pod_id: PodPlan(dynamic_paths[pod_id][:] if pod.dynamic else pod.path[:], False, pod_assignments[pod_id][:])
            for pod_id, pod in state.pods.items()}
        return State(state.tubes, state.teleports, pods)
    def simulate(self, state, keep_dynamic_paths = False):
        network_key = tuple(sorted(state.tubes)), tuple(sorted(state.teleports.items()))
        if network_key not in self.fs_cache:
            distances, module_distances = self.distances_to_targets(state)
            graph = tube_graph(state.tubes)
            self.fs_cache[network_key] = distances, module_distances, graph, self.wanted_edges(distances, graph), \
                self.initial_queues(distances), {}, {}
        distances, module_distances, graph, wanted_edges, initial, route_cache, choice_cache = self.fs_cache[network_key]
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
        result = Result()
        f_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if not pod.dynamic]
        d_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if pod.dynamic]
        fixed_usage = Counter(edge for _, pod in f_pods for edge in set(edges_of(pod.path)))
        shared_capacities = tuple((edge, state.tubes[edge]) for edge, count in sorted(fixed_usage.items()) if count > 1)
        key = network_key, shared_capacities, tuple((pod_id, tuple(pod.path), tuple(pod.assignments)) for pod_id, pod in f_pods)
        if d_pods and key not in self.fs_cache:
            self.fs_cache[key] = self.fixed_assignment_schedule(state, distances, module_distances, wanted_edges, f_pods,
                route_cache, choice_cache)
        fixed_reservations, summary = self.fs_cache[key] if d_pods else ([], Counter())
        if keep_dynamic_paths:
            result.reserved = ", ".join("({})x{}".format("-".join(map(str, path)), count)
                for path, count in summary.items()) or "-"
        positions = {pod_id: 0 for pod_id, _ in f_pods}
        at = {pod_id: -1 for pod_id, _ in d_pods}
        pending = {pod_id: (-1, -1) for pod_id, _ in d_pods}
        dynamic_paths = {pod_id: [] for pod_id, _ in d_pods}
        pod_assignments = {pod_id: [] for pod_id in state.pods}
        free = replace(state, tubes=dict.fromkeys(state.tubes, MAX_PODS))
        arrivals = Counter()
        for day in range(DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, arrivals, result)
            for passengers in queues.values():
                passengers.sort(key=BY_ID)
            if not d_pods and not FULL_DEBUG and not keep_dynamic_paths:
                if not any(wanted_edges[node_id, passenger.kind] for node_id, passengers in queues.items() for passenger in passengers):
                    break
                requests = self.path_pod_requests(f_pods, [], positions, {}, {}, {}, {}, graph)
                assigned = {pod_id for pod_id, pod in f_pods if day < len(pod.assignments) and pod.assignments[day]}
                moves = self.allocate_tube_capacity(requests, state, result, day, assigned)
                result.carrying_days.update(self.board_and_launch(queues, distances, state, moves, positions, {}, {}))
                self.settle(day + 1, queues, arrivals, result)
                continue
            reserved_passengers = fixed_reservations[day] if d_pods else set()
            active, passenger_priorities = self.daily_loads(state, queues, module_distances, result, reserved_passengers,
                route_cache, choice_cache)
            if not active:
                break
            fixed_jobs, _, _ = self.fixed_load_assignments(f_pods, positions, active, queues, wanted_edges, day)
            for pod_id, _ in f_pods:
                paths = fixed_jobs.get(pod_id)
                pod_assignments[pod_id].append(next(iter(paths)).nodes if paths else ())
            if not d_pods:
                requests = self.path_pod_requests(f_pods, [], positions, {}, {}, {}, {}, graph)
                moves = self.allocate_tube_capacity(requests, state, result, day, set(fixed_jobs))
                locations = {pod_id: pod.path[positions[pod_id]] for pod_id, pod in f_pods} if FULL_DEBUG else {}
                carrying = self.board_and_launch(queues, distances, state, moves, positions, {}, {})
                result.carrying_days.update(carrying)
                if FULL_DEBUG:
                    loads = ", ".join("({})x{}{}".format("-".join(map(str, path.nodes)), path.cap,
                        "L" if path.priority < 0 or path.cap < POD_SIZE else "H" if path.priority > 0 else "N") for path in active)
                    cells = []
                    for pod_id, pod in f_pods:
                        paths = fixed_jobs.get(pod_id, set())
                        path_text = assignment_text(paths, pod_id, requests, moves, carrying)
                        cells.append("{} ({})".format(path_text, locations[pod_id]))
                    row = [str(day + 1), loads, *cells]
                    result.initial_table.append(row[:])
                    result.capacity_table.append(row[:])
                    result.table.append(row)
                self.settle(day + 1, queues, arrivals, result)
                continue
            occupied = Counter()
            for pod_id, pod in f_pods:
                index = positions[pod_id]
                next_index = fixed_next_index(pod.path, index)
                if pod_id in fixed_jobs and next_index != index:
                    occupied[route_key(pod.path[index], pod.path[next_index])] += 1
            assignments, prefs = self.dispatch_dynamic_paths(active, d_pods, at, pending, queues,
                wanted_edges, passenger_priorities, reserved_passengers, result.delivered_by_module, graph, day)
            ideal = assignments.copy()
            self.fix_load_assignments(ideal, prefs, at, graph, free, Counter(), True)
            ideal_req = self.path_pod_requests(f_pods, d_pods, positions, at, pending, ideal, fixed_jobs, graph)
            if FULL_DEBUG:
                i_jobs = assignments.copy()
                i_req = self.path_pod_requests(f_pods, d_pods, positions, at, pending, assignments, fixed_jobs, graph)
                i_moves = self.allocate_tube_capacity(i_req, state, result, day, set())
                i_at = {pod_id: pod.path[positions[pod_id]] if not pod.dynamic else
                    at[pod_id] if at[pod_id] != -1 else i_req.get(pod_id, ("-",))[0]
                    for pod_id, pod in state.pods.items()}
                i_carry = self.board_and_launch({building_id: passengers[:] for building_id, passengers in queues.items()},
                    distances, state, i_moves, positions.copy(), at.copy(), pending.copy())
            self.fix_load_assignments(assignments, prefs, at, graph, state, occupied, True)
            capped = {pod_id for pod_id, path in ideal.items() if assignments.get(pod_id) != path}
            if FULL_DEBUG:
                c_jobs = ideal.copy()
                c_req = ideal_req
                c_moves = c_req
                c_at = {pod_id: pod.path[positions[pod_id]] if not pod.dynamic else
                    at[pod_id] if at[pod_id] != -1 else c_req.get(pod_id, ("-",))[0]
                    for pod_id, pod in state.pods.items()}
                c_carry = self.board_and_launch({building_id: passengers[:] for building_id, passengers in queues.items()},
                    distances, state, c_moves, positions.copy(), at.copy(), pending.copy())
            assignments, requests, moves = self.resolve_edge_conflicts(assignments, prefs, f_pods, d_pods,
                fixed_jobs, positions, at, pending, graph, occupied, result, state, day)
            blocked = {min(edges_of(ideal[p].nodes), key=state.tubes.__getitem__) for p in capped}
            for p in (set(ideal) | set(fixed_jobs)) - capped:
                r, m = ideal_req[p], moves.get(p)
                same = m == r
                if not same and p in ideal and m and at[p] != -1 and pending[p] == (-1, -1):
                    path = ideal[p].nodes
                    target = path[-1] if at[p] == path[0] else path[0]
                    same = graph_distance(graph, m[1], target) < graph_distance(graph, m[0], target)
                if not same:
                    blocked.add(route_key(*r))
            if blocked:
                result.congestion_by_edge.update(blocked)
                result.congestion_by_day[day] = Counter(blocked)
            day_assignments = assignments
            for pod_id, _ in d_pods:
                pod_assignments[pod_id].append(day_assignments[pod_id].nodes if pod_id in day_assignments else ())
            locations = {pod_id: pod.path[positions[pod_id]] if not pod.dynamic else
                at[pod_id] if at[pod_id] != -1 else requests.get(pod_id, ("-",))[0]
                for pod_id, pod in state.pods.items()} if FULL_DEBUG else {}
            for pod_id, _ in d_pods:
                if pending[pod_id] != (-1, -1) or pod_id not in requests:
                    continue
                pending[pod_id] = requests[pod_id]
                if at[pod_id] == -1:
                    at[pod_id] = requests[pod_id][0]
                    dynamic_paths[pod_id].append(requests[pod_id][0])
                dynamic_paths[pod_id].append(requests[pod_id][1])
            carrying = self.board_and_launch(queues, distances, state, moves, positions, at, pending)
            result.carrying_days.update(carrying)
            if FULL_DEBUG:
                loads = ", ".join("({})x{}{}".format("-".join(map(str, path.nodes)), path.cap,
                    "L" if path.priority < 0 or path.cap < POD_SIZE else "H" if path.priority > 0 else "N") for path in active)
                tables = ((result.initial_table, i_jobs, i_req, i_moves, i_carry, i_at),
                    (result.capacity_table, c_jobs, c_req, c_moves, c_carry, c_at),
                    (result.table, day_assignments, requests, moves, carrying, locations))
                for table, day_assignments, day_requests, day_moves, day_carrying, day_locations in tables:
                    cells = []
                    for pod_id, pod in sorted(state.pods.items()):
                        paths = fixed_jobs.get(pod_id, set()) if not pod.dynamic else \
                            {day_assignments[pod_id]} if pod_id in day_assignments else set()
                        cells.append("{} ({})".format(assignment_text(paths, pod_id, day_requests, day_moves, day_carrying),
                            day_locations[pod_id]))
                    table.append([str(day + 1), loads, *cells])
            self.settle(day + 1, queues, arrivals, result)
        result.pod_assignments = {pod_id: assignments + [()] * (DAYS - len(assignments))
            for pod_id, assignments in pod_assignments.items()}
        if keep_dynamic_paths:
            result.dynamic_paths = {pod_id: normalize_month_path(path) for pod_id, path in dynamic_paths.items()}
        return result
    def fixed_assignment_schedule(self, state, distances,
            module_distances, wanted_edges,
            f_pods, route_cache,
            choice_cache):
        if not f_pods:
            return [set() for _ in range(DAYS)], Counter()
        initial = self.initial_queues(distances)
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
        result = Result()
        positions = {pod_id: 0 for pod_id, _ in f_pods}
        arrivals = Counter()
        schedule = []
        summary = Counter()
        for day in range(DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, arrivals, result)
            for passengers in queues.values():
                passengers.sort(key=BY_ID)
            active, _ = self.daily_loads(state, queues, module_distances, result, set(), route_cache, choice_cache)
            _, reservations, claimed = self.fixed_load_assignments(f_pods, positions, active, queues, wanted_edges, day, True)
            schedule.append((reservations, claimed))
            if not active:
                schedule.extend((set(), {}) for _ in range(day + 1, DAYS))
                break
            requests = {}
            for pod_id, pod in f_pods:
                next_index = fixed_next_index(pod.path, positions[pod_id])
                if next_index != positions[pod_id]:
                    requests[pod_id] = pod.path[positions[pod_id]], pod.path[next_index]
            moves = self.allocate_tube_capacity(requests, state, result, day, set())
            self.board_and_launch(queues, distances, state, moves, positions, {}, {})
            self.settle(day + 1, queues, arrivals, result)
        delivered = {passenger.id for passengers in initial.values() for passenger in passengers} \
            - {passenger.id for passengers in queues.values() for passenger in passengers}
        future = set()
        filtered = [set() for _ in range(DAYS)]
        for day in range(DAYS - 1, -1, -1):
            reservations, claimed = schedule[day]
            future.update(reservations & delivered)
            filtered[day] = set(future)
            for path, passenger_ids in claimed.items():
                summary[path.nodes] += len(passenger_ids & delivered)
        return filtered, summary
    def fixed_load_assignments(self, f_pods, positions,
            active, queues,
            wanted_edges, day, claims = False):
        by_nodes = {}
        for path in active:
            by_nodes.setdefault(path.nodes, []).append(path)
        assignments = {}
        for pod_id, pod in f_pods:
            matches = by_nodes.get(pod.assignments[day], ())
            if matches:
                assignments[pod_id] = {min(matches, key=lambda item: (-item.priority, item.pool, item.destination))}
        if not claims:
            return assignments, set(), {}
        moves = {}
        for pod_id, pod in f_pods:
            next_index = fixed_next_index(pod.path, positions[pod_id])
            if next_index != positions[pod_id]:
                moves[pod_id] = pod.path[positions[pod_id]], pod.path[next_index]
        paths = {}
        for path in active:
            paths.setdefault((path.pool, path.nodes[:2]), []).append(path)
        claimed = {}
        reservations = set()
        seats = {pod_id: POD_SIZE for pod_id in moves}
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
        return assignments, reservations, claimed
    def daily_loads(self, state, queues, module_distances,
            result, reserved, route_cache,
            choice_cache):
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
                    loads.append(Load(pool, destination, run, load_count, priority, is_reserved))
                    key = pool, run[0], run[1]
                    own_priorities[key] = max(own_priorities.get(key, -1), priority)
        direction_priorities = {}
        for (_, source_id, target_id), priority in own_priorities.items():
            direction_priorities[source_id, target_id] = max(direction_priorities.get((source_id, target_id), -1), priority)
        prioritized = [path if path.reserved or path.priority == direction_priorities[path.nodes[:2]] else
            replace(path, priority=direction_priorities[path.nodes[:2]]) for path in loads]
        return prioritized, own_priorities
    def path_demands(self, state, distances,
            module_distances):
        demands = []
        for pool in self.speed_pools():
            options = [module.id for module in self.buildings.values() if module.kind == pool[1]
                and module_distances[module.id][pool[0]] == distances[pool[1]][pool[0]]]
            count = self.buildings[pool[0]].demand[pool[1]]
            for module_id in options:
                path = self.concrete_path(pool[0], module_id, state)
                for run in self.tube_path_runs(path, state):
                    demands.append(Load(pool, module_id, run, count))
        return demands
    def concrete_path(self, start_id, finish_id, state):
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
    def tube_path_runs(self, path, state):
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
    def dispatch_dynamic_paths(self, active, d_pods, current,
            pending, queues, wanted_edges,
            passenger_priorities, reserved_passengers, delivered,
            graph, day):
        assignments = {}
        prefs = {}
        inspected = set(reserved_passengers)
        edges = {path.nodes[:2] for path in active}
        eligible = {edge: [passenger for passenger in queues.get(edge[0], [])
            if edge in wanted_edges[edge[0], passenger.kind]] for edge in edges}
        for pod_id, _ in d_pods:
            source_id = current[pod_id]
            options = [path for path in active if (pending[pod_id] == (-1, -1) or
                (source_id, path.nodes[1] if source_id == path.nodes[0] else next_step(graph, source_id, path.nodes[0])) == pending[pod_id]) and
                (0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])) + len(path.nodes) - 1 <= DAYS - day]
            inspection_batches = {edge: [passenger for passenger in eligible[edge] if passenger.id not in inspected][:POD_SIZE]
                for edge in {path.nodes[:2] for path in options}}
            evaluated = []
            for path in options:
                priority = path.priority
                if min(path.cap, len(eligible[path.nodes[:2]])) < POD_SIZE:
                    priority = -1
                elif priority > 0:
                    balance = sum(passenger_priorities.get(((passenger.pad_id, passenger.kind), *path.nodes[:2]), 0)
                        for passenger in inspection_batches[path.nodes[:2]])
                    if balance < 0:
                        priority = -1
                evaluated.append(path if priority == path.priority else replace(path, priority=priority))
            prefs[pod_id] = sorted(evaluated,
                key=lambda path: self.path_assignment_key(path, pod_id, min(POD_SIZE, path.cap, len(eligible[path.nodes[:2]])),
                    current, delivered, graph))
            if prefs[pod_id]:
                assignments[pod_id] = prefs[pod_id][0]
                inspected.update(passenger.id for passenger in inspection_batches[assignments[pod_id].nodes[:2]])
        return assignments, prefs
    def resolve_edge_conflicts(self, jobs, prefs, f_pods,
            d_pods, fixed_jobs,
            positions, at, pending, graph,
            occupied, result, state, day):
        def make_req(v):
            r = self.path_pod_requests(f_pods, d_pods, positions, at, pending, v, fixed_jobs, graph)
            c = Counter(route_key(*move) for move in r.values())
            for e in sorted(c):
                while c[e] > state.tubes[e]:
                    for p in sorted(p for p, move in r.items() if route_key(*move) == e and p in v):
                        if pending[p] != (-1, -1) or at[p] in (-1, v[p].nodes[0]):
                            continue
                        t = v[p].nodes[0]
                        d = graph_distance(graph, at[p], t)
                        opts = [n for n in graph[at[p]] if route_key(at[p], n) != e and c[route_key(at[p], n)] <
                            state.tubes[route_key(at[p], n)] and graph_distance(graph, n, t) <= d]
                        if opts:
                            n = min(opts, key=lambda item: (graph_distance(graph, item, t), item))
                            c[e] -= 1
                            c[route_key(at[p], n)] += 1
                            r[p] = at[p], n
                            break
                    else:
                        break
            return r
        def load(values, edge):
            return sum(route_key(*move) == edge for move in values.values())
        def resolves(trial, pod_id):
            tr = make_req(trial)
            edge = route_key(*tr[pod_id])
            return load(tr, edge) <= state.tubes[edge], tr
        def valid(trial, uniform):
            checked = dict(trial)
            self.fix_load_assignments(checked, prefs, at, graph, state, occupied, uniform)
            return checked == trial
        req = make_req(jobs)
        lock = {p for p, _ in f_pods} | {p for p, _ in d_pods if pending[p] != (-1, -1)}
        seen = set()
        while True:
            key = tuple(sorted(jobs.items()))
            if key in seen:
                break
            seen.add(key)
            by_edge = {}
            for p, move in req.items():
                by_edge.setdefault(route_key(*move), []).append(p)
            done = False
            for edge in sorted(edge for edge, pods in by_edge.items() if len(pods) > state.tubes[edge]):
                pods = sorted(p for p in by_edge[edge] if p in jobs)
                for a in pods:
                    for b in pods:
                        levels = jobs[a].priority, jobs[b].priority
                        if a >= b or jobs[a] == jobs[b] or req[a] != req[b][::-1] or jobs[b] not in prefs[a] \
                                or jobs[a] not in prefs[b] or 0 in levels and min(levels) < 0 and \
                                not jobs[a].priority < 0:
                            continue
                        trial = dict(jobs)
                        trial[a], trial[b] = trial[b], trial[a]
                        tr = make_req(trial)
                        if load(tr, edge) >= len(by_edge[edge]):
                            continue
                        jobs, req = trial, tr
                        done = True
                        break
                    if done:
                        break
                if done:
                    break
                if not lock & set(by_edge[edge]):
                    continue
                for p in pods:
                    old = jobs[p]
                    for path in prefs[p][prefs[p].index(old) + 1:]:
                        if old.priority == 0 and path.priority < 0:
                            continue
                        trial = dict(jobs)
                        trial[p] = path
                        if valid(trial, True):
                            done, tr = resolves(trial, p)
                            if done:
                                jobs, req = trial, tr
                                break
                        if valid(trial, False):
                            continue
                        others = sorted(q for q, other in jobs.items() if q != p and other == path and old in prefs[q])
                        for q in others:
                            d1 = 0 if at[p] == -1 else graph_distance(graph, at[p], path.nodes[0])
                            d2 = 0 if at[q] == -1 else graph_distance(graph, at[q], path.nodes[0])
                            if d1 >= d2:
                                continue
                            trial = dict(jobs)
                            trial[p], trial[q] = trial[q], trial[p]
                            done, tr = resolves(trial, p)
                            if done:
                                jobs, req = trial, tr
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
            if not done:
                break
        levels = {p: path.priority for p, path in jobs.items()}
        levels.update((p, next(iter(paths)).priority) for p, paths in fixed_jobs.items() if paths)
        by_edge = {}
        for p, move in req.items():
            by_edge.setdefault(route_key(*move), []).append(p)
        norms = Counter(route_key(*move) for p, move in req.items() if levels.get(p) == 0)
        for edge, pods in sorted(by_edge.items()):
            excess = len(pods) - state.tubes[edge]
            normal = [p for p in pods if levels.get(p) == 0]
            for p in sorted(p for p in pods if p in jobs and levels[p] < 0):
                if excess <= 0 or not any(p < q for q in normal):
                    continue
                moves = [(a, b) for a in graph for b in graph[a]] if at[p] == -1 else [(at[p], b) for b in graph[at[p]]]
                moves = [move for move in moves if route_key(*move) != edge and
                    norms[route_key(*move)] < state.tubes[route_key(*move)]]
                if moves:
                    req[p] = min(moves)
                    excess -= 1
        return jobs, req, self.allocate_tube_capacity(req, state, result, day, set())
    def path_assignment_key(self, path, pod_id, boarding, current,
            delivered, graph):
        distance = 0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])
        return -path.priority, -boarding, distance, len(path.nodes) - 1, delivered[path.destination], \
            -path.cap, path.pool, path.destination, path.nodes
    def fix_load_assignments(self, assignments, prefs,
            current, graph, state, occupied,
            uniform):
        def load_cap(path):
            return (path.cap + POD_SIZE - 1) // POD_SIZE
        def allocation(values):
            def assign(path, seen):
                for edge in path_edges[path]:
                    for slot in range(occupied[edge], state.tubes[edge]):
                        key = edge, slot
                        if key in seen:
                            continue
                        seen.add(key)
                        if key not in matches or assign(matches[key], seen):
                            matches[key] = path
                            return True
                return False
            matches = {}
            failure = None
            for path in sorted(paths, key=lambda item: (min(load_cap(item), sum(max(0, state.tubes[edge] - occupied[edge])
                    for edge in path_edges[item])), -item.priority, item.pool, item.destination, item.nodes)):
                if values[path] > load_cap(path) and failure is None:
                    failure = path, False
                for _ in range(min(values[path], load_cap(path))):
                    if not assign(path, set()) and failure is None:
                        failure = path, True
            return len(matches), failure
        def overflow(values):
            return allocation(values)[1]
        def uniform_conflicts(values):
            return {path for path in paths if values[path] and any(candidate.priority == path.priority and values[candidate] < values[path] - 1
                and overflow(Counter({candidate: values[candidate] + 1})) is None for candidate in paths)}
        counts = Counter(assignments.values())
        owners = {}
        for pod_id, path in assignments.items():
            owners.setdefault(path, set()).add(pod_id)
        paths = sorted({path for pod_paths in prefs.values() for path in pod_paths},
            key=lambda item: (item.pool, item.destination, item.nodes))
        path_edges = {path: edges_of(path.nodes) for path in paths}
        distances = {(pod_id, path.nodes[0]): 0 if current[pod_id] == -1 else graph_distance(graph, current[pod_id], path.nodes[0])
            for pod_id in assignments for path in paths}
        indexes = {pod_id: prefs[pod_id].index(path) for pod_id, path in assignments.items()}
        while True:
            uneven = uniform_conflicts(counts) if uniform else set()
            exceeded = None if uneven else overflow(counts)
            if exceeded is None and not uneven:
                break
            path = exceeded[0] if exceeded else min(uneven, key=lambda item: (item.pool, item.destination, item.nodes))
            removed = []
            while True:
                pod_id = min(owners[path], key=lambda item: (-distances[item, path.nodes[0]], INF if indexes[item] + 1 == len(prefs[item]) else
                    distances[item, prefs[item][indexes[item] + 1].nodes[0]] - distances[item, path.nodes[0]], -item))
                owners[path].remove(pod_id)
                counts[path] -= 1
                del assignments[pod_id]
                removed.append(pod_id)
                next_uneven = uniform_conflicts(counts) if uniform else set()
                next_exceeded = None if next_uneven else overflow(counts)
                if not (next_exceeded and next_exceeded[0] == path or path in next_uneven):
                    break
            for pod_id in removed:
                indexes[pod_id] += 1
                if indexes[pod_id] == len(prefs[pod_id]):
                    continue
                next_path = prefs[pod_id][indexes[pod_id]]
                assignments[pod_id] = next_path
                owners.setdefault(next_path, set()).add(pod_id)
                counts[next_path] += 1
    def path_pod_requests(self, f_pods, d_pods,
            positions, current, pending, assignments,
            fixed_jobs, graph):
        requests = {}
        for pod_id, pod in f_pods:
            index = positions[pod_id]
            next_index = fixed_next_index(pod.path, index)
            if next_index != index:
                requests[pod_id] = pod.path[index], pod.path[next_index]
        if not d_pods:
            return requests
        options = {}
        protected_edges = set()
        for pod_id, pod in f_pods:
            index = positions[pod_id]
            for path in fixed_jobs.get(pod_id, ()):
                route = list(path.nodes) if pod.path[index] == path.nodes[0] else pod.path[index:] + \
                    (pod.path[1:index + 1] if pod.path[0] == pod.path[-1] else [])
                if path.nodes[0] in route[1:]:
                    route = route[:route.index(path.nodes[0], 1) + 1]
                protected_edges.update(edges_of(route))
        all_edges = [(a, b) for a in sorted(graph) for b in graph[a] if a < b]
        for pod_id, _ in d_pods:
            if pending[pod_id] != (-1, -1):
                options[pod_id] = [pending[pod_id]]
                continue
            if pod_id not in assignments:
                continue
            path = assignments[pod_id].nodes
            if current[pod_id] == -1:
                options[pod_id] = [path[:2]]
                delivering = True
            else:
                target_id = path[1] if current[pod_id] == path[0] else next_step(graph, current[pod_id], path[0])
                options[pod_id] = [(current[pod_id], target_id)]
                source_id = current[pod_id]
                delivering = current[pod_id] == path[0]
            if delivering:
                protected_edges.update(edges_of(path))
            else:
                protected_edges.add(route_key(*options[pod_id][0]))
                while source_id != path[0]:
                    target_id = next_step(graph, source_id, path[0])
                    protected_edges.add(route_key(source_id, target_id))
                    source_id = target_id
        requested_edges = {route_key(*move) for move in requests.values()} | {route_key(*moves[0]) for moves in options.values()}
        for pod_id, _ in d_pods:
            if pod_id in assignments:
                continue
            if pending[pod_id] != (-1, -1):
                options[pod_id] = [pending[pod_id]]
                continue
            moves = all_edges if current[pod_id] == -1 else [(current[pod_id], neighbor_id) for neighbor_id in graph[current[pod_id]]]
            options[pod_id] = [min(moves, key=lambda move: (route_key(*move) in protected_edges, route_key(*move) in requested_edges, move))]
        requests.update((pod_id, moves[0]) for pod_id, moves in options.items())
        return requests
    def distances_to_targets(self, state):
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
    def initial_queues(self, distances):
        queues = {}
        for pad in self.landing_pads():
            passengers = [Person(pad.id, kind, pad.id * 1000 + index) for index, kind in enumerate(pad.order) if distances[kind][pad.id] < INF]
            if passengers:
                queues[pad.id] = passengers
        return queues
    def wanted_edges(self, distances, graph):
        wanted = {}
        for kind, kind_distances in distances.items():
            for building_id, distance in kind_distances.items():
                options = [neighbor_id for neighbor_id in graph.get(building_id, []) if kind_distances[neighbor_id] < distance]
                wanted[building_id, kind] = tuple((building_id, neighbor_id) for neighbor_id in options)
        return wanted
    def teleport_phase(self, queues, distances, teleports):
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
    def settle(self, day, queues, arrivals, result):
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
    def allocate_tube_capacity(self, requests, state, result, day,
            assigned):
        moves = {}
        by_tube = {}
        for pod_id, move in requests.items():
            by_tube.setdefault(route_key(*move), []).append((pod_id, move))
        for edge, pods in by_tube.items():
            capacity = state.tubes[edge]
            selected = sorted(pods)[:capacity]
            if any(pod_id in assigned for pod_id, _ in sorted(pods)[capacity:]):
                daily = result.congestion_by_day.setdefault(day, Counter())
                if not daily[edge]:
                    result.congestion_by_edge[edge] += 1
                    daily[edge] = 1
            for pod_id, move in selected:
                moves[pod_id] = move
        return moves
    def board_and_launch(self, queues, distances, state,
            moves, positions, dynamic_current,
            dynamic_pending):
        by_start = {}
        for pod_id, (source_id, target_id) in moves.items():
            by_start.setdefault(source_id, []).append((pod_id, target_id))
        for candidates in by_start.values():
            candidates.sort()
        seats = {pod_id: POD_SIZE for pod_id in moves}
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
            if pod_id in positions:
                positions[pod_id] = fixed_next_index(state.pods[pod_id].path, positions[pod_id])
            else:
                dynamic_current[pod_id] = target_id
                dynamic_pending[pod_id] = (-1, -1)
            if onboard.get(pod_id):
                queues.setdefault(target_id, []).extend(onboard[pod_id])
        return set(onboard)
    def cheapest_hop_path(self, start_id, targets, hop_count, state):
        return self.cheapest_paths_by_hop(start_id, targets, hop_count, state).get(hop_count, [])
    def cheapest_path_with_hop_limit(self, start_id, targets, hop_limit,
            state):
        paths = self.cheapest_paths_by_hop(start_id, targets, hop_limit, state)
        return min(paths.values(), key=lambda path: (sum(0 if route_key(a, b) in state.tubes else
            tube_cost(self.buildings[a], self.buildings[b]) for a, b in zip(path, path[1:])), path), default=[])
    def cheapest_paths_by_hop(self, start_id, targets, hop_limit,
            state):
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
    def build_candidate_edge_graph(self, tubes):
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
    def can_add_tubes(self, tubes, existing_tubes):
        test_tubes = dict(existing_tubes)
        degrees = self.tube_degrees(test_tubes)
        for a, b in tubes:
            key = route_key(a, b)
            if key in test_tubes:
                continue
            if degrees[a] >= MAX_DEGREE or degrees[b] >= MAX_DEGREE:
                return False
            if not self.can_build_tube(a, b, test_tubes):
                return False
            test_tubes[key] = 1
            degrees[a] += 1
            degrees[b] += 1
        return True
    def can_build_tube(self, a, b, tubes):
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
    def speed_pools(self):
        pools = []
        for pad in self.landing_pads():
            pools.extend((pad.id, kind) for kind in sorted(pad.demand))
        return pools
    def landing_pads(self):
        return [building for building in sorted(self.buildings.values(), key=lambda item: item.id) if building.kind == 0]
    def tube_degrees(self, tubes):
        degrees = Counter()
        for a, b in tubes:
            degrees[a] += 1
            degrees[b] += 1
        return degrees
    def teleport_used_buildings(self, teleports):
        used = set(teleports)
        used.update(teleports.values())
        return used
    def next_pod_id(self, pods):
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
            assignments = ", ".join("-".join(map(str, path)) if path else "-" for path in self.pods[pod_id].assignments)
            print(f"pod id={pod_id}, assignments=[{assignments}], path=[{path_text}]", file=sys.stderr)
    def perfect_diversity(self, kind):
        demand = sum(pad.demand[kind] for pad in self.landing_pads())
        module_count = sum(building.kind == kind for building in self.buildings.values())
        balanced_population = (demand + module_count - 1) // module_count
        return sum(max(0, 50 - index) for index in range(balanced_population))
def route_key(a, b):
    return (a, b) if a < b else (b, a)
def edges_of(path):
    return tuple(route_key(*edge) for edge in zip(path, path[1:]))
def tube_cost(a, b):
    return isqrt(100 * ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)))
def orientation(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
def point_on_segment(point, a, b):
    return orientation(a, b, point) == 0 and min(a.x, b.x) <= point.x <= max(a.x, b.x) and min(a.y, b.y) <= point.y <= max(a.y, b.y)
def segments_intersect(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 == 0 and point_on_segment(c, a, b) or o2 == 0 and point_on_segment(d, a, b):
        return True
    if o3 == 0 and point_on_segment(a, c, d) or o4 == 0 and point_on_segment(b, c, d):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
def unique_new_tubes(path, tubes):
    result = []
    seen = set()
    for a, b in zip(path, path[1:]):
        edge = route_key(a, b)
        if edge not in tubes and edge not in seen:
            result.append(edge)
            seen.add(edge)
    return result
def tube_graph(tubes):
    graph = {}
    for a, b in tubes:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return {node: sorted(neighbors) for node, neighbors in graph.items()}
def graph_distance(graph, start_id, finish_id):
    if start_id == finish_id:
        return 0
    d = _G.setdefault(id(graph), (graph, {}, {}))[1]
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
def next_step(graph, start_id, finish_id):
    if start_id == finish_id:
        return start_id
    steps = _G.setdefault(id(graph), (graph, {}, {}))[2]
    key = start_id, finish_id
    if key in steps:
        return steps[key]
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
    return steps.setdefault(key, step)
def normalize_month_path(path):
    if len(path) >= DAYS + 1:
        return path[:DAYS + 1]
    if len(path) < 2:
        return path[:]
    edges = list(zip(path, path[1:]))
    if path[0] != path[-1]:
        edges.extend(zip(path[-1:0:-1], path[-2::-1]))
    result = [path[0]]
    for day in range(DAYS):
        result.append(edges[day % len(edges)][1])
    return result
def fixed_next_index(path, index):
    if len(path) < 2:
        return index
    if index < len(path) - 1:
        return index + 1
    if path[0] == path[-1]:
        return 1
    return index
if __name__ == "__main__":
    Planner().play()
