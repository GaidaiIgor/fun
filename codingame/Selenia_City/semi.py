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
OVERRIDE_COMMAND = "TUBE 4 8;TUBE 2 7;TUBE 1 2;TUBE 1 8;POD 1;POD 2;POD 3;POD 4"
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
    """Groups passengers on nodes; cap counts them, priority ranks them, batch holds delivery lengths, edges holds depth/edge pairs."""
    nodes: Pair
    cap: int = field(compare=False)
    priority: int = field(compare=False)
    batch: tuple[int, ...] = field(compare=False)
    edges: tuple[tuple[int, Pair], ...] = field(compare=False)
def assignment_text(paths: set[Load], pod_id: int, requests: dict, moves: dict, carrying: set) -> str:
    """Formats paths for pod_id using requests, accepted moves and the actual carrying pods."""
    return "/".join(("   " if pod_id not in requests else "E! " if pod_id not in moves else
        ".. " if pod_id in carrying and any(requests[pod_id] == edge for _, edge in path.edges) else "-> ") +
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
                used.update(edges_of(demand))
            origins = sorted({demand[0] for demand in demands})
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
            distances, _ = self.distances_to_targets(state)
            graph = tube_graph(state.tubes)
            self.fs_cache[network_key] = distances, graph, self.wanted_edges(distances, graph), self.initial_queues(distances), {}
        distances, graph, wanted_edges, initial, route_cache = self.fs_cache[network_key]
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
        result = Result()
        f_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if not pod.dynamic]
        d_pods = [(pod_id, pod) for pod_id, pod in sorted(state.pods.items()) if pod.dynamic]
        fixed_usage = Counter(edge for _, pod in f_pods for edge in set(edges_of(pod.path)))
        shared_capacities = tuple((edge, state.tubes[edge]) for edge, count in sorted(fixed_usage.items()) if count > 1)
        key = network_key, shared_capacities, tuple((pod_id, tuple(pod.path), tuple(pod.assignments)) for pod_id, pod in f_pods)
        if d_pods and key not in self.fs_cache:
            self.fs_cache[key] = self.fixed_assignment_schedule(state, distances, f_pods)
        fixed_reservations, summary = self.fs_cache[key] if d_pods else ([], Counter())
        if keep_dynamic_paths:
            result.reserved = ", ".join("({})x{}".format("-".join(map(str, path)), count)
                for path, count in summary.items()) or "-"
        positions = {pod_id: 0 for pod_id, _ in f_pods}
        at = {pod_id: -1 for pod_id, _ in d_pods}
        pending = {pod_id: (-1, -1) for pod_id, _ in d_pods}
        dynamic_paths = {pod_id: [] for pod_id, _ in d_pods}
        pod_assignments = {pod_id: [] for pod_id in state.pods}
        arrivals = Counter()
        for day in range(DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, arrivals, result)
            for passengers in queues.values():
                passengers.sort(key=BY_ID)
            if not d_pods and not FULL_DEBUG and not keep_dynamic_paths:
                if not any(wanted_edges[node_id, passenger.kind] for node_id, passengers in queues.items() for passenger in passengers):
                    break
                routes = self.pod_routes(f_pods, [], positions, {}, {}, {}, graph, day=day)
                requests = {pod_id: route[:2] for pod_id, route in routes.items() if len(route) > 1}
                assigned = {pod_id for pod_id, pod in f_pods if day < len(pod.assignments) and pod.assignments[day]}
                moves = self.allocate_tube_capacity(requests, state, result, day, assigned)
                result.carrying_days.update(self.board_and_launch(queues, distances, state, moves, positions, {}, {}))
                self.settle(day + 1, queues, arrivals, result)
                continue
            reserved_passengers = fixed_reservations[day] if d_pods else set()
            active = self.daily_loads(state, queues, distances, wanted_edges, reserved_passengers, route_cache)
            if not active:
                break
            by_edge = {load.nodes: load for load in active}
            fixed_jobs = {pod_id: {by_edge[pod.assignments[day][:2]]} for pod_id, pod in f_pods if pod.assignments[day][:2] in by_edge}
            for pod_id, _ in f_pods:
                paths = fixed_jobs.get(pod_id)
                pod_assignments[pod_id].append(next(iter(paths)).nodes if paths else ())
            if not d_pods:
                routes = self.pod_routes(f_pods, [], positions, {}, {}, {}, graph, day=day)
                requests = {pod_id: route[:2] for pod_id, route in routes.items() if len(route) > 1}
                moves = self.allocate_tube_capacity(requests, state, result, day, set(fixed_jobs))
                locations = {pod_id: pod.path[positions[pod_id]] for pod_id, pod in f_pods} if FULL_DEBUG else {}
                carrying = self.board_and_launch(queues, distances, state, moves, positions, {}, {})
                result.carrying_days.update(carrying)
                if FULL_DEBUG:
                    loads = ", ".join("({})x{}{}".format("-".join(map(str, path.nodes)), path.cap,
                        "L" if path.priority < 0 else "H" if path.priority > 0 else "N") for path in active)
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
            assignments, prefs = self.dispatch_dynamic_loads(active, d_pods, at, pending, graph, day)
            ideal = assignments.copy()
            self.fix_passenger_capacity(ideal, prefs)
            ideal_routes = self.pod_routes(f_pods, d_pods, positions, at, pending, ideal, graph, day=day, fixed_jobs=fixed_jobs)
            ideal_req = {pod_id: route[:2] for pod_id, route in ideal_routes.items() if len(route) > 1}
            if FULL_DEBUG:
                i_jobs = assignments.copy()
                i_routes = self.pod_routes(f_pods, d_pods, positions, at, pending, assignments, graph, day=day, fixed_jobs=fixed_jobs)
                i_req = {pod_id: route[:2] for pod_id, route in i_routes.items() if len(route) > 1}
                i_moves = i_req
                i_at = {pod_id: pod.path[positions[pod_id]] if not pod.dynamic else
                    at[pod_id] if at[pod_id] != -1 else i_req.get(pod_id, ("-",))[0]
                    for pod_id, pod in state.pods.items()}
                i_carry = self.board_and_launch({building_id: passengers[:] for building_id, passengers in queues.items()},
                    distances, state, i_moves, positions.copy(), at.copy(), pending.copy())
            assignments = ideal.copy()
            bookings, path_capped = self.fix_finite_capacity(assignments, prefs, state, occupied, at, pending, graph)
            if FULL_DEBUG:
                c_jobs = ideal.copy()
                c_req = ideal_req
                c_moves = c_req
                c_at = {pod_id: pod.path[positions[pod_id]] if not pod.dynamic else
                    at[pod_id] if at[pod_id] != -1 else c_req.get(pod_id, ("-",))[0]
                    for pod_id, pod in state.pods.items()}
                c_carry = self.board_and_launch({building_id: passengers[:] for building_id, passengers in queues.items()},
                    distances, state, c_moves, positions.copy(), at.copy(), pending.copy())
            routes = self.pod_routes(f_pods, d_pods, positions, at, pending, assignments, graph, bookings, day, fixed_jobs)
            self.resolve_alternative_routes(routes, assignments, fixed_jobs, pending, graph, state, queues, wanted_edges, day)
            self.idle_pod_routes(routes, assignments, fixed_jobs, at, pending, graph)
            requests = {pod_id: route[:2] for pod_id, route in routes.items() if len(route) > 1}
            moves = self.allocate_tube_capacity(requests, state, result, day, set())
            blocked = {route_key(*min(ideal[p].edges, key=lambda item: (state.tubes[route_key(*item[1])], item))[1])
                for p in path_capped if p in ideal and assignments.get(p) != ideal[p]}
            for p in (set(ideal) | set(fixed_jobs)) - path_capped:
                if p not in ideal_req:
                    continue
                alternative = p in ideal and assignments.get(p) == ideal[p] and moves.get(p) == requests.get(p) and \
                    routes[p][-1] == ideal_routes[p][-1] and len(routes[p]) == len(ideal_routes[p])
                if moves.get(p) != ideal_req[p] and not alternative:
                    blocked.add(route_key(*ideal_req[p]))
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
                    "L" if path.priority < 0 else "H" if path.priority > 0 else "N") for path in active)
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
    def fixed_assignment_schedule(self, state: State, distances: dict, f_pods: list) -> tuple:
        """Simulates f_pods on state using distances; returns daily reserved passenger ids and delivered edge counts."""
        if not f_pods:
            return [set() for _ in range(DAYS)], Counter()
        initial = self.initial_queues(distances)
        queues = {building_id: passengers[:] for building_id, passengers in initial.items()}
        result = Result()
        positions = {pod_id: 0 for pod_id, _ in f_pods}
        arrivals = Counter()
        schedule = []
        for day in range(DAYS):
            self.teleport_phase(queues, distances, state.teleports)
            self.settle(day, queues, arrivals, result)
            for passengers in queues.values():
                passengers.sort(key=BY_ID)
            before = {passenger.id: node for node, passengers in queues.items() for passenger in passengers}
            requests = {pod_id: (pod.path[positions[pod_id]], pod.path[fixed_next_index(pod.path, positions[pod_id])])
                for pod_id, pod in f_pods if fixed_next_index(pod.path, positions[pod_id]) != positions[pod_id]}
            moves = self.allocate_tube_capacity(requests, state, result, day, set())
            self.board_and_launch(queues, distances, state, moves, positions, {}, {})
            schedule.append({passenger.id: (before[passenger.id], node) for node, passengers in queues.items()
                for passenger in passengers if before[passenger.id] != node})
            self.settle(day + 1, queues, arrivals, result)
        delivered = {passenger.id for passengers in initial.values() for passenger in passengers} \
            - {passenger.id for passengers in queues.values() for passenger in passengers}
        future = set()
        filtered = [set() for _ in range(DAYS)]
        summary = Counter()
        for day in range(DAYS - 1, -1, -1):
            future.update(schedule[day].keys() & delivered)
            filtered[day] = set(future)
            summary.update(edge for passenger_id, edge in schedule[day].items() if passenger_id in delivered)
        return filtered, summary
    def daily_loads(self, state: State, queues: dict, distances: dict, wanted_edges: dict, reserved: set, route_cache: dict) -> list[Load]:
        """Groups queues by wanted_edges using state and distances, marks reserved loads and caches delivery edges in route_cache."""
        generating = {}
        for node, passengers in queues.items():
            for passenger in passengers:
                for edge in wanted_edges[node, passenger.kind]:
                    generating.setdefault(edge, []).append(passenger)
        loads = []
        for edge, passengers in sorted(generating.items()):
            delivery_edges = {}
            for kind in {passenger.kind for passenger in passengers}:
                key = edge, kind
                if key not in route_cache:
                    edges = {edge: 0}
                    todo = [edge[1]]
                    seen = set()
                    while todo:
                        node = todo.pop()
                        if node in seen or distances[kind][node] == 0:
                            continue
                        seen.add(node)
                        exit_id = state.teleports.get(node)
                        if exit_id is not None and distances[kind][exit_id] <= distances[kind][node]:
                            todo.append(exit_id)
                            continue
                        for following in wanted_edges[node, kind]:
                            edges[following] = distances[kind][edge[0]] - distances[kind][node]
                            todo.append(following[1])
                    route_cache[key] = tuple((depth, path_edge) for path_edge, depth in edges.items())
                for depth, path_edge in route_cache[key]:
                    delivery_edges[path_edge] = min(depth, delivery_edges.get(path_edge, INF))
            batch = tuple(distances[passenger.kind][edge[0]] for passenger in passengers[:POD_SIZE])
            loads.append(Load(edge, len(passengers), -int(all(passenger.id in reserved for passenger in passengers)), batch,
                tuple(sorted((depth, path_edge) for path_edge, depth in delivery_edges.items()))))
        return loads
    def path_demands(self, state, distances,
            module_distances):
        demands = []
        for pool in self.speed_pools():
            options = [module.id for module in self.buildings.values() if module.kind == pool[1]
                and module_distances[module.id][pool[0]] == distances[pool[1]][pool[0]]]
            for module_id in options:
                path = self.concrete_path(pool[0], module_id, state)
                for run in self.tube_path_runs(path, state):
                    demands.append(run)
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
    def dispatch_dynamic_loads(self, active: list[Load], d_pods: list, current: dict, pending: dict, graph: dict, day: int) -> tuple:
        """Ranks active loads for d_pods from current and pending positions on graph at day; returns assignments and preferences."""
        assignments, prefs = {}, {}
        scores = {}
        for pod_id, _ in d_pods:
            evaluated = []
            for load in active:
                reach = self.pod_distance(pod_id, load.nodes[0], current, pending, graph)
                if current[pod_id] == load.nodes[0] and pending[pod_id] == load.nodes:
                    reach = 0
                key = load, reach
                if key not in scores:
                    scores[key] = sum((50 - day - reach - length) / (reach + length)
                        for length in load.batch if reach + length <= DAYS - day)
                if scores[key] > 0:
                    evaluated.append((load, scores[key]))
            evaluated.sort(key=lambda item: (-item[0].priority, -item[1], -item[0].cap, item[0].nodes))
            prefs[pod_id] = [*evaluated, (None, 0)]
            if evaluated:
                assignments[pod_id] = evaluated[0][0]
        return assignments, prefs
    def pod_distance(self, pod_id: int, target: int, current: dict, pending: dict, graph: dict) -> int:
        """Returns distance to target on graph for pod_id, respecting current location and pending movement."""
        if current[pod_id] == -1:
            return 0
        if pending[pod_id] != (-1, -1):
            return 1 + graph_distance(graph, pending[pod_id][1], target)
        return graph_distance(graph, current[pod_id], target)
    def fix_passenger_capacity(self, assignments: dict, prefs: dict):
        """Advances assignments through prefs until generating passengers can supply all assigned pods."""
        while True:
            counts = Counter(assignments.values())
            load = next((load for load in sorted(counts, key=load_id) if counts[load] > (load.cap + POD_SIZE - 1) // POD_SIZE), None)
            if load is None:
                return
            pods = [pod_id for pod_id, assigned in assignments.items() if assigned == load]
            self.advance_assignment(min(pods, key=lambda pod_id: (-self.next_efficiency(pod_id, load, prefs), pod_id)), assignments, prefs)
    def fix_finite_capacity(self, assignments: dict, prefs: dict, state: State, occupied: Counter, current: dict, pending: dict, graph: dict) -> tuple:
        """Resolves assignments using prefs and state slots minus occupied; returns bookings and capped pods using current/pending graph distances."""
        def choices(pod_id: int, load: Load) -> list:
            """Returns reachable-edge choices for pod_id and load in distance/depth order."""
            key = pod_id, load
            if key not in options:
                options[key] = sorted((min(self.pod_distance(pod_id, node, current, pending, graph) for node in edge), depth, edge)
                    for depth, edge in load.edges)
            return options[key]
        def reserve() -> tuple:
            """Books current assignments upstream first; returns bookings, usage and the next conflict group."""
            bookings, owners = {}, {}
            usage = occupied.copy()
            on_path = []
            for pod_id, load in assignments.items():
                location = current[pod_id]
                depths = [depth + (location == edge[1]) for depth, edge in load.edges if location in edge]
                if location == -1 or depths:
                    on_path.append((min(depths, default=0), pod_id))
            for _, pod_id in sorted(on_path):
                load = assignments[pod_id]
                eligible = [(reach, depth, edge) for reach, depth, edge in choices(pod_id, load)
                    if reach < INF and usage[route_key(*edge)] < state.tubes[route_key(*edge)]]
                if not eligible:
                    group = {pod_id}
                    todo = [pod_id]
                    while todo:
                        member = todo.pop()
                        for _, _, edge in choices(member, assignments[member]):
                            for other in owners.get(route_key(*edge), ()):
                                if other not in group:
                                    group.add(other)
                                    todo.append(other)
                    return bookings, usage, group
                edge = min(eligible)[2]
                bookings[pod_id] = edge
                usage[route_key(*edge)] += 1
                owners.setdefault(route_key(*edge), []).append(pod_id)
            outside = set(assignments) - set(bookings)
            while outside:
                bids = {}
                for pod_id in sorted(outside):
                    eligible = [(reach, depth, edge) for reach, depth, edge in choices(pod_id, assignments[pod_id])
                        if reach < INF and usage[route_key(*edge)] < state.tubes[route_key(*edge)]]
                    if not eligible:
                        return bookings, usage, {pod_id}
                    bids[pod_id] = eligible
                reach, _, edge, pod_id = min((*bid, pod_id) for pod_id, candidates in bids.items() for bid in candidates)
                tied = {other for other, candidates in bids.items() if any(item[0] == reach and route_key(*item[2]) == route_key(*edge)
                    for item in candidates)}
                if len(tied) > state.tubes[route_key(*edge)] - usage[route_key(*edge)]:
                    return bookings, usage, tied
                bookings[pod_id] = edge
                usage[route_key(*edge)] += 1
                outside.remove(pod_id)
            return bookings, usage, set()
        options = {}
        loads = sorted({load for values in prefs.values() for load, _ in values if load}, key=load_id)
        capped = set()
        while True:
            self.fix_passenger_capacity(assignments, prefs)
            bookings, usage, group = reserve()
            if group:
                pod_id = min(group, key=lambda pod_id: (-self.next_efficiency(pod_id, assignments[pod_id], prefs), pod_id))
                capped.add(pod_id)
                self.advance_assignment(pod_id, assignments, prefs)
                continue
            counts = Counter(assignments.values())
            noncapped = [load for load in loads if counts[load] < (load.cap + POD_SIZE - 1) // POD_SIZE and
                any(usage[route_key(*edge)] < state.tubes[route_key(*edge)] for _, edge in load.edges)]
            uneven = next((load for load in loads if any(other.priority == load.priority and counts[load] > counts[other] + 1
                for other in noncapped)), None)
            if uneven is None:
                return bookings, capped
            pods = [pod_id for pod_id, assigned in assignments.items() if assigned == uneven]
            self.advance_assignment(min(pods, key=lambda pod_id: (-self.next_efficiency(pod_id, uneven, prefs), pod_id)), assignments, prefs)
    def next_efficiency(self, pod_id: int, load: Load, prefs: dict) -> float:
        """Returns efficiency of the entry after load in pod_id's prefs."""
        index = next(index for index, item in enumerate(prefs[pod_id]) if item[0] == load)
        return prefs[pod_id][index + 1][1]
    def advance_assignment(self, pod_id: int, assignments: dict, prefs: dict):
        """Advances pod_id in assignments to the next load in prefs."""
        index = next(index for index, item in enumerate(prefs[pod_id]) if item[0] == assignments[pod_id])
        del assignments[pod_id]
        if prefs[pod_id][index + 1][0]:
            assignments[pod_id] = prefs[pod_id][index + 1][0]
    def pod_routes(self, f_pods: list, d_pods: list, positions: dict, current: dict, pending: dict, assignments: dict, graph: dict,
            bookings: dict = None, day: int = 0, fixed_jobs: dict = None) -> dict:
        """Routes f_pods from positions and d_pods from current/pending toward assignments/bookings on graph; day and fixed_jobs guide idle pods."""
        bookings = bookings or {}
        routes = {}
        for pod_id, pod in f_pods:
            index = positions[pod_id]
            route = [pod.path[index]]
            for _ in range(DAYS - day):
                following = fixed_next_index(pod.path, index)
                if following == index:
                    break
                index = following
                route.append(pod.path[index])
            routes[pod_id] = tuple(route)
        for pod_id, _ in d_pods:
            if pod_id not in assignments:
                if pending[pod_id] != (-1, -1):
                    routes[pod_id] = pending[pod_id]
                continue
            edge = bookings.get(pod_id, assignments[pod_id].nodes)
            source = edge[0] if current[pod_id] == -1 else current[pod_id]
            if pod_id not in bookings or source in edge:
                target = edge[1] if source == edge[0] else edge[0]
            else:
                target = min(edge, key=lambda node: (self.pod_distance(pod_id, node, current, pending, graph), edge.index(node)))
            first = pending[pod_id][1] if pending[pod_id] != (-1, -1) else next_step(graph, source, target)
            routes[pod_id] = graph_route(graph, source, target, first)
        self.idle_pod_routes(routes, assignments, fixed_jobs or {}, current, pending, graph)
        return routes
    def idle_pod_routes(self, routes: dict, assignments: dict, fixed_jobs: dict, current: dict, pending: dict, graph: dict):
        """Updates idle routes from current/pending on graph to avoid assignments and fixed_jobs."""
        protected = {edge for pod_id in assignments for edge in edges_of(routes[pod_id])}
        for pod_id, loads in fixed_jobs.items():
            route = routes[pod_id]
            edge = next(iter(loads)).nodes
            target = edge[1] if route[0] == edge[0] else edge[0]
            end = route.index(target, 1) + 1 if target in route[1:] else len(route)
            protected.update(edges_of(route[:end]))
        idle = set(current) - set(assignments)
        for pod_id in idle:
            if pending[pod_id] == (-1, -1):
                routes.pop(pod_id, None)
        requested = {route_key(*route[:2]) for route in routes.values() if len(route) > 1}
        for pod_id in sorted(idle):
            if pending[pod_id] != (-1, -1):
                continue
            candidates = [(node, neighbor) for node in sorted(graph) for neighbor in graph[node]] if current[pod_id] == -1 else \
                [(current[pod_id], neighbor) for neighbor in graph[current[pod_id]]]
            move = min(candidates, key=lambda edge: (route_key(*edge) in protected, route_key(*edge) in requested, edge))
            routes[pod_id] = move
            requested.add(route_key(*move))
    def resolve_alternative_routes(self, routes: dict, assignments: dict, fixed_jobs: dict, pending: dict, graph: dict, state: State,
            queues: dict, wanted_edges: dict, day: int):
        """Reroutes assignments around state conflicts on graph; fixed_jobs, pending, queues, wanted_edges and day constrain route choices."""
        def waiting(pod_id: int, edge: Pair) -> int:
            """Returns consecutive days edge is blocked by pods preceding pod_id."""
            return next((offset for offset, occupied in enumerate(schedule) if
                sum(other < pod_id for other in occupied.get(route_key(*edge), ())) < state.tubes[route_key(*edge)]), len(schedule))
        def free_route(pod_id: int, route: Path) -> Path:
            """Finds an equal-length conflict-free alternative to route for pod_id."""
            def search(node: int, step: int) -> Path:
                """Returns a feasible suffix from node at step, or an empty tuple."""
                if step == length:
                    return (node,) if node == target else ()
                key = node, step
                if key in failed:
                    return ()
                for neighbor in graph[node]:
                    if graph_distance(graph, neighbor, target) > length - step - 1:
                        continue
                    occupied = attempts[step].get(route_key(node, neighbor), ()) if step < len(attempts) else ()
                    if sum(other != pod_id and (other < pod_id or other in assigned) for other in occupied) >= state.tubes[route_key(node, neighbor)]:
                        continue
                    tail = search(neighbor, step + 1)
                    if tail:
                        return (node, *tail)
                failed.add(key)
                return ()
            length, target = len(route) - 1, route[-1]
            failed = set()
            return search(route[0], 0)
        assigned = set(assignments) | set(fixed_jobs)
        schedule, attempts = self.project_routes(routes, state.tubes, DAYS - day)
        for pod_id in sorted(assignments):
            if pending[pod_id] != (-1, -1):
                continue
            route = routes[pod_id]
            current_cost = len(route) - 1 + waiting(pod_id, route[:2])
            carrying = any(route[:2] in wanted_edges[route[0], passenger.kind] for passenger in queues.get(route[0], ()))
            alternatives = []
            for neighbor in graph[route[0]]:
                candidate = graph_route(graph, route[0], route[-1], neighbor)
                if route[0] in candidate[1:] or carrying and len(candidate) != len(route):
                    continue
                cost = len(candidate) - 1 + waiting(pod_id, candidate[:2])
                if cost < current_cost:
                    alternatives.append((cost, candidate))
            if alternatives:
                routes[pod_id] = min(alternatives)[1]
                schedule, attempts = self.project_routes(routes, state.tubes, DAYS - day)
        for pod_id in sorted(assignments):
            if pending[pod_id] != (-1, -1):
                continue
            conflicts = any(pod_id in pods and sum(other <= pod_id or other in assigned for other in pods) > state.tubes[edge]
                for requests in attempts for edge, pods in requests.items())
            if conflicts and (alternative := free_route(pod_id, routes[pod_id])):
                routes[pod_id] = alternative
                schedule, attempts = self.project_routes(routes, state.tubes, DAYS - day)
    def project_routes(self, routes: dict, tubes: dict, days: int) -> tuple:
        """Projects routes through tubes for days; returns daily accepted and attempted edge usage."""
        positions = dict.fromkeys(routes, 0)
        schedule, attempts = [], []
        for _ in range(days):
            occupied, requests = {}, {}
            for pod_id in sorted(routes):
                route = routes[pod_id]
                index = positions[pod_id]
                if index == len(route) - 1:
                    continue
                edge = route_key(*route[index:index + 2])
                requests.setdefault(edge, []).append(pod_id)
                if len(occupied.get(edge, ())) < tubes[edge]:
                    occupied.setdefault(edge, []).append(pod_id)
                    positions[pod_id] += 1
            schedule.append(occupied)
            attempts.append(requests)
        return schedule, attempts
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
def load_id(path):
    return path.nodes
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
def graph_route(graph, start_id, finish_id, first_id):
    path = [start_id, first_id]
    while path[-1] != finish_id:
        path.append(next_step(graph, path[-1], finish_id))
    return tuple(path)
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
