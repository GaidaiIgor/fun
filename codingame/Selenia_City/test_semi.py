"""Runs the semi solver against a local text turn state without CodinGame stderr limits."""
from __future__ import annotations

import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import Selenia_City.semi as semi
from Selenia_City.semi import Building, Candidate, Planner, PlanState, Pod, SimulationResult, route_key

semi.FULL_DEBUG = True

def table_debug(self, result: SimulationResult, state: PlanState) -> str:
    """Formats result assignment rows using state pod headers."""
    headers = ["Day", "Loads", *("P{}{}".format(pod_id, "f" if not pod.dynamic else "")
        for pod_id, pod in sorted(state.pods.items()))]
    rows = [headers, *(row[:] for row in result.table)]
    for column in range(2, len(headers)):
        path_width = max((len(row[column].rsplit(" (", 1)[0]) for row in rows[1:]), default=0)
        for row in rows[1:]:
            path, location = row[column].rsplit(" (", 1)
            row[column] = f"{path.ljust(path_width)} ({location}"
    widths = [max(map(len, column)) for column in zip(*rows)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    table = [border, lines[0], border, *lines[1:]]
    table.extend([border] if len(lines) > 1 else [])
    return "Fixed reservations: " + result.reserved + "\nAssignments:\n" + "\n".join(table)


def score_debug(self, label: str, result: SimulationResult, cost: int) -> str:
    """Formats label score from result using cost."""
    demand = sum(sum(pad.demand.values()) for pad in self.landing_pads())
    stats = self.status_debug(result)
    if label != "after":
        landings = []
        for pad in self.landing_pads():
            demand_text = ", ".join(f"{kind}x{count}" for kind, count in sorted(pad.demand.items()))
            landings.append(f"landing {pad.id}: {demand_text}")
        iteration = ("Iteration 1",) if label == "before" else ()
        return "\n".join((*landings, "", *iteration, stats))
    return f"After: speed {result.speed}, diversity {result.diversity}, delivered {result.delivered}/{demand}, " \
        f"score: {result.score}, resources: {self.resources - cost}\n{stats}"


def status_debug(self, result: SimulationResult) -> str:
    """Formats all pool status from result."""
    return f"{self.pool_debug(result)}\n{self.diversity_debug(result)}"


def pool_debug(self, result: SimulationResult) -> str:
    """Formats speed-pool status from result."""
    lines = []
    for pool in self.speed_pools():
        pad_id, kind = pool
        max_speed = self.buildings[pad_id].demand[kind] * 50
        delivery_time = result.delivery_times[pool] if pool in result.delivery_times else "-"
        lines.append(f"speed pool {pool}: {result.speed_by_pool[pool]}/{max_speed}, delivery {delivery_time}")
    return "\n".join(lines)


def diversity_debug(self, result: SimulationResult) -> str:
    """Formats diversity-pool status from result."""
    lines = []
    for building in sorted(self.buildings.values(), key=lambda item: item.id):
        if building.kind <= 0:
            continue
        max_diversity = self.max_diversity(building.kind)
        if not max_diversity:
            continue
        perfect_diversity = self.perfect_diversity(building.kind)
        line = f"diversity pool {building.id}: {result.diversity_by_module[building.id]}/{perfect_diversity}/{max_diversity}, "
        lines.append(f"{line}delivered {result.delivered_by_module[building.id]}")
    return "\n".join(lines)


def state_delta_text(self, before: PlanState, after: PlanState) -> str:
    """Formats infrastructure and pod changes between before and after."""
    actions = []
    for edge in sorted(before.new_tubes - after.new_tubes):
        actions.append(f"DROP TUBE {edge[0]} {edge[1]}")
    new_edges = sorted(after.new_tubes - before.new_tubes)
    for edge in new_edges:
        actions.append(f"TUBE {edge[0]} {edge[1]}")
    for edge in new_edges:
        actions.extend(f"UPGRADE {edge[0]} {edge[1]}" for _ in range(after.tubes[edge] - 1))
    for edge in sorted(set(before.tubes) & set(after.tubes)):
        for _ in range(after.tubes[edge] - before.tubes[edge]):
            actions.append(f"UPGRADE {edge[0]} {edge[1]}")
        for _ in range(before.tubes[edge] - after.tubes[edge]):
            actions.append(f"DROP UPGRADE {edge[0]} {edge[1]}")
    for entrance_id in sorted(set(before.teleports) - set(after.teleports)):
        actions.append(f"DROP TELEPORT {entrance_id} {before.teleports[entrance_id]}")
    for entrance_id in sorted(set(after.teleports) - set(before.teleports)):
        actions.append(f"TELEPORT {entrance_id} {after.teleports[entrance_id]}")
    for pod_id in sorted(before.ops - after.ops):
        actions.append(f"REVERT POD {pod_id}" if pod_id in self.pods else f"DROP POD {pod_id}")
    for pod_id in sorted(before.ops & after.ops):
        if before.routes[pod_id] != after.routes[pod_id] or before.pairs[pod_id] != after.pairs[pod_id]:
            actions.append(f"POD {pod_id} AUTO")
    for pod_id in sorted(after.ops - before.ops):
        actions.append(f"POD {pod_id} AUTO")
    return ";".join(actions) if actions else "WAIT"


def state_action_text(self, state: PlanState) -> str:
    """Formats the complete planned infrastructure and pod actions in state."""
    actions = [action for action in state.actions if action and action.split()[0] in ("TUBE", "TELEPORT", "UPGRADE")]
    actions.extend(f"POD {pod_id} AUTO" for pod_id in sorted(state.ops))
    return ";".join(actions) if actions else "WAIT"


def selected_debug(self, best: Candidate, state: PlanState, result: SimulationResult, before_score: int):
    """Prints the selected branch and resulting grand-total plan."""
    score_gain = result.score - before_score
    path_text = ", ".join(map(str, best.bundle.path))
    text = f"selected: pair={best.pair}, path=[{path_text}], bundle={best.number}, actions={self.state_action_text(state)}, "
    semi.debug(f"{text}gain={score_gain}, cost={state.cost}, efficiency={score_gain / max(1, state.cost):.3f}, "
        f"resources left={self.resources - state.cost}")


Planner.table_debug = table_debug
Planner.score_debug = score_debug
Planner.status_debug = status_debug
Planner.pool_debug = pool_debug
Planner.diversity_debug = diversity_debug
Planner.state_delta_text = state_delta_text
Planner.state_action_text = state_action_text
Planner.selected_debug = selected_debug

TURN_STATE = """
month 10
resources 5324
module 0 1 20 15
module 1 2 140 15
landing 2 40 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
landing 3 80 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
landing 4 120 45 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
module 5 2 20 75
module 6 1 140 75
module 7 3 10 45
landing 8 150 45 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
tube 0 2 1
tube 1 4 1
tube 2 3 1
tube 3 4 1
tube 3 5 1
tube 3 6 1
pod id=1, path=[2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 3, 6, 3, 5, 3, 6, 3, 2, 0, 2]
pod id=2, path=[3, 6, 3, 5, 3, 6, 3, 5, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1]
"""


def run_turn_state(text: str) -> str:
    """Parses text into a Planner, runs debug printing and choose_actions, and returns the command."""
    planner = parse_turn_state(text)
    with redirect_stderr(sys.stdout):
        planner.print_debug_input()
        actions = planner.choose_actions()
    command = ";".join(actions) if actions else "WAIT"
    print(command)
    return command


def parse_turn_state(text: str) -> Planner:
    """Parses resources, buildings, routes, pods, and pod itineraries from text."""
    planner = Planner()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        match parts[0]:
            case "month":
                planner.month = int(parts[1]) - 1
            case "resources":
                planner.resources = int(parts[1])
            case "landing":
                demand, order = parse_demand("".join(parts[4:]))
                planner.buildings[int(parts[1])] = Building(int(parts[1]), 0, int(parts[2]), int(parts[3]), demand, order)
            case "module":
                planner.buildings[int(parts[1])] = Building(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            case "tube":
                planner.tubes[route_key(int(parts[1]), int(parts[2]))] = int(parts[3])
            case "teleport":
                planner.teleports[int(parts[1])] = int(parts[2])
            case "pod":
                pod_id, path = parse_pod_line(line)
                planner.pods[pod_id] = Pod(pod_id, path)
            case _:
                raise ValueError(f"Unknown turn-state line: {line}")
    return planner


def parse_demand(text: str) -> tuple[Counter[int], list[int]]:
    """Parses none, comma-separated kinds, or kind:count demand text."""
    demand = Counter()
    order = []
    if text == "none":
        return demand, order
    for item in text.split(","):
        if ":" in item:
            kind_text, count_text = item.split(":")
            order.extend([int(kind_text)] * int(count_text))
        else:
            order.append(int(item))
    demand.update(order)
    return demand, order


def parse_path(text: str) -> list[int]:
    """Parses a pod path text written with commas or spaces."""
    return [int(item) for item in text.replace(",", " ").split()]


def parse_pod_line(line: str) -> tuple[int, list[int]]:
    """Parses pod id and itinerary from line."""
    id_text, path_text = line.removeprefix("pod id=").split(", path=[")
    return int(id_text), parse_path(path_text.removesuffix("]"))


if __name__ == "__main__":
    run_turn_state(TURN_STATE)
