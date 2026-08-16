"""Runs the semi solver against a local text turn state without CodinGame stderr limits."""
from __future__ import annotations

import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import Selenia_City.semi as semi
from Selenia_City.semi import Building, Candidate, Planner, PlanState, PodPlan, SimulationResult, route_key

TURN_STATE = """
month 1
resources 5000
module 0 1 20 15
module 1 2 140 15
landing 2 40 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
landing 3 80 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
landing 4 120 45 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
module 5 2 20 75
module 6 1 140 75
"""

semi.FULL_DEBUG = True

def table_debug(self, result: SimulationResult, state: PlanState) -> str:
    """Formats result assignment rows using state pod headers."""
    headers = ["Day", "Loads", *("P{}{}".format(pod_id, "f" if not pod.dynamic else "")
        for pod_id, pod in sorted(state.pods.items()))]
    initial_assignments = format_assignment_table(headers, result.initial_table)
    assignments = format_assignment_table(headers, result.table)
    edges = sorted(result.congestion_by_edge)
    congestion_rows = [["Day", *(f"{a}-{b}" for a, b in edges)],
        *([str(day + 1), *(str(result.congestion_by_day.get(day, {}).get(edge, 0)) for edge in edges)]
            for day in range(semi.MONTH_DAYS))]
    initial = "\nInitial assignments:\n" + initial_assignments if result.initial_table else ""
    return "Fixed reservations: " + result.reserved + initial + "\nAssignments:\n" + assignments + \
        "\nCongestion:\n" + format_table(congestion_rows)


def format_assignment_table(headers: list[str], data: list[list[str]]) -> str:
    """Formats assignment data under headers and aligns pod locations."""
    rows = [headers, *(row[:] for row in data)]
    for column in range(2, len(headers)):
        path_width = max((len(row[column].rsplit(" (", 1)[0]) for row in rows[1:]), default=0)
        for row in rows[1:]:
            path, location = row[column].rsplit(" (", 1)
            row[column] = f"{path.ljust(path_width)} ({location}"
    return format_table(rows)


def format_table(rows: list[list[str]]) -> str:
    """Formats rows as an aligned table and returns its text."""
    widths = [max(map(len, column)) for column in zip(*rows)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([border, lines[0], border, *lines[1:], border])


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
    speed = sum(result.speed_by_pool.values())
    diversity = sum(result.diversity_by_module.values())
    delivered = sum(result.delivered_by_pool.values())
    return f"After: speed {speed}, diversity {diversity}, delivered {delivered}/{demand}, " \
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


def max_diversity(self, kind: int) -> int:
    """Calculates the theoretical diversity maximum for kind."""
    demand = sum(pad.demand[kind] for pad in self.landing_pads())
    return sum(max(0, 50 - index) for index in range(demand))


def state_action_text(self, state: PlanState, base: PlanState = None) -> str:
    """Formats planned actions in state completely, or as changes relative to base."""
    if base is not None:
        actions = [f"DROP TUBE {edge[0]} {edge[1]}" for edge in sorted(set(base.tubes) - set(state.tubes))]
        actions.extend(f"DROP TELEPORT {source} {target}" for source, target in sorted(base.teleports.items())
            if state.teleports.get(source) != target)
        for edge in sorted(set(state.tubes) - set(base.tubes)):
            actions.append(f"TUBE {edge[0]} {edge[1]}")
            actions.extend(f"UPGRADE {edge[0]} {edge[1]}" for _ in range(state.tubes[edge] - 1))
        actions.extend(f"TELEPORT {source} {target}" for source, target in sorted(state.teleports.items())
            if base.teleports.get(source) != target)
        for edge in sorted(set(state.tubes) & set(base.tubes)):
            difference = state.tubes[edge] - base.tubes[edge]
            command = "UPGRADE" if difference > 0 else "DROP UPGRADE"
            actions.extend(f"{command} {edge[0]} {edge[1]}" for _ in range(abs(difference)))
        actions.extend(f"DROP POD {pod_id}" for pod_id in sorted(set(base.pods) - set(state.pods)))
        actions.extend(f"REVERT POD {pod_id}" for pod_id in sorted((base.ops - state.ops) & set(state.pods)))
        actions.extend(f"REVERT POD {pod_id}" for pod_id in sorted((set(state.pods) - set(base.pods)) - state.ops))
        actions.extend(f"POD {pod_id}" for pod_id in sorted(state.ops - base.ops))
        return ";".join(actions) if actions else "WAIT"
    actions = [action for action in state.actions if action and action.split()[0] in ("TUBE", "TELEPORT", "UPGRADE")]
    actions.extend(f"DROP POD {pod_id}" for pod_id in sorted(set(self.pods) - set(state.pods)))
    actions.extend(f"POD {pod_id}" for pod_id in sorted(state.ops))
    return ";".join(actions) if actions else "WAIT"


def selected_debug(self, best: Candidate, state: PlanState, result: SimulationResult, before_score: int):
    """Prints the selected branch and resulting grand-total plan."""
    path_text = ", ".join(map(str, best.bundle.path))
    text = f"selected: pair={best.pair}, path=[{path_text}], bundle={best.bundle.debug_id}, actions={self.state_action_text(state)}, "
    gains = result.score - before_score, best.marginal_gain, best.round_gain
    costs = state.cost, best.marginal_cost, best.round_cost
    efficiencies = tuple(semi.score_efficiency(gain, cost) for gain, cost in zip(gains, costs))
    gain_text = "/".join(f"{gain:+d}" for gain in gains)
    cost_text = "/".join(f"{cost:+d}" for cost in costs)
    efficiency_text = "/".join(f"{efficiency:+.3f}" for efficiency in efficiencies)
    semi.debug(f"{text}global gains={gain_text}, costs={cost_text}, efficiencies={efficiency_text}, "
        f"resources left={self.resources - state.cost}")


Planner.table_debug = table_debug
Planner.score_debug = score_debug
Planner.status_debug = status_debug
Planner.pool_debug = pool_debug
Planner.diversity_debug = diversity_debug
Planner.max_diversity = max_diversity
Planner.state_action_text = state_action_text
Planner.selected_debug = selected_debug


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
                pod_id, assignments, path = parse_pod_line(line)
                planner.pods[pod_id] = PodPlan(path, False, assignments)
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


def parse_pod_line(line: str) -> tuple[int, list[tuple[int, ...]], list[int]]:
    """Parses pod id, daily assignments and itinerary from line."""
    id_text, values = line.removeprefix("pod id=").split(", assignments=[")
    assignments_text, path_text = values.split("], path=[")
    assignments = [tuple(map(int, item.split("-"))) if item != "-" else () for item in assignments_text.split(", ")]
    return int(id_text), assignments, parse_path(path_text.removesuffix("]"))


if __name__ == "__main__":
    run_turn_state(TURN_STATE)
