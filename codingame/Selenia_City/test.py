"""Evaluates one Selenia City command by reusing the planner implementation."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from Selenia_City.main import MAX_PODS, MAX_TUBES_PER_BUILDING, POD_COST, POD_REFUND, TELEPORT_COST
from Selenia_City.main import Building, Planner, Pod, point_on_segment, route_key, segments_intersect, tube_cost

TURN_STATE = \
"""
resources 5000
module 0 2 80 75
landing 1 80 15 2:50
landing 2 110 45 1:50
module 3 1 50 45
"""
OUTPUT_COMMAND = "TUBE 1 0;POD 1 1 0 1;TUBE 2 0;TUBE 0 3;POD 2 2 0 2;POD 3 3 0 3"


def print_score_after_command():
    """Prints the main-style resource and score line after applying OUTPUT_COMMAND to TURN_STATE."""
    planner = parse_turn_state(TURN_STATE)
    starting_resources = planner.resources
    reason = apply_actions(planner, OUTPUT_COMMAND)
    if reason:
        print(f"impossible: {reason}")
        return
    planned_pods = {pod_id: pod.path[:] for pod_id, pod in planner.pods.items()}
    score_text = planner.score_debug_text("after", planned_pods, dict(planner.teleports))
    print(f"resources_after {planner.resources} spent {starting_resources - planner.resources} {score_text}")


def parse_turn_state(text: str) -> Planner:
    """Parses compact debug turn-state text into an import-friendly Planner instance."""
    planner = Planner()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("pair_cost ") or line.startswith("score_") or line.startswith("resources_after "):
            continue
        parts = line.split()
        match parts[0]:
            case "resources":
                planner.resources = int(parts[1])
            case "module":
                planner.buildings[int(parts[1])] = Building(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            case "landing":
                planner.buildings[int(parts[1])] = Building(int(parts[1]), 0, int(parts[2]), int(parts[3]), parse_demand(parts[4]))
            case "tube":
                planner.tubes[route_key(int(parts[1]), int(parts[2]))] = int(parts[3])
            case "teleport":
                planner.teleports[int(parts[1])] = int(parts[2])
            case "pod":
                planner.pods[int(parts[1])] = Pod(int(parts[1]), [int(item) for item in " ".join(parts[2:]).replace("-", " ").split()])
    return planner


def parse_demand(text: str) -> Counter[int]:
    """Parses a comma-separated demand list into astronaut-type counts."""
    demand = Counter()
    if text == "none":
        return demand
    for item in text.split(","):
        astronaut_type, count = item.split(":")
        demand[int(astronaut_type)] = int(count)
    return demand


def apply_actions(planner: Planner, text: str) -> str | None:
    """Applies a semicolon-separated output command to the planner state."""
    for action in text.split(";"):
        cleaned = action.strip()
        parts = cleaned.split()
        if not parts or parts[0] == "WAIT":
            continue
        match parts[0]:
            case "TUBE":
                reason = apply_tube(planner, int(parts[1]), int(parts[2]))
            case "UPGRADE":
                reason = apply_upgrade(planner, int(parts[1]), int(parts[2]))
            case "TELEPORT":
                reason = apply_teleport(planner, int(parts[1]), int(parts[2]))
            case "POD":
                reason = apply_pod(planner, int(parts[1]), [int(item) for item in parts[2:]])
            case "DESTROY":
                reason = apply_destroy(planner, int(parts[1]))
            case _:
                return f"{cleaned}: unknown action {parts[0]}"
        if reason:
            return f"{cleaned}: {reason}"
    return None


def apply_tube(planner: Planner, a: int, b: int) -> str | None:
    """Builds one tube if main planner rules consider it valid and affordable."""
    cost = tube_cost(planner.buildings[a], planner.buildings[b])
    degrees = planner.get_tube_degrees()
    reason = tube_rule_error(planner, a, b, degrees)
    if reason:
        return reason
    if cost > planner.resources:
        return f"tube costs {cost}, resources {planner.resources}"
    planner.resources -= cost
    planner.tubes[route_key(a, b)] = 1
    return None


def tube_rule_error(planner: Planner, a: int, b: int, degrees: Counter[int]) -> str | None:
    """Explains why a tube cannot be built before resource checks, or returns None if rules allow it."""
    if a == b:
        return "tube endpoints are identical"
    if route_key(a, b) in planner.tubes:
        return "tube already exists"
    if degrees[a] >= MAX_TUBES_PER_BUILDING:
        return f"building {a} already has {MAX_TUBES_PER_BUILDING} tubes"
    if degrees[b] >= MAX_TUBES_PER_BUILDING:
        return f"building {b} already has {MAX_TUBES_PER_BUILDING} tubes"
    first = planner.buildings[a]
    second = planner.buildings[b]
    for building in planner.buildings.values():
        if building.id not in (a, b) and point_on_segment(building, first, second):
            return f"tube would pass through building {building.id}"
    for c, d in planner.tubes:
        if len({a, b, c, d}) == 4 and segments_intersect(first, second, planner.buildings[c], planner.buildings[d]):
            return f"tube would cross tube {c}-{d}"
    return None


def apply_upgrade(planner: Planner, a: int, b: int) -> str | None:
    """Upgrades one existing tube if it is affordable."""
    key = route_key(a, b)
    if key not in planner.tubes:
        return "tube does not exist"
    cost = tube_cost(planner.buildings[a], planner.buildings[b]) * (planner.tubes[key] + 1)
    if cost > planner.resources:
        return f"upgrade costs {cost}, resources {planner.resources}"
    planner.resources -= cost
    planner.tubes[key] += 1
    return None


def apply_teleport(planner: Planner, a: int, b: int) -> str | None:
    """Builds one teleporter if its endpoints are free and affordable."""
    used = planner.get_teleport_used_buildings()
    if a == b:
        return "teleporter endpoints are identical"
    if a in used:
        return f"building {a} already has a teleporter"
    if b in used:
        return f"building {b} already has a teleporter"
    if TELEPORT_COST > planner.resources:
        return f"teleport costs {TELEPORT_COST}, resources {planner.resources}"
    planner.resources -= TELEPORT_COST
    planner.teleports[a] = b
    return None


def apply_pod(planner: Planner, pod_id: int, path: list[int]) -> str | None:
    """Builds one pod if the itinerary uses existing tubes and is affordable."""
    if not 1 <= pod_id <= MAX_PODS:
        return f"pod id {pod_id} is outside 1..{MAX_PODS}"
    if pod_id in planner.pods:
        return f"pod id {pod_id} already exists"
    if len(path) < 2:
        return "pod path has fewer than 2 stops"
    missing = next((route_key(a, b) for a, b in zip(path, path[1:]) if route_key(a, b) not in planner.tubes), None)
    if missing:
        return f"missing tube {missing[0]}-{missing[1]}"
    if POD_COST > planner.resources:
        return f"pod costs {POD_COST}, resources {planner.resources}"
    planner.resources -= POD_COST
    planner.pods[pod_id] = Pod(pod_id, path)
    return None


def apply_destroy(planner: Planner, pod_id: int) -> str | None:
    """Destroys one existing pod and refunds resources."""
    if pod_id not in planner.pods:
        return f"pod id {pod_id} does not exist"
    del planner.pods[pod_id]
    planner.resources += POD_REFUND
    return None


if __name__ == "__main__":
    print_score_after_command()
