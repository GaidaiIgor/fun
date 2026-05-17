"""Evaluates one Selenia City command by reusing the planner implementation."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from Selenia_City.main import MAX_PODS, MAX_TUBES_PER_BUILDING, POD_COST, POD_REFUND, TELEPORT_COST, Building, Planner, Pod, route_key, tube_cost

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
    if not apply_actions(planner, OUTPUT_COMMAND):
        print("impossible")
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


def apply_actions(planner: Planner, text: str) -> bool:
    """Applies a semicolon-separated output command to the planner state."""
    for action in text.split(";"):
        parts = action.strip().split()
        if not parts or parts[0] == "WAIT":
            continue
        match parts[0]:
            case "TUBE":
                ok = apply_tube(planner, int(parts[1]), int(parts[2]))
            case "UPGRADE":
                ok = apply_upgrade(planner, int(parts[1]), int(parts[2]))
            case "TELEPORT":
                ok = apply_teleport(planner, int(parts[1]), int(parts[2]))
            case "POD":
                ok = apply_pod(planner, int(parts[1]), [int(item) for item in parts[2:]])
            case "DESTROY":
                ok = apply_destroy(planner, int(parts[1]))
            case _:
                return False
        if not ok:
            return False
    return True


def apply_tube(planner: Planner, a: int, b: int) -> bool:
    """Builds one tube if main planner rules consider it valid and affordable."""
    cost = tube_cost(planner.buildings[a], planner.buildings[b])
    degrees = planner.get_tube_degrees()
    if cost > planner.resources or degrees[a] >= MAX_TUBES_PER_BUILDING or degrees[b] >= MAX_TUBES_PER_BUILDING \
            or not planner.can_build_tube(a, b, planner.tubes, []):
        return False
    planner.resources -= cost
    planner.tubes[route_key(a, b)] = 1
    return True


def apply_upgrade(planner: Planner, a: int, b: int) -> bool:
    """Upgrades one existing tube if it is affordable."""
    key = route_key(a, b)
    if key not in planner.tubes:
        return False
    cost = tube_cost(planner.buildings[a], planner.buildings[b]) * (planner.tubes[key] + 1)
    if cost > planner.resources:
        return False
    planner.resources -= cost
    planner.tubes[key] += 1
    return True


def apply_teleport(planner: Planner, a: int, b: int) -> bool:
    """Builds one teleporter if its endpoints are free and affordable."""
    used = planner.get_teleport_used_buildings()
    if a == b or TELEPORT_COST > planner.resources or a in used or b in used:
        return False
    planner.resources -= TELEPORT_COST
    planner.teleports[a] = b
    return True


def apply_pod(planner: Planner, pod_id: int, path: list[int]) -> bool:
    """Builds one pod if the itinerary uses existing tubes and is affordable."""
    if POD_COST > planner.resources or not 1 <= pod_id <= MAX_PODS or pod_id in planner.pods or len(path) < 2:
        return False
    if not all(route_key(a, b) in planner.tubes for a, b in zip(path, path[1:])):
        return False
    planner.resources -= POD_COST
    planner.pods[pod_id] = Pod(pod_id, path)
    return True


def apply_destroy(planner: Planner, pod_id: int) -> bool:
    """Destroys one existing pod and refunds resources."""
    if pod_id not in planner.pods:
        return False
    del planner.pods[pod_id]
    planner.resources += POD_REFUND
    return True


if __name__ == "__main__":
    print_score_after_command()
