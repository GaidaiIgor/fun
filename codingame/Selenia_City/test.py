"""Evaluates one Selenia City command by reusing the planner implementation."""

from __future__ import annotations

from collections import Counter

from Selenia_City.main import MAX_PODS, POD_COST, POD_REFUND, TELEPORT_COST, Building, Planner, Pod, route_key, tube_cost

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
    """Prints the planner-estimated score after applying OUTPUT_COMMAND to TURN_STATE."""
    planner = parse_turn_state(TURN_STATE)
    apply_actions(planner, OUTPUT_COMMAND)
    score, speed, diversity, delivered, _, _ = planner.score_from_pods({pod_id: pod.path[:] for pod_id, pod in planner.pods.items()},
                                                                       dict(planner.teleports))
    demand = sum(sum(building.demand.values()) for building in planner.buildings.values() if building.kind == 0)
    print(f"score_after total {score} speed {speed} diversity {diversity} delivered {delivered}/{demand} stranded {demand - delivered}")


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


def apply_actions(planner: Planner, text: str):
    """Applies a semicolon-separated output command to the planner state."""
    for action in text.split(";"):
        parts = action.strip().split()
        if not parts or parts[0] == "WAIT":
            continue
        match parts[0]:
            case "TUBE":
                apply_tube(planner, int(parts[1]), int(parts[2]))
            case "UPGRADE":
                apply_upgrade(planner, int(parts[1]), int(parts[2]))
            case "TELEPORT":
                apply_teleport(planner, int(parts[1]), int(parts[2]))
            case "POD":
                apply_pod(planner, int(parts[1]), [int(item) for item in parts[2:]])
            case "DESTROY":
                apply_destroy(planner, int(parts[1]))


def apply_tube(planner: Planner, a: int, b: int):
    """Builds one tube if main planner rules consider it valid and affordable."""
    cost = tube_cost(planner.buildings[a], planner.buildings[b])
    if cost <= planner.resources and planner.can_build_tube(a, b, planner.tubes, []):
        planner.resources -= cost
        planner.tubes[route_key(a, b)] = 1


def apply_upgrade(planner: Planner, a: int, b: int):
    """Upgrades one existing tube if it is affordable."""
    key = route_key(a, b)
    if key in planner.tubes:
        cost = tube_cost(planner.buildings[a], planner.buildings[b]) * (planner.tubes[key] + 1)
        if cost <= planner.resources:
            planner.resources -= cost
            planner.tubes[key] += 1


def apply_teleport(planner: Planner, a: int, b: int):
    """Builds one teleporter if its endpoints are free and affordable."""
    used = planner.get_teleport_used_buildings()
    if TELEPORT_COST <= planner.resources and a not in used and b not in used:
        planner.resources -= TELEPORT_COST
        planner.teleports[a] = b


def apply_pod(planner: Planner, pod_id: int, path: list[int]):
    """Builds or replaces one pod if the itinerary uses existing tubes and is affordable."""
    if POD_COST <= planner.resources and 1 <= pod_id <= MAX_PODS and all(route_key(a, b) in planner.tubes for a, b in zip(path, path[1:])):
        planner.resources -= POD_COST
        planner.pods[pod_id] = Pod(pod_id, path)


def apply_destroy(planner: Planner, pod_id: int):
    """Destroys one existing pod and refunds resources."""
    if pod_id in planner.pods:
        del planner.pods[pod_id]
        planner.resources += POD_REFUND


if __name__ == "__main__":
    print_score_after_command()
