"""Runs the semi solver against a local text turn state without CodinGame stderr limits."""
from __future__ import annotations

import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from Selenia_City.semi import Building, Planner, Pod, route_key

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
    """Parses resources, buildings, routes, pods, and service areas from text."""
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
                pod_id, service_area, path = parse_pod_line(line)
                planner.pods[pod_id] = Pod(pod_id, path)
                planner.service_areas[pod_id] = service_area
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


def parse_pod_line(line: str) -> tuple[int, set[tuple[int, int]], list[int]]:
    """Parses pod id, service area, and path from inline pod line."""
    pod_text, path_text = line.removeprefix("pod ").split(", path=[")
    id_text, area_text = pod_text.split(", service={")
    pod_id = int(id_text.removeprefix("id="))
    return pod_id, parse_service_area(area_text.removesuffix("}")), parse_path(path_text.removesuffix("]"))


def parse_service_area(text: str) -> set[tuple[int, int]]:
    """Parses service-area text written as comma-separated a-b edges."""
    area = set()
    for item in text.split(", "):
        edge_parts = item.split("-")
        area.add(route_key(int(edge_parts[0]), int(edge_parts[1])))
    return area


if __name__ == "__main__":
    run_turn_state(TURN_STATE)
