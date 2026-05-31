"""Evaluates one Selenia City command by reusing the regression test helpers."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from Selenia_City.main import Planner
from Selenia_City.unit_tests import apply_actions, parse_turn_state

TURN_STATE = \
"""
month 7
resources 2035
module 0 1 106 9
landing 1 104 37 1:20
module 2 2 148 10
landing 3 47 13 1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2
module 4 3 91 19
landing 5 46 66 1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3
module 6 4 110 43
landing 7 28 10 1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2
module 8 5 159 46
landing 9 108 85 1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3
module 10 6 20 50
landing 11 95 74 1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6
module 12 7 10 88
landing 13 64 23 1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4
tube 0 1 1
tube 0 2 1
tube 0 3 1
tube 0 4 1
tube 0 7 1
tube 1 6 1
tube 1 11 1
tube 4 5 1
tube 5 10 1
tube 6 8 1
tube 6 9 1
pod 1 1-0-1-0-1-6-1-0-1-0-1-6-1-0-1-0-1-0-1-0-1
pod 2 3-0-3-0-3-0-3-0-3-0-3-0-3-0-3-0-3-0-3-0-3
pod 3 5-4-5-4-0-4-0-4-5-4-0-4-5-10-5-4-0-4-0-4-0
pod 4 7-0-7-0-2-0-2-0-7-0-2-0-2-0-2-0-2-0-2-0-2
pod 5 9-6-9-6-8-6-9-6-8-6-9-6-9-6-8-6-9-6-8-6-9
pod 6 11-1-11-1-11-1-0-1-0-1-0-1-0-1-0-1-0-1-0-1-0
"""
# OUTPUT_COMMAND = "DESTROY 1;TUBE 1 4;TUBE 7 13;POD 1 1 0 1 0 1 0 1 4 1 6 1 6 1 4 1 6 1 0 1 0 1;POD 7 13 7 13 7 0 7 0 7 13 7 0 7 0 7 0 7 0 7 0 7 0"
OUTPUT_COMMAND = "TUBE 1 4; TUBE 4 13; TUBE 5 12; DESTROY 3; POD 3 AUTO(5-12, 5-10, 4-5); DESTROY 2; POD 2 AUTO(0-3, 0-4, 4-13); DESTROY 6; POD 6 AUTO(1-11, 0-1, 1-4); DESTROY 1; POD 1 AUTO(0-1, 1-6)"
# OUTPUT_COMMAND = "TUBE 3 7; TUBE 0 6; TUBE 3 4; POD 4 AUTO(3-7, 3-4); DESTROY 2; POD 2 AUTO(0-3, 0-6)"


def print_score_after_command():
    """Prints resources and score after applying OUTPUT_COMMAND to TURN_STATE."""
    planner = parse_turn_state(TURN_STATE)
    starting_resources = planner.resources
    reason, resolved_actions = apply_command_with_auto(planner, OUTPUT_COMMAND)
    for action in resolved_actions:
        print(f"resolved_auto {action}")
    if reason:
        print(f"impossible: {reason}")
        return
    planned_pods = {pod_id: pod.path[:] for pod_id, pod in planner.pods.items()}
    score_text = planner.score_debug_text("after", planned_pods, dict(planner.teleports), dict(planner.tubes))
    print(f"resources_after {planner.resources} spent {starting_resources - planner.resources} {score_text}")


def apply_command_with_auto(planner: Planner, command: str) -> tuple[str, list[str]]:
    """Applies command to planner while expanding AUTO pod route syntax."""
    actions = [action.strip() for action in command.split(";") if action.strip()]
    context = deepcopy(planner)
    autos = []
    expanded_actions = []
    for action in actions:
        auto = parse_auto_action(action)
        if auto:
            autos.append((len(expanded_actions), *auto));expanded_actions.append("")
            continue
        reason = apply_actions(context, action)
        if reason:
            return reason, []
        expanded_actions.append(action)
    resolved_actions = []
    try:resolved_paths = context.resolve_auto_routes([(pod_id,edges)for(_,_,pod_id,edges)in autos])
    except ValueError as error:return f"AUTO: {error}", resolved_actions
    for index, action, pod_id, edges in autos:
        path = resolved_paths[pod_id]
        expanded_actions[index] = "POD {} {}".format(pod_id, " ".join(map(str, path)))
        resolved_actions.append(expanded_actions[index])
    for action in expanded_actions:
        reason = apply_actions(planner, action)
        if reason:
            return reason, resolved_actions
    return "", resolved_actions


def parse_auto_action(action: str) -> tuple[str, int, list[tuple[int, int]]]:
    """Returns AUTO action metadata for pod_id and service edges, or an empty tuple."""
    parts = action.split(maxsplit=2)
    if len(parts) != 3 or parts[0] != "POD" or not parts[2].strip().startswith("AUTO("):
        return ()
    return action, int(parts[1]), parse_auto_edges(parts[2].strip())


def parse_auto_edges(text: str) -> list[tuple[int, int]]:
    """Parses AUTO edge list text into unordered service edge endpoint pairs."""
    if not text.endswith(")"):
        raise ValueError(f"malformed AUTO route {text}")
    edges = []
    for item in text[5:-1].split(","):
        edge_parts = item.strip().split("-")
        if len(edge_parts) != 2:
            raise ValueError(f"malformed AUTO edge {item.strip()}")
        edges.append((int(edge_parts[0]), int(edge_parts[1])))
    return edges


if __name__ == "__main__":
    print_score_after_command()
