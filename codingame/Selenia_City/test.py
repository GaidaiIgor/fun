"""Evaluates one Selenia City command by reusing the regression test helpers."""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from Selenia_City.unit_tests import apply_actions, parse_turn_state

TURN_STATE = \
"""
month 3
resources 1865
module 0 1 106 9
landing 1 104 37 1:20
module 2 2 148 10
landing 3 47 13 1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2
module 4 3 91 19
landing 5 46 66 1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3
tube 0 1 1
tube 0 2 1
tube 0 3 1
pod 1 1-0-1-0-2-0-2-0-1
pod 2 3-0-3
"""
OUTPUT_COMMAND = "TUBE 4 5; TUBE 0 4; POD 3 5 4 5 4 0 4 5 4 0 4 5"


def print_score_after_command():
    """Prints resources and score after applying OUTPUT_COMMAND to TURN_STATE."""
    planner = parse_turn_state(TURN_STATE)
    starting_resources = planner.resources
    reason = apply_actions(planner, OUTPUT_COMMAND)
    if reason:
        print(f"impossible: {reason}")
        return
    planned_pods = {pod_id: pod.path[:] for pod_id, pod in planner.pods.items()}
    score_text = planner.score_debug_text("after", planned_pods, dict(planner.teleports), dict(planner.tubes))
    print(f"resources_after {planner.resources} spent {starting_resources - planner.resources} {score_text}")


if __name__ == "__main__":
    print_score_after_command()
