"""Tests Selenia City planner regressions against benchmark moves."""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr
from io import StringIO
from time import perf_counter
import unittest

from Selenia_City.main import MAX_PODS, MAX_TUBES_PER_BUILDING, POD_COST, POD_REFUND, TELEPORT_COST
from Selenia_City.main import Building, Planner, Pod, point_on_segment, route_key, segments_intersect, tube_cost


class PlannerRegressionTests(unittest.TestCase):
    """Checks that planner output scores at least as well as known benchmark moves."""

    def test_two_destination_pad_builds_full_initial_service(self):
        """Verifies a two-type landing pad gets a full-service initial network."""
        state = """
month 1
resources 3000
landing 0 30 20 1:25,2:25
module 1 1 130 20
module 2 2 130 70
"""
        benchmark_move = "TUBE 0 1; TUBE 1 2; POD 1 0 1 0 1 0 1 0 1 0 1 2 1 2 1 2"
        benchmark_score = 4045
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_existing_staged_pod_can_be_split_into_edge_pods(self):
        """Verifies selling a staged pod can fund more efficient dedicated edge pods."""
        state = """
month 2
resources 1550
landing 0 30 20 1:25,2:25
module 1 1 130 20
module 2 2 130 70
tube 0 1 1
tube 1 2 1
pod 1 0-1-0-1-0-1-0-1-0-1-2-1-2-1-2
"""
        benchmark_move = "DESTROY 1; POD 1 0 1 0; POD 2 2 1 2"
        benchmark_score = 4125
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_loaded_shared_module_should_not_receive_balanced_extra_load(self):
        """Verifies route cadence accounts for load already assigned to a shared module."""
        state = """
month 1
resources 5000
module 0 1 20 15
module 1 2 140 15
landing 2 40 45 1:50
landing 3 80 45 1:25,2:25
landing 4 120 45 2:50
module 5 2 20 75
module 6 1 140 75
"""
        benchmark_move = "TUBE 2 0; POD 1 2 0 2; TUBE 4 1; POD 2 4 1 4; TUBE 3 5; TUBE 3 2; POD 3 3 5 3 5 3 5 3 2 3 2 3 2"
        benchmark_score = 10120
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_reroutes_idle_existing_pod_to_downstream_transfer(self):
        """Verifies an existing pod can be rerouted to share downstream transfer work."""
        state = """
month 2
resources 2502
module 0 1 106 9
landing 1 104 37 1:20
module 2 2 148 10
landing 3 47 13 1:11,2:11
tube 0 1 1
pod 1 1-0-1
"""
        benchmark_move = "TUBE 0 3; TUBE 0 2; POD 2 3 0 3; DESTROY 1; POD 1 1 0 1 0 2 0 2 0"
        benchmark_score = 3579
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_loop_closure_uses_shortest_return_path(self):
        """Verifies loop closure does not repeat useful work after the last delivery."""
        state = """
month 2
resources 2502
module 0 1 106 9
landing 1 104 37 1:20
module 2 2 148 10
landing 3 47 13 1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2
tube 0 1 1
pod 1 1-0-1
"""
        benchmark_move = "DESTROY 1;TUBE 0 2;TUBE 3 0;POD 1 1 0 1 0 2 0 2 0 1;POD 2 3 0 3"
        benchmark_score = score_command(state, benchmark_move)
        planner_move = choose_planner_command(state)
        self.assertIn("POD 1 1 0 1 0 2 0 2 0 1", planner_move)
        self.assertNotIn("POD 1 1 0 1 0 2 0 2 0 2 0 2 0 1 0 1", planner_move)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_two_hop_route_bundles_passengers_before_downstream_trip(self):
        """Verifies two-hop routes can repeat feeder trips to improve pod utilization."""
        state = """
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
        benchmark_move = "TUBE 4 5; TUBE 0 4; POD 3 5 4 5 4 0 4 5 4 0 4 5"
        benchmark_score = 5413
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_transfer_route_connects_missing_module_type(self):
        """Verifies one route can connect a pad to existing network and a missing module type."""
        state = transfer_route_state()
        benchmark_move = "TUBE 5 7; TUBE 5 6; POD 4 7 5 6 5 7"
        benchmark_score = 7185
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_transfer_route_search_beats_recorded_timeout_baseline(self):
        """Verifies this timeout-prone turn plans under 400 milliseconds."""
        state = transfer_route_state()
        target_seconds = 0.4
        best_seconds = max(timed_planner_run(state) for _ in range(3))
        self.assertLess(best_seconds, target_seconds)

    def test_connects_new_island_by_replacing_multiple_service_pods(self):
        """Verifies new disconnected buildings can be connected by rerouting a variable pod bundle."""
        state = new_island_state()
        benchmark_move = "TUBE 5 9; TUBE 8 9; DESTROY 1; DESTROY 2; " \
            "POD 1 1 0 1 0 3 0 2 0 3 0 2 0 3 0 2 0 2 0 2 0 2 0; POD 2 9 5 9 5 9 8 9 5 9"
        benchmark_score = 8791
        self.assertEqual(score_command(state, benchmark_move), benchmark_score)
        planner_move = choose_planner_command(state)
        self.assertGreaterEqual(score_command(state, planner_move), benchmark_score)

    def test_new_island_replacement_search_beats_recorded_timeout_baseline(self):
        """Verifies this new-island replacement turn plans under 400 milliseconds."""
        state = new_island_state()
        target_seconds = 0.4
        best_seconds = max(timed_planner_run(state) for _ in range(3))
        self.assertLess(best_seconds, target_seconds)

    def test_same_pad_boarding_uses_input_passenger_order(self):
        """Verifies same-pad passengers board by input order instead of type order."""
        state = """
month 1
resources 0
landing 0 0 0 2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1
module 1 1 20 0
module 2 2 20 10
landing 3 10 0 none
tube 0 3 1
tube 3 1 1
tube 3 2 1
pod 1 0-3-0
pod 2 3-2-3
"""
        self.assertEqual(score_command(state, "WAIT"), 925)


def choose_planner_command(state: str) -> str:
    """Returns the planner command for a compact turn-state string."""
    planner = parse_turn_state(state)
    with redirect_stderr(StringIO()):
        actions = planner.choose_actions()
    command = ";".join(actions) or "WAIT"
    assert_looped_planner_pods(command)
    return command


def timed_planner_run(state: str) -> float:
    """Returns elapsed seconds for one planner command on state."""
    start = perf_counter()
    choose_planner_command(state)
    return perf_counter() - start


def transfer_route_state() -> str:
    """Returns the month-four transfer-route state that triggered timeout risk."""
    return """
month 4
resources 2338
module 0 1 106 9
landing 1 104 37 1:20
module 2 2 148 10
landing 3 47 13 1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2
module 4 3 91 19
landing 5 46 66 1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3
module 6 4 110 43
landing 7 28 10 1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2
tube 0 1 1
tube 0 2 1
tube 0 3 1
tube 0 4 1
tube 4 5 1
pod 1 1-0-1-0-2-0-2-0-1
pod 2 3-0-3
pod 3 5-4-5-4-0-4-5-4-0-4-5
"""


def new_island_state() -> str:
    """Returns the month-five new-island state that requires replacing multiple pods."""
    return """
month 5
resources 1877
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
tube 0 1 1
tube 0 2 1
tube 0 3 1
tube 0 4 1
tube 4 5 1
tube 5 6 1
tube 5 7 1
pod 1 1-0-1-0-2-0-2-0-1
pod 2 3-0-3
pod 3 5-4-5-4-0-4-5-4-0-4-5
pod 4 7-5-7-5-6-5-7-5-6-5-7
"""


def assert_looped_planner_pods(command: str):
    """Checks every planner-created pod route ends at its starting node."""
    for action in command.split(";"):
        parts = action.strip().split()
        if parts and parts[0] == "POD":
            path = [int(item) for item in parts[2:]]
            if path[0] != path[-1]:
                raise AssertionError(f"unlooped planner pod route: {action}")


def score_command(state: str, command: str) -> int:
    """Returns the exact monthly score after applying a command to a compact state."""
    planner = parse_turn_state(state)
    error = apply_actions(planner, command)
    if error is not None:
        raise AssertionError(error)
    planned_pods = {pod_id: pod.path[:] for pod_id, pod in planner.pods.items()}
    return planner.actual_score_from_pods(planned_pods, dict(planner.teleports), dict(planner.tubes))[0]


def parse_turn_state(text: str) -> Planner:
    """Parses compact debug-style turn state into a Planner instance."""
    planner = Planner()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("month ") or line.startswith("pair_cost ") or line.startswith("score_") or line.startswith("resources_after "):
            continue
        parts = line.split()
        match parts[0]:
            case "resources":
                planner.resources = int(parts[1])
            case "module":
                planner.buildings[int(parts[1])] = Building(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            case "landing":
                demand, order = parse_demand(parts[4])
                planner.buildings[int(parts[1])] = Building(int(parts[1]), 0, int(parts[2]), int(parts[3]), demand, order)
            case "tube":
                planner.tubes[route_key(int(parts[1]), int(parts[2]))] = int(parts[3])
            case "teleport":
                planner.teleports[int(parts[1])] = int(parts[2])
            case "pod":
                planner.pods[int(parts[1])] = Pod(int(parts[1]), [int(item) for item in " ".join(parts[2:]).replace("-", " ").split()])
    return planner


def parse_demand(text: str) -> tuple[Counter[int], list[int]]:
    """Parses comma-separated astronaut demand into counts by type."""
    demand = Counter()
    order = []
    if text == "none":
        return demand, order
    for item in text.split(","):
        if ":" in item:
            astronaut_type, count = item.split(":")
            order.extend([int(astronaut_type)] * int(count))
        else:
            order.append(int(item))
    demand.update(order)
    return demand, order


def apply_actions(planner: Planner, text: str) -> str:
    """Applies a semicolon-separated command string to planner state."""
    for action in text.split(";"):
        cleaned = action.strip()
        parts = cleaned.split()
        if not parts or parts[0] == "WAIT":
            continue
        match parts[0]:
            case "TUBE":
                error = apply_tube(planner, int(parts[1]), int(parts[2]))
            case "UPGRADE":
                error = apply_upgrade(planner, int(parts[1]), int(parts[2]))
            case "TELEPORT":
                error = apply_teleport(planner, int(parts[1]), int(parts[2]))
            case "POD":
                error = apply_pod(planner, int(parts[1]), [int(item) for item in parts[2:]])
            case "DESTROY":
                error = apply_destroy(planner, int(parts[1]))
            case _:
                return f"{cleaned}: unknown action {parts[0]}"
        if error is not None:
            return f"{cleaned}: {error}"
    return None


def apply_tube(planner: Planner, a: int, b: int) -> str:
    """Builds one tube if geometry, degree, and resource rules allow it."""
    cost = tube_cost(planner.buildings[a], planner.buildings[b])
    error = tube_rule_error(planner, a, b)
    if error is not None:
        return error
    if cost > planner.resources:
        return f"tube costs {cost}, resources {planner.resources}"
    planner.resources -= cost
    planner.tubes[route_key(a, b)] = 1
    return None


def tube_rule_error(planner: Planner, a: int, b: int) -> str:
    """Explains why a tube cannot be built, or returns None when it is legal."""
    degrees = planner.get_tube_degrees()
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


def apply_upgrade(planner: Planner, a: int, b: int) -> str:
    """Upgrades an existing tube if resources allow it."""
    key = route_key(a, b)
    if key not in planner.tubes:
        return "tube does not exist"
    cost = tube_cost(planner.buildings[a], planner.buildings[b]) * (planner.tubes[key] + 1)
    if cost > planner.resources:
        return f"upgrade costs {cost}, resources {planner.resources}"
    planner.resources -= cost
    planner.tubes[key] += 1
    return None


def apply_teleport(planner: Planner, a: int, b: int) -> str:
    """Builds a teleporter if endpoints and resources allow it."""
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


def apply_pod(planner: Planner, pod_id: int, path: list[int]) -> str:
    """Builds a pod if the id, path, tubes, and resources allow it."""
    if not 1 <= pod_id <= MAX_PODS:
        return f"pod id {pod_id} is outside 1..{MAX_PODS}"
    if pod_id in planner.pods:
        return f"pod id {pod_id} already exists"
    if len(path) < 2:
        return "pod path has fewer than 2 stops"
    missing = next((route_key(a, b) for a, b in zip(path, path[1:]) if route_key(a, b) not in planner.tubes), None)
    if missing is not None:
        return f"missing tube {missing[0]}-{missing[1]}"
    if POD_COST > planner.resources:
        return f"pod costs {POD_COST}, resources {planner.resources}"
    planner.resources -= POD_COST
    planner.pods[pod_id] = Pod(pod_id, path)
    return None


def apply_destroy(planner: Planner, pod_id: int) -> str:
    """Destroys an existing pod and refunds resources."""
    if pod_id not in planner.pods:
        return f"pod id {pod_id} does not exist"
    del planner.pods[pod_id]
    planner.resources += POD_REFUND
    return None


if __name__ == "__main__":
    unittest.main()
