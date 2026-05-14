"""Solves CodinGame Dont Panic by planning a route with existing and built elevators."""

from functools import cache
import sys


LEFT = -1
RIGHT = 1
DIRECTIONS = {"LEFT": LEFT, "RIGHT": RIGHT}
ROUND_LOSS_COST = 3
IMPOSSIBLE = 10 ** 9


def debug(label: str, *values: object):
    """Prints compact diagnostic values to stderr."""
    print(f"DEBUG {label}", *values, file=sys.stderr, flush=True)


def nearest_elevator(floor: int, pos: int, step: int, skip_current: bool) -> int | None:
    """Returns the first elevator on floor from pos toward step, optionally ignoring an elevator at pos."""
    if not skip_current and pos in elevators_by_floor[floor]:
        return pos
    if step == RIGHT:
        return min((elevator_pos for elevator_pos in elevators_by_floor[floor] if elevator_pos > pos), default=None)
    return max((elevator_pos for elevator_pos in elevators_by_floor[floor] if elevator_pos < pos), default=None)


def is_between(pos: int, target: int, obstacle: int, step: int) -> bool:
    """Checks whether obstacle blocks movement from pos to target in step direction."""
    return step == RIGHT and pos <= obstacle <= target or step == LEFT and target <= obstacle <= pos


def add_cost(first_cost: tuple[int, int, int], second_cost: tuple[int, int, int]) -> tuple[int, int, int]:
    """Combines route costs whose fields are estimated rounds, sacrificed clones, and built elevators."""
    return first_cost[0] + second_cost[0], first_cost[1] + second_cost[1], first_cost[2] + second_cost[2]


@cache
def best_cost(floor: int, pos: int, direction: int, elevators_left: int, clones_left: int) -> tuple[int, int, int]:
    """Finds the best route cost from a clone state, using available elevators_left and clones_left resources."""
    state = floor, pos, direction, elevators_left, clones_left

    if floor == exit_floor:
        if pos == exit_pos:
            choices[state] = exit_pos, False, state
            return 0, 0, 0
        step = RIGHT if exit_pos > pos else LEFT
        needs_block = direction != step
        obstacle = nearest_elevator(floor, pos, step, needs_block)
        if obstacle is not None and is_between(pos, exit_pos, obstacle, step) or clones_left < needs_block:
            return IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE
        choices[state] = exit_pos, False, state
        return abs(exit_pos - pos) + ROUND_LOSS_COST * needs_block, needs_block, 0

    best = (IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE)
    if pos in elevators_by_floor[floor]:
        best = evaluate_transition(state, best, pos, direction, False, 0)

    for step in (LEFT, RIGHT):
        needs_block = direction != step
        target_pos = nearest_elevator(floor, pos, step, needs_block)
        if target_pos is not None and target_pos != pos:
            best = evaluate_transition(state, best, target_pos, step, False, needs_block)
        if elevators_left > 0:
            best = evaluate_builds(state, best, step, needs_block)

    return best


def evaluate_transition(
        state: tuple[int, int, int, int, int], best: tuple[int, int, int], target_pos: int, next_direction: int, builds: bool, blocks: int
) -> tuple[int, int, int]:
    """Updates best with one transition to target_pos and returns the improved cost tuple."""
    floor, pos, direction, elevators_left, clones_left = state
    losses = blocks + builds
    if clones_left < losses:
        return best
    next_state = floor + 1, target_pos, next_direction, elevators_left - builds, clones_left - losses
    transition_cost = abs(target_pos - pos) + 1 + ROUND_LOSS_COST * losses, losses, builds
    candidate = add_cost(transition_cost, best_cost(*next_state))
    if candidate < best:
        choices[state] = target_pos, builds, next_state
        return candidate
    return best


def evaluate_builds(
        state: tuple[int, int, int, int, int], best: tuple[int, int, int], step: int, needs_block: bool
) -> tuple[int, int, int]:
    """Evaluates every legal elevator build before the next blocking elevator and returns the best cost."""
    floor, pos, direction, elevators_left, clones_left = state
    if pos not in elevators_by_floor[floor]:
        best = evaluate_transition(state, best, pos, direction, True, 0)

    obstacle = nearest_elevator(floor, pos, step, needs_block)
    for target_pos in important_positions:
        if target_pos in elevators_by_floor[floor] or step == RIGHT and (target_pos <= pos or obstacle is not None and obstacle <= target_pos):
            continue
        if step == LEFT and (pos <= target_pos or obstacle is not None and target_pos <= obstacle):
            continue
        best = evaluate_transition(state, best, target_pos, step, True, needs_block)
    return best


def make_route(start_floor: int, start_pos: int, start_direction: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    """Builds floor targets and the subset of targets where a new elevator must be built from a start state."""
    global important_positions

    targets = {}
    builds = set()
    state = start_floor, start_pos, start_direction, nb_additional_elevators, nb_total_clones - 1
    debug("ROUTE_START", state)
    route_cost = best_cost(*state)
    debug("ROUTE_INITIAL_COST", route_cost)
    if route_cost[0] > nb_rounds:
        important_positions = tuple(range(width))
        choices.clear()
        best_cost.cache_clear()
        route_cost = best_cost(*state)
        debug("ROUTE_FULL_WIDTH_COST", route_cost)

    while state[0] < exit_floor:
        if state not in choices:
            debug("ROUTE_MISSING_CHOICE", "state", state, "route_cost", route_cost, "choices", len(choices))
        target_pos, should_build, state = choices[state]
        debug("ROUTE_STEP", "floor", state[0] - 1, "target", target_pos, "build", should_build, "next_state", state)
        targets[state[0] - 1] = target_pos
        if should_build:
            builds.add((state[0] - 1, target_pos))

    targets[exit_floor] = exit_pos
    debug("ROUTE_TARGETS", targets)
    debug("ROUTE_BUILDS", builds)
    return targets, builds


nb_floors, width, nb_rounds, exit_floor, exit_pos, nb_total_clones, nb_additional_elevators, nb_elevators = map(int, input().split())
elevators_by_floor = [set() for _ in range(nb_floors)]
choices = {}

for _ in range(nb_elevators):
    elevator_floor, elevator_pos = map(int, input().split())
    elevators_by_floor[elevator_floor].add(elevator_pos)

important_positions = {0, width - 1, exit_pos}
for elevator_positions in elevators_by_floor:
    for elevator_pos in elevator_positions:
        important_positions.add(elevator_pos)
        if elevator_pos > 0:
            important_positions.add(elevator_pos - 1)
        if elevator_pos < width - 1:
            important_positions.add(elevator_pos + 1)
important_positions = tuple(sorted(important_positions))
debug("INIT", "floors", nb_floors, "width", width, "rounds", nb_rounds, "exit", (exit_floor, exit_pos))
debug("INIT_RESOURCES", "clones", nb_total_clones, "extra", nb_additional_elevators, "elevators", nb_elevators)
debug("ELEVATORS", [sorted(elevator_positions) for elevator_positions in elevators_by_floor])
debug("IMPORTANT_POSITIONS", important_positions)

targets_by_floor = None
elevators_to_build = set()
built_elevators = set()
turn = 0

while True:
    clone_floor_text, clone_pos_text, direction_text = input().split()
    clone_floor = int(clone_floor_text)
    clone_pos = int(clone_pos_text)
    turn += 1

    if clone_floor == -1:
        debug("TURN", turn, "floor", clone_floor, "pos", clone_pos, "direction", direction_text, "action", "WAIT", "reason", "no leading clone")
        print("WAIT")
        continue

    direction = DIRECTIONS[direction_text]
    if targets_by_floor is None:
        targets_by_floor, elevators_to_build = make_route(clone_floor, clone_pos, direction)

    if clone_floor not in targets_by_floor:
        debug("MISSING_TARGET", "turn", turn, "floor", clone_floor, "pos", clone_pos, "direction", direction_text, "targets", targets_by_floor)
    target_pos = targets_by_floor[clone_floor]
    needs_build = (clone_floor, clone_pos) in elevators_to_build and (clone_floor, clone_pos) not in built_elevators
    if needs_build and clone_pos == target_pos:
        built_elevators.add((clone_floor, clone_pos))
        debug("TURN", turn, "f", clone_floor, "p", clone_pos, "dir", direction_text, "target", target_pos, "action", "ELEVATOR")
        debug("BUILT_ELEVATORS", sorted(built_elevators))
        print("ELEVATOR")
        continue

    needs_block = direction == LEFT and clone_pos < target_pos or direction == RIGHT and clone_pos > target_pos
    action = "BLOCK" if needs_block else "WAIT"
    debug("TURN", turn, "f", clone_floor, "p", clone_pos, "dir", direction_text, "t", target_pos, "build", needs_build, "block", needs_block, "act", action)
    print(action)
