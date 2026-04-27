"""Runs a simple Mad Pod Racing bot."""

import math
import sys
from dataclasses import dataclass

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
from scipy.optimize import direct


DRAG = 0.85
CHECKPOINT_RADIUS = 600

BOOST_THRUST = 650
MAX_TURN_DEG = 18
BOOST_ANGLE_TOL = 1

CHECKPOINT_BONUS = 100000
COMMAND_TARGET_DIST = 10000
PREDICT_TURNS = 2
DIRECT_MAXFUN = 120


@dataclass(slots=True)
class Pod:
    """Stores one pod state.
    :var pod_ind: Pod index inside its team.
    :var position: Pod center coordinates.
    :var velocity: Pod speed vector after the previous turn friction and truncation.
    :var direction: Pod angle in degrees from the positive x-axis, positive toward negative y.
    :var next_checkpoint_ind: Index of the current checkpoint in checkpoints, or None when unknown.
    """
    pod_ind: int
    position: NDArray[int]
    velocity: NDArray[int]
    direction: float
    next_checkpoint_ind: int | None


@dataclass(slots=True)
class GameState:
    """Stores the current turn state.
    :var laps: Number of laps to complete.
    :var checkpoints: Circuit checkpoints.
    :var my_pods: Our pod states.
    :var foe_pods: Opponent pod states.
    :var boosts: Number of unused team boosts.
    """
    laps: int
    checkpoints: list[NDArray[int]]
    my_pods: list[Pod]
    foe_pods: list[Pod]
    boosts: int

    def log(self):
        """Prints the complete game state."""
        log(repr(self))


def main():
    """Runs the game loop."""
    game_state = read_initial_game_state()
    while True:
        game_state = update_game_state(game_state)
        game_state.log()

        for target_pos, thrust in choose_move(game_state):
            print(*target_pos, thrust)


def read_initial_game_state() -> GameState:
    """Reads immutable race initialization.
    :return: Initial game state before the first turn input.
    """
    laps = int(input())
    checkpoint_count = int(input())
    checkpoints = [np.array(tuple(map(int, input().split()))) for _ in range(checkpoint_count)]
    return GameState(laps, checkpoints, [], [], 1)


def update_game_state(prev_game_state: GameState) -> GameState:
    """Updates game state.
    :param prev_game_state: Previous game state carrying persistent race data.
    :return: Parsed game state for the current turn.
    """
    our_pods = [read_pod(pod_ind) for pod_ind in range(2)]
    foe_pods = [read_pod(pod_ind) for pod_ind in range(2)]
    return GameState(prev_game_state.laps, prev_game_state.checkpoints, our_pods, foe_pods, prev_game_state.boosts)


def read_pod(pod_ind: int) -> Pod:
    """Reads one pod state.
    :param pod_ind: Pod index inside its team.
    :return: Pod state with the game angle converted to the bot angle convention.
    """
    x, y, vx, vy, angle, next_checkpoint_ind = map(int, input().split())
    return Pod(pod_ind, np.array((x, y)), np.array((vx, vy)), normalize_angle(-angle), next_checkpoint_ind)


def choose_move(game_state: GameState) -> list[tuple[NDArray[int], int | str]]:
    """Chooses commands for both pods.
    :param game_state: Current game state.
    :return: Target coordinates and thrust command for each pod.
    """
    commands = []
    for pod in game_state.my_pods:
        commands.append(choose_pod_move(game_state, pod))
        if commands[-1][1] == "BOOST":
            game_state.boosts -= 1
    return commands


def choose_pod_move(game_state: GameState, pod: Pod) -> tuple[NDArray[int], int | str]:
    """Chooses one pod command minimizing predicted distance to the next checkpoint.
    :param game_state: Current game state.
    :param pod: Pod to command.
    :return: Target coordinates and thrust command.
    """
    checkpoint_pos = game_state.checkpoints[pod.next_checkpoint_ind]
    checkpoint_delta = checkpoint_pos - pod.position
    checkpoint_direction = -math.degrees(math.atan2(checkpoint_delta[1], checkpoint_delta[0]))
    direction_delta_guess = np.clip(normalize_angle(checkpoint_direction - pod.direction), -MAX_TURN_DEG, MAX_TURN_DEG)
    direction_guess = pod.direction + direction_delta_guess
    guess_moves = np.tile(np.array((direction_guess, 100)), PREDICT_TURNS)
    bounds = [(-360, 360), (0, 100)] * PREDICT_TURNS
    result = direct(lambda moves: score_moves(game_state, pod, moves), bounds, maxfun=DIRECT_MAXFUN)
    direction = normalize_angle(result.x[0])
    thrust = round(result.x[1])
    checkpoint_dist = linalg.norm(checkpoint_delta)
    if abs(normalize_angle(pod.direction - checkpoint_direction)) <= BOOST_ANGLE_TOL and checkpoint_dist > 5000 and game_state.boosts:
        thrust = "BOOST"

    log(f"Pod {pod.pod_ind} move:")
    guess_moves = ", ".join(f"{value:.3g}" for value in guess_moves)
    opt_moves = ", ".join(f"{value:.3g}" for value in result.x)
    log(f"guess moves=[{guess_moves}]")
    log(f"opt moves=[{opt_moves}]; score={round(result.fun)}")
    log(f"opt success={result.success}; nfev={result.nfev}; nit={result.nit}; message={result.message}")
    log("Predicted:")
    predicted_pod = pod
    for move_ind in range(0, len(result.x), 2):
        predicted_pod, _ = predict_next(predicted_pod, game_state.checkpoints, result.x[move_ind], result.x[move_ind + 1])
        log(f"pos={predicted_pod.position}; CP={predicted_pod.next_checkpoint_ind}")

    direction_rad = math.radians(direction)
    target_pos = np.rint(pod.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * COMMAND_TARGET_DIST).astype(int)
    return target_pos, thrust


def score_moves(game_state: GameState, pod: Pod, moves: NDArray[float]) -> float:
    """Scores a move over the prediction horizon.
    :param game_state: Current game state.
    :param pod: Pod to command.
    :param moves: Alternating direction and thrust values for every predicted turn.
    :return: Lower score for better predicted race progress.
    """
    predicted_pod, passed_checkpoints = predict_turns(pod, game_state.checkpoints, moves)
    return linalg.norm(game_state.checkpoints[predicted_pod.next_checkpoint_ind] - predicted_pod.position) - passed_checkpoints * CHECKPOINT_BONUS


def predict_turns(current: Pod, checkpoints: list[NDArray[int]], moves: NDArray[float]) -> tuple[Pod, int]:
    """Predicts pod state after multiple turns.
    :param current: Current pod state.
    :param checkpoints: Circuit checkpoints.
    :param moves: Alternating direction and thrust values for every future turn.
    :return: Predicted pod state and number of checkpoints crossed.
    """
    pod = current
    passed_checkpoints = 0
    for move_ind in range(0, len(moves), 2):
        pod, turn_passed_checkpoints = predict_next(pod, checkpoints, moves[move_ind], moves[move_ind + 1])
        passed_checkpoints += turn_passed_checkpoints
    return pod, passed_checkpoints


def predict_next(current: Pod, checkpoints: list[NDArray[int]], direction: float, thrust: float) -> tuple[Pod, int]:
    """Predicts next pod state after one turn.
    :param current: Current pod state.
    :param checkpoints: Circuit checkpoints.
    :param direction: Desired pod direction in degrees.
    :param thrust: Thrust level to apply.
    :return: Predicted pod state and number of checkpoints crossed during movement.
    """
    direction_delta = np.clip(normalize_angle(direction - current.direction), -MAX_TURN_DEG, MAX_TURN_DEG)
    next_direction = normalize_angle(current.direction + direction_delta)
    next_direction_rad = math.radians(next_direction)
    acceleration = np.array((math.cos(next_direction_rad), -math.sin(next_direction_rad))) * thrust
    velocity = current.velocity + acceleration
    segment_start = current.position.astype(float)
    segment_end = current.position + velocity
    passed_checkpoints = 0
    while passed_checkpoints < len(checkpoints) \
            and checkpoint_crossed(segment_start, segment_end, checkpoints[(current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)]):
        passed_checkpoints += 1

    position = np.floor(segment_end + 0.5).astype(int)
    velocity = (velocity * DRAG).astype(int)
    next_checkpoint_ind = (current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)
    return Pod(current.pod_ind, position, velocity, next_direction, next_checkpoint_ind), passed_checkpoints


def checkpoint_crossed(start: NDArray[float], end: NDArray[float], checkpoint: NDArray[int]) -> bool:
    """Checks whether a movement segment enters a checkpoint radius.
    :param start: Movement segment start.
    :param end: Movement segment end.
    :param checkpoint: Checkpoint center.
    :return: Whether the segment intersects the checkpoint.
    """
    movement = end - start
    relative_start = start - checkpoint
    a = np.dot(movement, movement)
    if a == 0:
        return np.dot(relative_start, relative_start) <= CHECKPOINT_RADIUS ** 2
    closest_delta = relative_start + movement * np.clip(-np.dot(relative_start, movement) / a, 0, 1)
    return np.dot(closest_delta, closest_delta) <= CHECKPOINT_RADIUS ** 2


def normalize_angle(angle: float) -> float:
    """Normalizes an angle to [-180, 180) degrees.
    :param angle: Angle in degrees.
    :return: Equivalent angle in the normalized range.
    """
    return (angle + 180) % 360 - 180


def log(msg: str):
    """Prints a debug message.
    :param msg: Message to print.
    """
    print(msg, file=sys.stderr)


def test():
    """Runs the local prediction sanity test."""
    checkpoints = [np.array([11498, 6051]), np.array([9095, 1838])]
    current = Pod(0, np.array([9988, 6216]), np.array([573, 134]), 0, 0)
    game_state = GameState(0, checkpoints, [current], [], 0)
    score1 = score_moves(game_state, current, [6.24, 100, 6.24, 100])
    score2 = score_moves(game_state, current, [90, 100, 90, 100])
    print(score1, score2)


def test2():
    """Compares optimizer behavior on the local hard-turn case."""
    checkpoints = [np.array([11498, 6051]), np.array([9095, 1838])]
    current = Pod(0, np.array([9988, 6216]), np.array([573, 134]), 0, 0)
    game_state = GameState(0, checkpoints, [current], [], 0)
    moves1 = [6.24, 100, 6.24, 100]
    moves2 = [90, 100, 90, 100]
    print("score1", score_moves(game_state, current, moves1), moves1)
    print("score2", score_moves(game_state, current, moves2), moves2)
    result = direct(lambda move: score_moves(game_state, current, move), [(-360, 360), (0, 100)] * PREDICT_TURNS, maxfun=DIRECT_MAXFUN)
    print("direct", result.fun, result.x, result.nfev, result.message)


main()
# test()
# test2()
