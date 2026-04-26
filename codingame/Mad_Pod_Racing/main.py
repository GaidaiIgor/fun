"""Runs a simple Mad Pod Racing bot."""

import math
import sys
from dataclasses import dataclass

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
from scipy.optimize import minimize


DRAG = 0.85
BOOST_THRUST = 650
BOOST_ANGLE_TOL = 1
COMMAND_TARGET_DIST = 10000
MAX_TURN_DEG = 18


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


def main():
    """Runs the game loop."""
    game_state = read_initial_game_state()
    while True:
        game_state = update_game_state(game_state)

        for pod in game_state.my_pods:
            log(f"Our pod {pod.pod_ind}:")
            log(f"pos={pod.position}; vel={pod.velocity}; |v|={linalg.norm(pod.velocity):.3g}; angle={pod.direction:.3g}; CP={pod.next_checkpoint_ind}")
            log(f"CP dist={round(linalg.norm(game_state.checkpoints[pod.next_checkpoint_ind] - pod.position))}")
        for pod in game_state.foe_pods:
            log(f"Enemy pod {pod.pod_ind}:")
            log(f"pos={pod.position}; vel={pod.velocity}; |v|={linalg.norm(pod.velocity):.3g}; angle={pod.direction:.3g}; CP={pod.next_checkpoint_ind}")
            log(f"CP dist={round(linalg.norm(game_state.checkpoints[pod.next_checkpoint_ind] - pod.position))}")

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
    direction_guess = normalize_angle(pod.direction + direction_delta_guess)
    direction_bounds = (pod.direction - MAX_TURN_DEG, pod.direction + MAX_TURN_DEG)
    result = minimize(lambda move: linalg.norm(predict_next(pod, move[0], move[1]).position - checkpoint_pos),
                      np.array((direction_guess, 100)), bounds=(direction_bounds, (0, 100)), method="Nelder-Mead")
    direction = normalize_angle(result.x[0])
    thrust = round(result.x[1])
    checkpoint_dist = linalg.norm(checkpoint_delta)
    if abs(normalize_angle(pod.direction - checkpoint_direction)) <= BOOST_ANGLE_TOL and checkpoint_dist > 5000 and game_state.boosts:
        thrust = "BOOST"

    log(f"Pod {pod.pod_ind} move:")
    log(f"optimized move=({result.x[0]:.3g}, {result.x[1]:.3g})")
    log("Predicted:")
    predicted_pod = predict_next(pod, direction, BOOST_THRUST if thrust == "BOOST" else thrust)
    guess_dist = linalg.norm(predict_next(pod, direction_guess, 100).position - checkpoint_pos)
    log(f"pos={predicted_pod.position}; guess CP dist={round(guess_dist)}; opt CP dist={round(result.fun)}")

    direction_rad = math.radians(direction)
    target_pos = np.rint(pod.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * COMMAND_TARGET_DIST).astype(int)
    return target_pos, thrust


def predict_next(current: Pod, direction: float, thrust: float) -> Pod:
    """Predicts next pod state after one turn.
    :param current: Current pod state.
    :param direction: Desired pod direction in degrees.
    :param thrust: Thrust level to apply.
    :return: Predicted pod state.
    """
    direction_delta = normalize_angle(direction - current.direction)
    direction_delta = np.clip(direction_delta, -MAX_TURN_DEG, MAX_TURN_DEG)
    next_direction = normalize_angle(current.direction + direction_delta)
    next_direction_rad = math.radians(next_direction)
    acceleration = np.array((math.cos(next_direction_rad), -math.sin(next_direction_rad))) * thrust
    velocity = current.velocity + acceleration
    position = np.floor(current.position + velocity + 0.5).astype(int)
    velocity = (velocity * DRAG).astype(int)
    return Pod(current.pod_ind, position, velocity, next_direction, None)


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


main()
