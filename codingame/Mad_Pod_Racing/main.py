"""Runs a simple Mad Pod Racing bot."""

import math
import sys
from dataclasses import dataclass

import numpy as np
from numpy import linalg
from numpy.typing import NDArray


DRAG = 0.85
CHECKPOINT_RADIUS = 600
BOOST_THRUST = 650
MAX_TURN_DEG = 18

BOOST_ANGLE_TOL = 1
CHECKPOINT_BONUS = 100000
COMMAND_TARGET_DIST = 10000
PREDICT_TURNS = 6
GRID_BEAM_WIDTH = 30
GRID_TURN_DELTAS = (-18, -9, 0, 9, 18)
GRID_THRUSTS = (0, 50, 100)


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

    def get_next_checkpoint_distance(self, checkpoints: list[NDArray[int]]) -> float:
        """Computes distance to the next checkpoint.
        :param checkpoints: Circuit checkpoints.
        :return: Distance to the next checkpoint of the pod.
        """
        return linalg.norm(checkpoints[self.next_checkpoint_ind] - self.position)

    def log(self):
        """Prints one pod state."""
        log(f"{self.pod_ind}: pos=({self.position[0]}, {self.position[1]}); vel=({self.velocity[0]}, {self.velocity[1]}); "
            f"dir={self.direction:g}; CP={self.next_checkpoint_ind}")


@dataclass(slots=True)
class FutureState:
    """Stores a predicted future state.
    :var moves: Alternating direction and thrust values that produced this state.
    :var pod: Predicted pod state.
    :var passed_checkpoints: Number of checkpoints crossed by the move sequence.
    """
    moves: list[float]
    pod: Pod
    passed_checkpoints: int

    def get_score(self, checkpoints: list[NDArray[int]]) -> float:
        """Scores predicted race progress.
        :param checkpoints: Circuit checkpoints.
        :return: Lower score for better predicted race progress.
        """
        return self.pod.get_next_checkpoint_distance(checkpoints) - self.passed_checkpoints * CHECKPOINT_BONUS


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
        """Prints pod states."""
        log("My pods:")
        for pod in self.my_pods:
            pod.log()
        log("Enemy pods:")
        for pod in self.foe_pods:
            pod.log()


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
    """Chooses one pod command by searching discrete future turns.
    :param game_state: Current game state.
    :param pod: Pod to command.
    :return: Target coordinates and thrust command.
    """
    best_moves = grid_search_moves(pod, game_state.checkpoints)
    direction = normalize_angle(best_moves[0])
    thrust = round(best_moves[1])

    checkpoint_delta = game_state.checkpoints[pod.next_checkpoint_ind] - pod.position
    checkpoint_direction = -math.degrees(math.atan2(checkpoint_delta[1], checkpoint_delta[0]))
    if abs(normalize_angle(pod.direction - checkpoint_direction)) <= BOOST_ANGLE_TOL and pod.get_next_checkpoint_distance(game_state.checkpoints) > 5000 \
            and game_state.boosts:
        thrust = "BOOST"

    log(f"Pod {pod.pod_ind} move:")
    best_moves_text = ", ".join(f"{value:.3g}" for value in best_moves)
    log(f"grid moves=[{best_moves_text}]")
    log("Predicted:")
    future_state = FutureState([], pod, 0)
    for move_ind in range(0, len(best_moves), 2):
        future_state = predict_next(future_state.pod, game_state.checkpoints, best_moves[move_ind], best_moves[move_ind + 1])
        log(f"pos={future_state.pod.position}; CP={future_state.pod.next_checkpoint_ind}")

    direction_rad = math.radians(direction)
    target_pos = np.rint(pod.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * COMMAND_TARGET_DIST).astype(int)
    return target_pos, thrust


def grid_search_moves(pod: Pod, checkpoints: list[NDArray[int]]) -> list[float]:
    """Searches a discrete move grid over several future turns.
    :param pod: Pod to command.
    :param checkpoints: Circuit checkpoints.
    :return: Best move sequence.
    """
    candidates = [FutureState([], pod, 0)]
    for _ in range(PREDICT_TURNS):
        next_candidates = []
        for candidate in candidates:
            for turn_delta in GRID_TURN_DELTAS:
                for thrust in GRID_THRUSTS:
                    direction = normalize_angle(candidate.pod.direction + turn_delta)
                    future_state = predict_next(candidate.pod, checkpoints, direction, thrust)
                    next_candidates.append(
                        FutureState(candidate.moves + future_state.moves, future_state.pod, candidate.passed_checkpoints + future_state.passed_checkpoints))
        candidates = sorted(next_candidates, key=lambda candidate: candidate.get_score(checkpoints))[:GRID_BEAM_WIDTH]
    return candidates[0].moves


def predict_turns(current: Pod, checkpoints: list[NDArray[int]], moves: list[float] | NDArray[float]) -> FutureState:
    """Predicts pod state after multiple turns.
    :param current: Current pod state.
    :param checkpoints: Circuit checkpoints.
    :param moves: Alternating direction and thrust values for every future turn.
    :return: Predicted future state.
    """
    future_state = FutureState([], current, 0)
    for move_ind in range(0, len(moves), 2):
        next_state = predict_next(future_state.pod, checkpoints, moves[move_ind], moves[move_ind + 1])
        future_state = FutureState(future_state.moves + next_state.moves, next_state.pod, future_state.passed_checkpoints + next_state.passed_checkpoints)
    return future_state


def predict_next(current: Pod, checkpoints: list[NDArray[int]], direction: float, thrust: float) -> FutureState:
    """Predicts next pod state after one turn.
    :param current: Current pod state.
    :param checkpoints: Circuit checkpoints.
    :param direction: Desired pod direction in degrees.
    :param thrust: Thrust level to apply.
    :return: Predicted future state after one turn.
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
    return FutureState([direction, thrust], Pod(current.pod_ind, position, velocity, next_direction, next_checkpoint_ind), passed_checkpoints)


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
    score1 = predict_turns(current, checkpoints, [6.24, 100, 6.24, 100]).get_score(checkpoints)
    score2 = predict_turns(current, checkpoints, [90, 100, 90, 100]).get_score(checkpoints)
    print(score1, score2)


main()
# test()
