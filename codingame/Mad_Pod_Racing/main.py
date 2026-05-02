"""Runs a simple Mad Pod Racing bot."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import OptimizeResult


# Base game constants
DRAG = 0.85
CHECKPOINT_RADIUS = 600
BOOST_THRUST = 650
MAX_TURN_DEG = 18
SHIELD_COOLDOWN_TURNS = 3
TARGET_DISTANCE = 10000

# Racer behavior constants
PREDICT_TURNS = 4
CHECKPOINT_BONUS = 20000
BOOST_ANGLE_TOL = 1
BOOST_MIN_DIST = 5000

# Brute behavior constants
AGGRO_DISTANCE = 5000
TRACK_DISTANCE = 1600
SHIELD_DISTANCE = 1200
RACER_AVOID_DISTANCE = 1200
PATROL_THRUST_ANGLE = 45


@dataclass(slots=True)
class BasePod:
    """Stores one pod state.
    :var ind: Pod index inside its team.
    :var position: Pod center coordinates.
    :var velocity: Pod speed vector.
    :var direction: Pod angle in degrees from the positive x-axis, positive toward negative y.
    :var next_checkpoint_ind: Index of the current checkpoint in checkpoints.
    :var shield_cooldown: Remaining shield cooldown turns.
    """
    ind: int
    position: NDArray[float]
    velocity: NDArray[float]
    direction: float
    next_checkpoint_ind: int
    shield_cooldown: int = 0

    def get_next_checkpoint_distance(self, checkpoints: list[NDArray[int]]) -> float:
        """Computes distance to the next checkpoint.
        :param checkpoints: Circuit checkpoints.
        :return: Distance to the next checkpoint of the pod.
        """
        return linalg.norm(checkpoints[self.next_checkpoint_ind] - self.position)

    def get_direction_target(self, direction: float) -> NDArray[int]:
        """Computes a far target point in a direction.
        :param direction: Target direction.
        :return: Target coordinates.
        """
        direction_rad = math.radians(direction)
        return np.rint(self.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * TARGET_DISTANCE).astype(int)

    def log(self):
        """Prints one pod state."""
        log(f"{self.ind}: pos=({self.position[0]}, {self.position[1]}); vel=({self.velocity[0]}, {self.velocity[1]}); "
            f"dir={self.direction:g}; CP ind={self.next_checkpoint_ind}")


@dataclass(slots=True)
class RacerPod(BasePod):
    """Stores racer pod state."""

    def choose_command(self, game_state: GameState) -> tuple[NDArray[int], int | str]:
        """Chooses one racer command.
        :param game_state: Current game state.
        :return: Target coordinates and thrust command.
        """
        return self.choose_move(game_state)[:2]

    def choose_move(self, game_state: GameState) -> tuple[NDArray[int], int | str, NDArray[float]]:
        """Chooses one racer command and planned future moves.
        :param game_state: Current game state.
        :return: Target coordinates, thrust command and planned future moves.
        """
        checkpoint_delta = game_state.checkpoints[self.next_checkpoint_ind] - self.position
        checkpoint_direction = -math.degrees(math.atan2(checkpoint_delta[1], checkpoint_delta[0]))
        if game_state.turn_ind == 0:
            return game_state.checkpoints[self.next_checkpoint_ind], 100, np.array((normalize_angle(checkpoint_direction - self.direction), 100), dtype=float)

        result = self.optimize_moves(game_state.checkpoints)
        direction = normalize_angle(self.direction + result.x[0])
        thrust = round(result.x[1])
        if abs(normalize_angle(self.direction - checkpoint_direction)) <= BOOST_ANGLE_TOL and \
            self.get_next_checkpoint_distance(game_state.checkpoints) > BOOST_MIN_DIST and game_state.boosts:
            thrust = "BOOST"

        log(f"Pod {self.ind} move:")
        opt_moves = ", ".join(f"{value:.3g}" for value in result.x)
        log(f"opt moves=[{opt_moves}]; score={round(result.fun)}")
        log(f"opt success={result.success}; nfev={result.nfev}; message={result.message}")
        log("Predicted:")
        for future_state in predict_turns(self, game_state.checkpoints, result.x):
            log(f"pos={future_state.pod.position}; CP={future_state.pod.next_checkpoint_ind}")

        return self.get_direction_target(direction), thrust, result.x

    def optimize_moves(self, checkpoints: list[NDArray[int]]) -> OptimizeResult:
        """Optimizes future moves for one pod.
        :param checkpoints: Circuit checkpoints.
        :return: SciPy optimization result with constrained moves written to x.
        """
        move_bounds = optimize.Bounds(np.tile(np.array((-MAX_TURN_DEG, 0), dtype=float), PREDICT_TURNS),
                                      np.tile(np.array((MAX_TURN_DEG, 100), dtype=float), PREDICT_TURNS))
        result = optimize.minimize(lambda moves: predict_turns(self, checkpoints, moves)[-1].get_score(checkpoints), self.get_optimizer_guess_moves(),
                                   method="L-BFGS-B", bounds=move_bounds, options={"maxiter": np.iinfo(np.int32).max})
        result.x = constrain_moves(result.x)
        return result

    @staticmethod
    def get_optimizer_guess_moves() -> NDArray[float]:
        """Computes the initial optimizer move guess.
        :return: Alternating direction delta and thrust values.
        """
        return np.tile(np.array((0, 100), dtype=float), PREDICT_TURNS)


@dataclass(slots=True)
class BrutePod(BasePod):
    """Stores brute pod state."""

    def choose_command(self, game_state: GameState) -> tuple[NDArray[int], int | str]:
        """Chooses one brute command.
        :param game_state: Current game state.
        :return: Target coordinates and thrust command.
        """
        victim = self.find_victim(game_state)
        if victim is None:
            log(f"Brute: patrol")
            target_pos, thrust = self.choose_patrol_command(game_state)
            return self.avoid_racer(game_state, target_pos, thrust)

        log(f"Brute {self.ind}: ram foe {victim.ind}")
        return self.avoid_racer(game_state, np.rint(victim.position + victim.velocity).astype(int), "SHIELD" if self.should_shield(game_state, victim) else 100)

    def find_victim(self, game_state: GameState) -> BasePod | None:
        """Finds an opponent worth ramming.
        :param game_state: Current game state.
        :return: Selected opponent pod, or None.
        """
        victims = [foe_pod for foe_pod in game_state.foe_pods if self.is_victim(foe_pod, game_state.checkpoints)]
        return min(victims, key=lambda foe_pod: linalg.norm(foe_pod.position - self.position), default=None)

    def is_victim(self, foe_pod: BasePod, checkpoints: list[NDArray[int]]) -> bool:
        """Checks whether an opponent is a brute target.
        :param foe_pod: Opponent pod.
        :param checkpoints: Circuit checkpoints.
        :return: Whether the opponent is close enough and moving toward the brute.
        """
        track_distance = self.get_track_distance(foe_pod.position, checkpoints)
        pod_distance = linalg.norm(foe_pod.position - self.position)
        approach_speed = np.dot(foe_pod.velocity, self.position - foe_pod.position)
        return track_distance <= TRACK_DISTANCE and pod_distance <= AGGRO_DISTANCE and approach_speed > 0

    @staticmethod
    def get_track_distance(position: NDArray[float], checkpoints: list[NDArray[int]]) -> float:
        """Computes distance to the closest checkpoint segment.
        :param position: Position to test.
        :param checkpoints: Circuit checkpoints.
        :return: Distance to the closest segment joining two checkpoint centers.
        """
        return min(BrutePod.get_segment_distance(position, checkpoints[checkpoint_ind], checkpoints[(checkpoint_ind + 1) % len(checkpoints)])
                   for checkpoint_ind in range(len(checkpoints)))

    @staticmethod
    def get_segment_distance(position: NDArray[float], start: NDArray[int], end: NDArray[int]) -> float:
        """Computes distance to a segment.
        :param position: Position to test.
        :param start: Segment start.
        :param end: Segment end.
        :return: Distance from position to the segment.
        """
        return linalg.norm(position - BrutePod.get_closest_point_on_segment(position, start, end))

    @staticmethod
    def get_closest_point_on_segment(position: NDArray[float], start: NDArray[int], end: NDArray[int]) -> NDArray[float]:
        """Computes the closest point on a segment.
        :param position: Position to project.
        :param start: Segment start.
        :param end: Segment end.
        :return: Closest point on the segment.
        """
        segment = end - start
        return start + segment * np.clip(np.dot(position - start, segment) / np.dot(segment, segment), 0, 1)

    def choose_patrol_command(self, game_state: GameState) -> tuple[NDArray[int], int]:
        """Chooses a counterflow patrol command.
        :param game_state: Current game state.
        :return: Target coordinates and thrust.
        """
        active_segment = self.get_active_segment(game_state.checkpoints)
        if active_segment is None:
            return self.get_direction_target(self.direction), 0

        segment_ind, segment_start, segment_end = active_segment
        if segment_ind is not None and self.should_advance_segment(game_state, segment_ind, segment_start, segment_end):
            segment_ind = (segment_ind - 1) % len(game_state.checkpoints)
            segment_start, segment_end = self.get_counterflow_segment(game_state.checkpoints, segment_ind)
        segment_direction = self.get_segment_direction(segment_start, segment_end)
        return np.rint(segment_end).astype(int), 0 if abs(normalize_angle(self.direction - segment_direction)) > PATROL_THRUST_ANGLE else 100

    def get_active_segment(self, checkpoints: list[NDArray[int]]) -> tuple[int | None, NDArray[float], NDArray[float]] | None:
        """Chooses the active patrol segment.
        :param checkpoints: Circuit checkpoints.
        :return: Active segment index, start and end, or None when no patrol is needed.
        """
        candidates = []
        for segment_ind in range(len(checkpoints)):
            segment_start, segment_end = self.get_counterflow_segment(checkpoints, segment_ind)
            closest_point = self.get_closest_point_on_segment(self.position, segment_start, segment_end)
            if linalg.norm(self.position - closest_point) <= TRACK_DISTANCE:
                candidates.append((linalg.norm(segment_end - closest_point), segment_ind, segment_start, segment_end))

        if len(candidates) == len(checkpoints):
            return None
        if candidates:
            _, segment_ind, segment_start, segment_end = max(candidates, key=lambda candidate: candidate[0])
            return segment_ind, segment_start, segment_end

        return None, self.position, self.get_nearest_track_point(checkpoints)

    def get_nearest_track_point(self, checkpoints: list[NDArray[int]]) -> NDArray[float]:
        """Finds the nearest point on any track segment.
        :param checkpoints: Circuit checkpoints.
        :return: Closest track point.
        """
        points = [self.get_closest_point_on_segment(self.position, *self.get_counterflow_segment(checkpoints, segment_ind))
                  for segment_ind in range(len(checkpoints))]
        return min(points, key=lambda point: linalg.norm(point - self.position))

    @staticmethod
    def get_counterflow_segment(checkpoints: list[NDArray[int]], segment_ind: int) -> tuple[NDArray[int], NDArray[int]]:
        """Gets a counterflow track segment.
        :param checkpoints: Circuit checkpoints.
        :param segment_ind: Segment index equal to its end checkpoint.
        :return: Segment start and end.
        """
        return checkpoints[(segment_ind + 1) % len(checkpoints)], checkpoints[segment_ind]

    @staticmethod
    def get_segment_direction(start: NDArray[float], end: NDArray[float]) -> float:
        """Computes segment direction.
        :param start: Segment start.
        :param end: Segment end.
        :return: Direction angle.
        """
        segment = end - start
        return -math.degrees(math.atan2(segment[1], segment[0]))

    def should_advance_segment(self, game_state: GameState, segment_ind: int, segment_start: NDArray[float], segment_end: NDArray[float]) -> bool:
        """Checks whether brute should start turning toward the next segment.
        :param game_state: Current game state.
        :param segment_ind: Active segment index.
        :param segment_start: Active segment start.
        :param segment_end: Active segment end.
        :return: Whether to advance to the next segment.
        """
        next_segment_start, next_segment_end = self.get_counterflow_segment(game_state.checkpoints, (segment_ind - 1) % len(game_state.checkpoints))
        turn_count = math.ceil(abs(normalize_angle(self.get_segment_direction(next_segment_start, next_segment_end) - self.direction)) / MAX_TURN_DEG)
        if turn_count == 0:
            return self.get_segment_distance(self.position, next_segment_start, next_segment_end) <= TRACK_DISTANCE

        pod = self
        for turn_ind in range(turn_count):
            pod = self.predict_segment_following(pod, game_state.checkpoints, segment_start, segment_end, game_state.turn_ind == 0 and turn_ind == 0)
            if self.get_segment_distance(pod.position, next_segment_start, next_segment_end) <= TRACK_DISTANCE:
                return True
        return False

    @staticmethod
    def predict_segment_following(pod: BasePod, checkpoints: list[NDArray[int]], segment_start: NDArray[float], segment_end: NDArray[float],
                                  first_turn: bool) -> BasePod:
        """Predicts one turn of normal segment following.
        :param pod: Pod to predict.
        :param checkpoints: Circuit checkpoints.
        :param segment_start: Active segment start.
        :param segment_end: Active segment end.
        :param first_turn: Whether this move ignores the turn angle limit.
        :return: Predicted pod.
        """
        segment_direction = BrutePod.get_segment_direction(segment_start, segment_end)
        target_direction = BrutePod.get_segment_direction(pod.position, segment_end)
        thrust = 0 if abs(normalize_angle(pod.direction - segment_direction)) > PATROL_THRUST_ANGLE else 100
        return predict_next(pod, checkpoints, normalize_angle(target_direction - pod.direction), thrust, first_turn).pod

    def should_shield(self, game_state: GameState, victim: BasePod) -> bool:
        """Checks whether the brute should shield before impact.
        :param game_state: Current game state.
        :param victim: Selected opponent pod.
        :return: Whether to activate shield.
        """
        next_distance = linalg.norm(victim.position + victim.velocity - self.position - self.velocity)
        return self.shield_cooldown == 0 and next_distance <= SHIELD_DISTANCE

    def avoid_racer(self, game_state: GameState, target_pos: NDArray[int], thrust: int | str) -> tuple[NDArray[int], int | str]:
        """Adjusts command to avoid our racer.
        :param game_state: Current game state.
        :param target_pos: Planned command target.
        :param thrust: Planned command thrust.
        :return: Safe target coordinates and thrust command.
        """
        racer = next(pod for pod in game_state.my_pods if isinstance(pod, RacerPod))
        next_distance = linalg.norm(self.predict_command_position(target_pos, thrust, game_state.turn_ind == 0) - racer.position - racer.velocity)
        if next_distance >= RACER_AVOID_DISTANCE:
            return target_pos, thrust

        log(f"Brute {self.ind}: avoid racer")
        return np.rint(self.position * 2 - racer.position).astype(int), 100

    def predict_command_position(self, target_pos: NDArray[int], thrust: int | str, first_turn: bool) -> NDArray[float]:
        """Predicts next brute position for a command.
        :param target_pos: Command target.
        :param thrust: Command thrust.
        :param first_turn: Whether this command ignores the turn angle limit.
        :return: Predicted position after movement.
        """
        if thrust == "SHIELD":
            return self.position + self.velocity

        target_delta = target_pos - self.position
        target_direction = -math.degrees(math.atan2(target_delta[1], target_delta[0]))
        direction_delta, thrust = constrain_moves([normalize_angle(target_direction - self.direction), thrust], first_turn)
        direction_rad = math.radians(normalize_angle(self.direction + direction_delta))
        return self.position + self.velocity + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * thrust


@dataclass(slots=True)
class FutureState:
    """Stores a predicted future state.
    :var moves: Alternating direction delta and thrust values that produced this state.
    :var pod: Predicted pod state.
    :var passed_checkpoints: Number of checkpoints crossed by the move sequence.
    """
    moves: list[float]
    pod: BasePod
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
    :var turn_ind: Current turn index.
    :var laps: Number of laps to complete.
    :var checkpoints: Circuit checkpoints.
    :var my_pods: Our pod states.
    :var foe_pods: Opponent pod states.
    :var boosts: Number of unused team boosts.
    """
    turn_ind: int
    laps: int
    checkpoints: list[NDArray[int]]
    my_pods: list[RacerPod | BrutePod]
    foe_pods: list[BasePod]
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
    return GameState(-1, laps, checkpoints, [], [], 1)


def update_game_state(prev_game_state: GameState) -> GameState:
    """Updates game state.
    :param prev_game_state: Previous game state carrying persistent race data.
    :return: Parsed game state for the current turn.
    """
    if prev_game_state.turn_ind == -1:
        our_pods = [read_pod(0, RacerPod), read_pod(1, BrutePod)]
    else:
        our_pods = [read_pod(0, RacerPod, max(0, prev_game_state.my_pods[0].shield_cooldown - 1)),
                    read_pod(1, BrutePod, max(0, prev_game_state.my_pods[1].shield_cooldown - 1))]
    foe_pods = [read_pod(pod_ind, BasePod) for pod_ind in range(2)]
    return GameState(prev_game_state.turn_ind + 1, prev_game_state.laps, prev_game_state.checkpoints, our_pods, foe_pods, prev_game_state.boosts)


def read_pod(pod_ind: int, pod_type: type[BasePod], shield_cooldown: int = 0) -> BasePod:
    """Reads one pod state.
    :param pod_ind: Pod index inside its team.
    :param pod_type: Pod class to instantiate.
    :param shield_cooldown: Remaining shield cooldown turns.
    :return: Pod state with the game angle converted to the bot angle convention.
    """
    x, y, vx, vy, angle, next_checkpoint_ind = map(int, input().split())
    return pod_type(pod_ind, np.array((x, y), dtype=float), np.array((vx, vy), dtype=float), normalize_angle(-angle), next_checkpoint_ind, shield_cooldown)


def choose_move(game_state: GameState) -> list[tuple[NDArray[int], int | str]]:
    """Chooses commands for both pods.
    :param game_state: Current game state.
    :return: Target coordinates and thrust command for each pod.
    """
    commands = []
    for pod in game_state.my_pods:
        target_pos, thrust = pod.choose_command(game_state)
        commands.append((target_pos, thrust))
        if thrust == "BOOST":
            game_state.boosts -= 1
        if thrust == "SHIELD":
            pod.shield_cooldown = SHIELD_COOLDOWN_TURNS
    return commands


def predict_turns(current: BasePod, checkpoints: list[NDArray[int]], moves: list[float] | NDArray[float], first_turn: bool = False) -> list[FutureState]:
    """Predicts pod state after multiple turns.
    :param current: Current pod state.
    :param checkpoints: Circuit checkpoints.
    :param moves: Alternating direction delta and thrust values for every future turn.
    :param first_turn: Whether the first move ignores the turn angle limit.
    :return: Predicted future states after every turn.
    """
    future_states = []
    for move_ind in range(0, len(moves), 2):
        next_state = predict_next(future_states[-1].pod if future_states else current, checkpoints, moves[move_ind], moves[move_ind + 1],
                                  first_turn and move_ind == 0)
        future_states.append(FutureState((future_states[-1].moves if future_states else []) + next_state.moves, next_state.pod,
                                         (future_states[-1].passed_checkpoints if future_states else 0) + next_state.passed_checkpoints))
    return future_states


def predict_next(current: BasePod, checkpoints: list[NDArray[int]], direction_delta: float, thrust: float, first_turn: bool = False) -> FutureState:
    """Predicts next pod state after one turn.
    :param current: Current pod state.
    :param checkpoints: Circuit checkpoints.
    :param direction_delta: Desired direction change in degrees.
    :param thrust: Thrust level to apply.
    :param first_turn: Whether this move ignores the turn angle limit.
    :return: Predicted future state after one turn.
    """
    direction_delta, thrust = constrain_moves([direction_delta, thrust], first_turn)
    next_direction = normalize_angle(current.direction + direction_delta)
    next_direction_rad = math.radians(next_direction)
    acceleration = np.array((math.cos(next_direction_rad), -math.sin(next_direction_rad))) * thrust
    velocity = current.velocity + acceleration
    segment_end = current.position + velocity
    passed_checkpoints = 0
    while passed_checkpoints < len(checkpoints) \
            and linalg.norm(checkpoints[(current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)] - segment_end) <= CHECKPOINT_RADIUS:
        passed_checkpoints += 1

    velocity = velocity * DRAG
    next_checkpoint_ind = (current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)
    pod = type(current)(current.ind, segment_end, velocity, next_direction, next_checkpoint_ind, current.shield_cooldown)
    return FutureState([direction_delta, thrust], pod, passed_checkpoints)


def constrain_moves(moves: list[float] | NDArray[float], first_turn: bool = False) -> NDArray[float]:
    """Applies model-level move boundaries.
    :param moves: Alternating direction delta and thrust values.
    :param first_turn: Whether the first direction delta ignores the turn angle limit.
    :return: Moves with clipped direction deltas and thrusts.
    """
    moves = np.array(moves).copy()
    moves[0] = normalize_angle(moves[0]) if first_turn else np.clip(moves[0], -MAX_TURN_DEG, MAX_TURN_DEG)
    moves[2::2] = np.clip(moves[2::2], -MAX_TURN_DEG, MAX_TURN_DEG)
    moves[1::2] = np.clip(moves[1::2], 0, 100)
    return moves


def normalize_angle(angle: float | NDArray[float]) -> float | NDArray[float]:
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


if __name__ == "__main__":
    main()
