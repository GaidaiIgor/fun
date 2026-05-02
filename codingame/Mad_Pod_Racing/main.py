"""Runs a Mad Pod Racing bot with one optimized racer pod and one heuristic brute pod."""

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
MAX_THRUST_ANGLE = 45


@dataclass(slots=True)
class BasePod:
    """Stores server-visible pod state shared by racers, brutes and opponents.
    Direction is the bot angle in degrees from the positive x-axis, with positive angles pointing toward decreasing screen y.
    next_checkpoint_ind is the checkpoint the pod must enter next, and shield_cooldown is only meaningful for our pods.
    """
    ind: int
    position: NDArray[float]
    velocity: NDArray[float]
    direction: float
    next_checkpoint_ind: int
    shield_cooldown: int = 0

    def get_next_checkpoint_distance(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns Euclidean distance from the pod center to the checkpoint indexed by next_checkpoint_ind."""
        return linalg.norm(checkpoints[self.next_checkpoint_ind] - self.position)

    def get_direction_target(self, direction: float) -> NDArray[int]:
        """Converts a bot direction angle into a far integer point that Codingame accepts as a command target."""
        direction_rad = math.radians(direction)
        return np.rint(self.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * TARGET_DISTANCE).astype(int)

    def log(self):
        """Prints index, position, velocity, direction and next checkpoint for debugging."""
        log(f"{self.ind}: pos=({self.position[0]}, {self.position[1]}); vel=({self.velocity[0]}, {self.velocity[1]}); "
            f"dir={self.direction:g}; CP ind={self.next_checkpoint_ind}")


@dataclass(slots=True)
class RacerPod(BasePod):
    """Represents our racing pod, which follows checkpoints in order using continuous move optimization."""

    def choose_command(self, game_state: GameState) -> tuple[NDArray[int], int | str]:
        """Drops planned future moves from choose_move and returns only the command target and thrust."""
        return self.choose_move(game_state)[:2]

    def choose_move(self, game_state: GameState) -> tuple[NDArray[int], int | str, NDArray[float]]:
        """Chooses one racer command and the optimized future move sequence that produced it.
        The first turn aims straight at the next checkpoint with full thrust. Later turns optimize direction delta and thrust pairs,
        use the first optimized direction delta as the command, and optionally replace thrust with BOOST.
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
        """Optimizes alternating direction delta and thrust coordinates for the configured prediction horizon.
        The objective is the final FutureState score after applying model-level move constraints and checkpoint progress.
        result.x is constrained again after minimize returns so downstream code sees valid model inputs.
        """
        move_bounds = optimize.Bounds(np.tile(np.array((-MAX_TURN_DEG, 0), dtype=float), PREDICT_TURNS),
                                      np.tile(np.array((MAX_TURN_DEG, 100), dtype=float), PREDICT_TURNS))
        result = optimize.minimize(lambda moves: predict_turns(self, checkpoints, moves)[-1].get_score(checkpoints), self.get_optimizer_guess_moves(),
                                   method="L-BFGS-B", bounds=move_bounds, options={"maxiter": np.iinfo(np.int32).max})
        result.x = constrain_moves(result.x)
        return result

    @staticmethod
    def get_optimizer_guess_moves() -> NDArray[float]:
        """Builds the neutral optimizer seed: zero direction change and full base thrust for each predicted turn."""
        return np.tile(np.array((0, 100), dtype=float), PREDICT_TURNS)


@dataclass(slots=True)
class BrutePod(BasePod):
    """Represents our disruptive pod, which rams useful victims or patrols the track counterflow."""

    def choose_command(self, game_state: GameState) -> tuple[NDArray[int], int | str]:
        """Chooses the brute action for this turn and runs the result through racer avoidance."""
        victim = self.find_victim(game_state)
        if victim is None:
            log(f"Brute: patrol")
            target_pos, thrust = self.choose_patrol_command(game_state.checkpoints)
            return self.avoid_racer(game_state, target_pos, thrust)

        log(f"Brute {self.ind}: ram foe {victim.ind}")
        return self.avoid_racer(game_state, np.rint(victim.position + victim.velocity).astype(int), "SHIELD" if self.should_shield(game_state, victim) else 100)

    def find_victim(self, game_state: GameState) -> BasePod | None:
        """Finds the nearest opponent that satisfies the victim heuristic, or returns None when no target is worth attacking."""
        victims = [foe_pod for foe_pod in game_state.foe_pods if self.is_victim(foe_pod, game_state.checkpoints)]
        return min(victims, key=lambda foe_pod: linalg.norm(foe_pod.position - self.position), default=None)

    def is_victim(self, foe_pod: BasePod, checkpoints: list[NDArray[int]]) -> bool:
        """Checks whether an opponent is close to the track, within aggro range, and moving toward the brute.
        Moving toward the brute is detected by a positive dot product between foe velocity and the vector from foe to brute.
        """
        track_distance = self.get_track_distance(foe_pod.position, checkpoints)
        pod_distance = linalg.norm(foe_pod.position - self.position)
        approach_speed = np.dot(foe_pod.velocity, self.position - foe_pod.position)
        return track_distance <= TRACK_DISTANCE and pod_distance <= AGGRO_DISTANCE and approach_speed > 0

    @staticmethod
    def get_track_distance(position: NDArray[float], checkpoints: list[NDArray[int]]) -> float:
        """Returns the shortest distance from a position to any finite segment joining consecutive checkpoint centers."""
        return min(BrutePod.get_segment_distance(position, checkpoints[checkpoint_ind], checkpoints[(checkpoint_ind + 1) % len(checkpoints)])
                   for checkpoint_ind in range(len(checkpoints)))

    @staticmethod
    def get_segment_distance(position: NDArray[float], start: NDArray[int], end: NDArray[int]) -> float:
        """Returns distance from a position to the finite segment from start to end."""
        return linalg.norm(position - BrutePod.get_closest_point_on_segment(position, start, end))

    def choose_patrol_command(self, checkpoints: list[NDArray[int]]) -> tuple[NDArray[int], int]:
        """Chooses a simple counterflow patrol command.
        The brute aims at the end of the active segment.
        If it is facing more than PATROL_THRUST_ANGLE away from the segment direction, it turns without thrust.
        """
        segment_start, segment_end = self.get_active_segment(checkpoints)
        segment_direction = self.get_segment_direction(segment_start, segment_end)
        return np.rint(segment_end).astype(int), 0 if abs(normalize_angle(self.direction - segment_direction)) > MAX_THRUST_ANGLE else 100

    def get_active_segment(self, checkpoints: list[NDArray[int]]) -> tuple[NDArray[float], NDArray[float]]:
        """Chooses the counterflow segment the brute should currently follow.
        Each checkpoint edge is treated as a reverse-direction patrol segment. Segments whose closest point is within TRACK_DISTANCE
        are candidates. If any candidates exist, the active segment is the candidate with the largest remaining distance to its end.
        If no track segment is close, the active segment is a temporary route from the brute position to the nearest point on the track.
        The returned tuple is segment start and segment end.
        """
        segments = []
        for segment_ind in range(len(checkpoints)):
            segment_start = checkpoints[(segment_ind + 1) % len(checkpoints)]
            segment_end = checkpoints[segment_ind]
            closest_point = self.get_closest_point_on_segment(self.position, segment_start, segment_end)
            segments.append((segment_start, segment_end, closest_point, linalg.norm(self.position - closest_point),
                             linalg.norm(segment_end - closest_point)))

        candidates = [segment for segment in segments if segment[3] <= TRACK_DISTANCE]
        if candidates:
            return max(candidates, key=lambda segment: segment[4])[:2]
        return self.position, min(segments, key=lambda segment: segment[3])[2]

    @staticmethod
    def get_closest_point_on_segment(position: NDArray[float], start: NDArray[int], end: NDArray[int]) -> NDArray[float]:
        """Projects a position onto the line from start to end and clamps the projection back onto the finite segment."""
        segment = end - start
        return start + segment * np.clip(np.dot(position - start, segment) / np.dot(segment, segment), 0, 1)

    @staticmethod
    def get_segment_direction(start: NDArray[float], end: NDArray[float]) -> float:
        """Returns the bot direction angle of the vector from start to end."""
        segment = end - start
        return -math.degrees(math.atan2(segment[1], segment[0]))

    def avoid_racer(self, game_state: GameState, target_pos: NDArray[int], thrust: int | str) -> tuple[NDArray[int], int | str]:
        """Keeps the planned brute command unless its predicted next position gets too close to the racer."""
        racer = next(pod for pod in game_state.my_pods if isinstance(pod, RacerPod))
        next_distance = linalg.norm(predict_next_2(self, game_state.checkpoints, target_pos, thrust, game_state.turn_ind == 0).pod.position
                                    - racer.position - racer.velocity)
        if next_distance >= RACER_AVOID_DISTANCE:
            return target_pos, thrust

        log(f"Brute {self.ind}: avoid racer")
        return np.rint(self.position * 2 - racer.position).astype(int), 100

    def should_shield(self, game_state: GameState, victim: BasePod) -> bool:
        """Checks whether the predicted next-turn brute-victim distance is close enough to spend a shield."""
        next_distance = linalg.norm(victim.position + victim.velocity - self.position - self.velocity)
        return self.shield_cooldown == 0 and next_distance <= SHIELD_DISTANCE


@dataclass(slots=True)
class FutureState:
    """Stores one predicted turn outcome.
    moves is the cumulative direction delta and thrust sequence used to reach pod.
    passed_checkpoints counts how many checkpoints were advanced across that sequence.
    """
    moves: list[float]
    pod: BasePod
    passed_checkpoints: int

    def get_score(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns the optimizer score: remaining checkpoint distance minus progress bonuses, with lower values better."""
        return self.pod.get_next_checkpoint_distance(checkpoints) - self.passed_checkpoints * CHECKPOINT_BONUS


@dataclass(slots=True)
class GameState:
    """Stores all turn data needed by decision logic: race metadata, our pods, opponent pods and shared boosts."""
    turn_ind: int
    laps: int
    checkpoints: list[NDArray[int]]
    my_pods: list[RacerPod | BrutePod]
    foe_pods: list[BasePod]
    boosts: int

    def log(self):
        """Prints our pods first and opponent pods second in compact debugging format."""
        log("My pods:")
        for pod in self.my_pods:
            pod.log()
        log("Enemy pods:")
        for pod in self.foe_pods:
            pod.log()


def main():
    """Reads initialization once, then repeats server turn read, debug logging, decision and command output."""
    game_state = read_initial_game_state()
    while True:
        game_state = update_game_state(game_state)
        game_state.log()

        for target_pos, thrust in choose_move(game_state):
            print(*target_pos, thrust)


def read_initial_game_state() -> GameState:
    """Reads race constants sent before the turn loop and creates the empty turn -1 state."""
    laps = int(input())
    checkpoint_count = int(input())
    checkpoints = [np.array(tuple(map(int, input().split()))) for _ in range(checkpoint_count)]
    return GameState(-1, laps, checkpoints, [], [], 1)


def update_game_state(prev_game_state: GameState) -> GameState:
    """Reads all server pod lines for the next turn while preserving race constants, boost count and shield cooldowns."""
    if prev_game_state.turn_ind == -1:
        our_pods = [read_pod(0, RacerPod), read_pod(1, BrutePod)]
    else:
        our_pods = [read_pod(0, RacerPod, max(0, prev_game_state.my_pods[0].shield_cooldown - 1)),
                    read_pod(1, BrutePod, max(0, prev_game_state.my_pods[1].shield_cooldown - 1))]
    foe_pods = [read_pod(pod_ind, BasePod) for pod_ind in range(2)]
    return GameState(prev_game_state.turn_ind + 1, prev_game_state.laps, prev_game_state.checkpoints, our_pods, foe_pods, prev_game_state.boosts)


def read_pod(pod_ind: int, pod_type: type[BasePod], shield_cooldown: int = 0) -> BasePod:
    """Parses one six-integer server pod line and converts the server angle convention into the bot convention."""
    x, y, vx, vy, angle, next_checkpoint_ind = map(int, input().split())
    return pod_type(pod_ind, np.array((x, y), dtype=float), np.array((vx, vy), dtype=float), normalize_angle(-angle), next_checkpoint_ind, shield_cooldown)


def choose_move(game_state: GameState) -> list[tuple[NDArray[int], int | str]]:
    """Chooses both pod commands and applies command side effects to shared boosts and pod shield cooldowns."""
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
    """Applies alternating direction delta and thrust pairs one turn at a time.
    The returned list contains the FutureState after each predicted turn, with cumulative moves and checkpoint passes.
    """
    future_states = []
    for move_ind in range(0, len(moves), 2):
        next_state = predict_next(future_states[-1].pod if future_states else current, checkpoints, moves[move_ind], moves[move_ind + 1],
                                  first_turn and move_ind == 0)
        future_states.append(FutureState((future_states[-1].moves if future_states else []) + next_state.moves, next_state.pod,
                                         (future_states[-1].passed_checkpoints if future_states else 0) + next_state.passed_checkpoints))
    return future_states


def predict_next_2(current: BasePod, checkpoints: list[NDArray[int]], target_pos: NDArray[int], thrust: float | str, first_turn: bool = False) \
    -> FutureState:
    """Predicts one turn from a command target point instead of an explicit direction delta.
    SHIELD is modeled as zero thrust because acceleration is skipped.
    """
    target_delta = target_pos - current.position
    target_direction = -math.degrees(math.atan2(target_delta[1], target_delta[0]))
    return predict_next(current, checkpoints, normalize_angle(target_direction - current.direction), 0 if thrust == "SHIELD" else thrust, first_turn)


def predict_next(current: BasePod, checkpoints: list[NDArray[int]], direction_delta: float, thrust: float, first_turn: bool = False) -> FutureState:
    """Predicts one turn without collisions using the Codingame movement order.
    The move is constrained, direction is updated, acceleration is added to velocity, position advances, checkpoints are counted at
    the final position, and drag is applied to velocity. Predicted position and velocity remain floats.
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
    """Clips a move vector to model limits.
    The first direction delta can be any normalized angle on the first turn. All other direction deltas are clipped to +/-18 degrees,
    and all thrust coordinates are clipped to the base thrust range from 0 to 100.
    """
    moves = np.array(moves).copy()
    moves[0] = normalize_angle(moves[0]) if first_turn else np.clip(moves[0], -MAX_TURN_DEG, MAX_TURN_DEG)
    moves[2::2] = np.clip(moves[2::2], -MAX_TURN_DEG, MAX_TURN_DEG)
    moves[1::2] = np.clip(moves[1::2], 0, 100)
    return moves


def normalize_angle(angle: float | NDArray[float]) -> float | NDArray[float]:
    """Maps degrees into the half-open range [-180, 180), preserving numpy arrays elementwise."""
    return (angle + 180) % 360 - 180


def log(msg: str):
    """Writes a debug line to stderr so stdout stays reserved for game commands."""
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
