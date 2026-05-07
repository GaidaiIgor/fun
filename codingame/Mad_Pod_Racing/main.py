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
COLLISION_RADIUS = 800
BOOST_THRUST = 650
MAX_TURN_DEG = 18

# Common behavior constants
OPTIMIZER_CHECKPOINT_RADIUS = 590
TARGET_DISTANCE = 10000
CHECKPOINT_BONUS = 20000

# Racer behavior constants
PREDICT_TURNS = 2
BOOST_ANGLE_TOL = 1
BOOST_MIN_DIST = 5000

# Brute behavior constants
MAX_CHARGE_ANGLE = 45
AHEAD_DIST = 2000

# Debug
DEBUG = True

@dataclass(slots=True)
class BasePod:
    """Stores server-visible pod state shared by racers, brutes and opponents.
    Direction is the bot angle in degrees from the positive x-axis, with positive angles pointing toward decreasing screen y.
    next_checkpoint_ind is the checkpoint the pod must enter next, and checkpoint_passes counts passed checkpoints observed across turns.
    """
    ind: int
    position: NDArray[float]
    velocity: NDArray[float]
    direction: float
    next_checkpoint_ind: int
    passed_checkpoints: int = 0

    def get_next_checkpoint_distance(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns Euclidean distance from the pod center to the checkpoint indexed by next_checkpoint_ind."""
        return linalg.norm(checkpoints[self.next_checkpoint_ind] - self.position)

    def get_race_progress(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns a monotonic progress estimate from passed checkpoint count and distance left to the next checkpoint."""
        return self.passed_checkpoints * CHECKPOINT_BONUS - self.get_next_checkpoint_distance(checkpoints)

    def get_direction_target(self, direction: float) -> NDArray[int]:
        """Converts a bot direction angle into a far integer point that Codingame accepts as a command target."""
        direction_rad = math.radians(direction)
        return np.rint(self.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * TARGET_DISTANCE).astype(int)

    def log(self):
        """Prints index, position, velocity, direction and next checkpoint for debugging."""
        log(f"{self.ind}: pos=({self.position[0]}, {self.position[1]}); vel=({self.velocity[0]}, {self.velocity[1]}); "
            f"dir={self.direction:g}; CP ind={self.next_checkpoint_ind}; CP passed={self.passed_checkpoints}")


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
    """Represents our disruptive pod, which actively moves toward the lead enemy or a point in front of it."""

    def choose_command(self, game_state: GameState, racer_command: tuple[NDArray[int], int | str] | None = None) -> tuple[NDArray[int], int | str]:
        """Chooses a brute command against the first enemy pod, using SHIELD when next-turn motion predicts impact."""
        enemy = self.get_lead_enemy(game_state)
        target_pos = self.choose_target(game_state, enemy, racer_command)
        return target_pos, "SHIELD" if self.does_next_motion_collide(game_state, (target_pos, "SHIELD"), enemy) else 100

    @staticmethod
    def get_lead_enemy(game_state: GameState) -> BasePod:
        """Returns the opponent pod with the greatest observed race progress."""
        return max(game_state.foe_pods, key=lambda pod: pod.get_race_progress(game_state.checkpoints))

    def choose_target(self, game_state: GameState, enemy: BasePod, racer_command: tuple[NDArray[int], int | str] | None) -> NDArray[int]:
        """Chooses the main target segment from the brute to the enemy or to an ahead point, then applies racer avoidance."""
        if abs(normalize_angle(self.get_segment_direction(enemy.position, self.position) - enemy.direction)) <= MAX_CHARGE_ANGLE:
            log(f"Brute: direct foe {enemy.ind}")
            target_pos = np.rint(enemy.position).astype(int)
        else:
            log(f"Brute: ahead foe {enemy.ind}")
            enemy_direction = math.radians(enemy.direction)
            target_pos = np.rint(enemy.position + np.array((math.cos(enemy_direction), -math.sin(enemy_direction))) * AHEAD_DIST).astype(int)

        return target_pos if racer_command is None else self.avoid_racer(game_state, target_pos, racer_command)

    def avoid_racer(self, game_state: GameState, target_pos: NDArray[int], racer_command: tuple[NDArray[int], int | str]) -> NDArray[int]:
        """Returns the closest far target line toward target_pos that avoids the racer next-turn motion corridor."""
        direct_direction = self.get_segment_direction(self.position, target_pos)
        for direction_offset in range(181):
            directions = [direct_direction] if direction_offset == 0 else [direct_direction + direction_offset, direct_direction - direction_offset]
            candidates = [self.get_direction_target(normalize_angle(direction)) for direction in directions]
            candidates = [candidate for candidate in candidates
                          if not self.does_next_motion_collide(game_state, (candidate, 100), game_state.my_pods[0], racer_command)]
            if candidates:
                return min(candidates, key=lambda candidate: linalg.norm(candidate - target_pos))
        log("Brute: no racer-safe direction found")
        return np.rint(target_pos).astype(int)

    def does_next_motion_collide(self, game_state: GameState, my_command: tuple[NDArray[int], int | str], pod: BasePod,
                                 pod_command: tuple[NDArray[int], int | str] | None = None) -> bool:
        """Checks whether the brute planned next-turn segment comes within collision distance of another pod next-turn segment.
        The racer uses its planned command; opponents are predicted as if they keep direction and use thrust 100.
        """
        if pod_command is None:
            pod_end = predict_next(pod, game_state.checkpoints, 0, 100).pod.position
        else:
            pod_end = predict_next_2(pod, game_state.checkpoints, pod_command[0], pod_command[1], game_state.turn_ind == 0).pod.position
        my_end = predict_next_2(self, game_state.checkpoints, my_command[0], my_command[1], game_state.turn_ind == 0).pod.position
        return self.get_min_approach_distance(self.position, my_end, pod.position, pod_end) <= COLLISION_RADIUS

    @staticmethod
    def get_min_approach_distance(start_1: NDArray[float], end_1: NDArray[float], start_2: NDArray[float], end_2: NDArray[float]) -> float:
        """Returns the closest synchronized distance between two points moving linearly from start to end over one turn."""
        relative_position = start_1 - start_2
        relative_velocity = end_1 - start_1 - end_2 + start_2
        if not np.any(relative_velocity):
            return linalg.norm(relative_position)
        closest_time = np.clip(-np.dot(relative_position, relative_velocity) / np.dot(relative_velocity, relative_velocity), 0, 1)
        return linalg.norm(relative_position + relative_velocity * closest_time)

    @staticmethod
    def get_segment_direction(start: NDArray[float], end: NDArray[float]) -> float:
        """Returns the bot direction angle of the vector from start to end."""
        segment = end - start
        return -math.degrees(math.atan2(segment[1], segment[0]))


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
    """Reads all server pod lines for the next turn while preserving race constants and boost count."""
    if prev_game_state.turn_ind == -1:
        our_pods = [read_pod(0, RacerPod, prev_game_state), read_pod(1, BrutePod, prev_game_state)]
        foe_pods = [read_pod(pod_ind, BasePod, prev_game_state) for pod_ind in range(2)]
    else:
        our_pods = [read_pod(0, RacerPod, prev_game_state), read_pod(1, BrutePod, prev_game_state)]
        foe_pods = [read_pod(pod_ind, BasePod, prev_game_state) for pod_ind in range(2)]
    return GameState(prev_game_state.turn_ind + 1, prev_game_state.laps, prev_game_state.checkpoints, our_pods, foe_pods, prev_game_state.boosts)


def read_pod(pod_ind: int, pod_type: type[BasePod], prev_game_state: GameState) -> BasePod:
    """Parses one six-integer server pod line and converts the server angle convention into the bot convention."""
    x, y, vx, vy, angle, next_checkpoint_ind = map(int, input().split())
    if prev_game_state.turn_ind == -1:
        checkpoint_passes = 0
    else:
        prev_pods = prev_game_state.my_pods if pod_type is not BasePod else prev_game_state.foe_pods
        passed_checkpoints = (next_checkpoint_ind - prev_pods[pod_ind].next_checkpoint_ind) % len(prev_game_state.checkpoints)
        checkpoint_passes = prev_pods[pod_ind].passed_checkpoints + passed_checkpoints
    return pod_type(pod_ind, np.array((x, y), dtype=float), np.array((vx, vy), dtype=float), normalize_angle(-angle), next_checkpoint_ind, checkpoint_passes)


def choose_move(game_state: GameState) -> list[tuple[NDArray[int], int | str]]:
    """Chooses both pod commands and applies command side effects to shared boosts."""
    commands = [game_state.my_pods[0].choose_move(game_state)[:2]]
    commands.append(game_state.my_pods[1].choose_command(game_state, commands[0]))
    for _, thrust in commands:
        if thrust == "BOOST":
            game_state.boosts -= 1
    return commands


def predict_turns(current: BasePod, checkpoints: list[NDArray[int]], moves: list[float] | NDArray[float], first_turn: bool = False) -> list[FutureState]:
    """Applies alternating direction delta and thrust pairs one turn at a time.
    The returned list contains the FutureState after each predicted turn, with cumulative moves and checkpoint passes.
    """
    future_states = []
    for move_ind in range(0, len(moves), 2):
        next_state = predict_next(future_states[-1].pod if future_states else current, checkpoints, moves[move_ind], moves[move_ind + 1],
                                  first_turn and move_ind == 0, OPTIMIZER_CHECKPOINT_RADIUS)
        future_states.append(FutureState((future_states[-1].moves if future_states else []) + next_state.moves, next_state.pod,
                                         (future_states[-1].passed_checkpoints if future_states else 0) + next_state.passed_checkpoints))
    return future_states


def predict_next_2(current: BasePod, checkpoints: list[NDArray[int]], target_pos: NDArray[int], thrust: float | str, first_turn: bool = False) \
    -> FutureState:
    """Predicts one turn from a command target point instead of an explicit direction delta.
    Command thrust strings are passed through to predict_next so move constraint logic can preserve them.
    """
    target_delta = target_pos - current.position
    target_direction = -math.degrees(math.atan2(target_delta[1], target_delta[0]))
    return predict_next(current, checkpoints, normalize_angle(target_direction - current.direction), thrust, first_turn)


def predict_next(current: BasePod, checkpoints: list[NDArray[int]], direction_delta: float, thrust: float | str, first_turn: bool = False,
                 checkpoint_radius: int = CHECKPOINT_RADIUS) -> FutureState:
    """Predicts one turn without collisions using the Codingame movement order.
    The move is constrained, direction is updated, acceleration is added to velocity, position advances, checkpoints are counted at
    the final position, and drag is applied to velocity. BOOST is 650 acceleration, and SHIELD is 0 acceleration.
    """
    direction_delta, thrust = constrain_moves([direction_delta, thrust], first_turn)
    thrust = BOOST_THRUST if thrust == "BOOST" else 0 if thrust == "SHIELD" else thrust
    next_direction = normalize_angle(current.direction + direction_delta)
    next_direction_rad = math.radians(next_direction)
    acceleration = np.array((math.cos(next_direction_rad), -math.sin(next_direction_rad))) * thrust
    velocity = current.velocity + acceleration
    segment_end = current.position + velocity
    passed_checkpoints = 0
    while passed_checkpoints < len(checkpoints) \
            and linalg.norm(checkpoints[(current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)] - segment_end) <= checkpoint_radius:
        passed_checkpoints += 1

    velocity = velocity * DRAG
    next_checkpoint_ind = (current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)
    pod = type(current)(current.ind, segment_end, velocity, next_direction, next_checkpoint_ind, current.passed_checkpoints + passed_checkpoints)
    return FutureState([direction_delta, thrust], pod, passed_checkpoints)


def constrain_moves(moves: list[float | str] | NDArray[float], first_turn: bool = False) -> list[float | str] | NDArray[float]:
    """Clips a move vector to model limits.
    The first direction delta can be any normalized angle on the first turn. All other direction deltas are clipped to +/-18 degrees,
    base thrust coordinates are clipped to 0..100, and command thrust strings are preserved for predict_next to interpret.
    """
    moves[0] = normalize_angle(moves[0]) if first_turn else np.clip(moves[0], -MAX_TURN_DEG, MAX_TURN_DEG)
    for move_ind in range(1, len(moves), 2):
        if not isinstance(moves[move_ind], str):
            moves[move_ind] = np.clip(moves[move_ind], 0, 100)
    for move_ind in range(2, len(moves), 2):
        moves[move_ind] = np.clip(moves[move_ind], -MAX_TURN_DEG, MAX_TURN_DEG)
    return moves


def normalize_angle(angle: float | NDArray[float]) -> float | NDArray[float]:
    """Maps degrees into the half-open range [-180, 180), preserving numpy arrays elementwise."""
    return (angle + 180) % 360 - 180


def log(msg: str):
    """Writes a debug line to stderr so stdout stays reserved for game commands."""
    if DEBUG:
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
