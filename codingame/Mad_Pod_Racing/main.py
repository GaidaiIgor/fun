"""Runs a Mad Pod Racing bot with one optimized racer pod and one heuristic brute pod."""

from __future__ import annotations

import math
import sys
import time
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
RACER_PREDICT_TURNS = 2
BOOST_ANGLE_TOL = 1
BOOST_MIN_DIST = 5000

# Brute behavior constants
MAX_POSITION_ANGLE = 90
MAX_DIRECTION_ANGLE = 45
ATTACK_DIST_FRAC = 0.5
PARKING_DIST = 1000
RACER_AVOID_RADIUS = 950
BRUTE_PREDICT_TURNS = 5

# Debug
DEBUG = True
TIMING = True
TIMER_START = time.perf_counter()
TIMER_LAST = TIMER_START

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

    def log(self):
        """Prints index, position, velocity, direction and next checkpoint for debugging."""
        log(f"{self.ind}: pos=({self.position[0]}, {self.position[1]}); vel=({self.velocity[0]}, {self.velocity[1]}); "
            f"dir={self.direction:g}; CP ind={self.next_checkpoint_ind}; CP passed={self.passed_checkpoints}")


@dataclass(slots=True)
class RacerPod(BasePod):
    """Represents our racing pod, which follows checkpoints in order using continuous move optimization."""

    def choose_command(self, game_state: GameState) -> tuple[float, float | str]:
        """Drops planned future moves from choose_move and returns only the command direction and thrust."""
        return self.choose_move(game_state)[:2]

    def choose_move(self, game_state: GameState) -> tuple[float, float | str, NDArray[float]]:
        """Chooses one racer command and the optimized future move sequence that produced it.
        The first turn aims straight at the next checkpoint. Later turns optimize direction delta and thrust pairs, use the first optimized
        direction delta as the command, and optionally replace thrust with BOOST.
        """
        checkpoint_delta = game_state.checkpoints[self.next_checkpoint_ind] - self.position
        checkpoint_direction = -math.degrees(math.atan2(checkpoint_delta[1], checkpoint_delta[0]))
        if game_state.turn_ind == 0:
            return checkpoint_direction, "BOOST" if self.get_next_checkpoint_distance(game_state.checkpoints) > BOOST_MIN_DIST and game_state.boosts else 100, \
                np.array((), dtype=float)

        result = self.optimize_moves(game_state.checkpoints)
        direction = normalize_angle(self.direction + result.x[0])
        thrust = result.x[1]
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

        return direction, thrust, result.x

    def optimize_moves(self, checkpoints: list[NDArray[int]]) -> OptimizeResult:
        """Optimizes alternating direction delta and thrust coordinates for the configured prediction horizon.
        The objective is the final FutureState score after applying model-level move constraints and checkpoint progress.
        result.x is constrained again after minimize returns so downstream code sees valid model inputs.
        """
        move_bounds = optimize.Bounds(np.tile(np.array((-MAX_TURN_DEG, 0), dtype=float), RACER_PREDICT_TURNS),
                                      np.tile(np.array((MAX_TURN_DEG, 100), dtype=float), RACER_PREDICT_TURNS))
        result = optimize.minimize(lambda moves: predict_turns(self, checkpoints, moves)[-1].get_score(checkpoints), self.get_optimizer_guess_moves(),
                                   method="L-BFGS-B", bounds=move_bounds, options={"maxiter": np.iinfo(np.int32).max})
        result.x = constrain_moves(result.x)
        return result

    @staticmethod
    def get_optimizer_guess_moves() -> NDArray[float]:
        """Builds the neutral optimizer seed: zero direction change and full base thrust for each predicted turn."""
        return np.tile(np.array((0, 100), dtype=float), RACER_PREDICT_TURNS)


@dataclass(slots=True)
class BrutePod(BasePod):
    """Represents our disruptive pod, which attacks the lead enemy or waits counterflow at the end of that enemy next segment."""

    def choose_command(self, game_state: GameState, racer_command: tuple[float, float | str] | None = None) -> tuple[float, float | str]:
        """Chooses the brute command against the lead enemy, applying racer avoidance and replacing thrust with SHIELD when impact is predicted."""
        enemy = self.get_lead_enemy(game_state)
        if game_state.turn_ind == 0:
            target_pos = game_state.checkpoints[(enemy.next_checkpoint_ind + 1) % len(game_state.checkpoints)]
            return get_segment_direction(self.position, target_pos), 100
        direction, thrust = self.choose_base_command(game_state, enemy)
        if racer_command is not None:
            direction, thrust_override = self.avoid_racer(game_state, direction, racer_command)
            thrust = thrust if thrust_override is None else thrust_override
        shield = self.does_next_motion_collide(game_state, (direction, "SHIELD"), enemy)
        return direction, "SHIELD" if shield else thrust

    @staticmethod
    def get_lead_enemy(game_state: GameState) -> BasePod:
        """Returns the opponent pod with the greatest observed race progress."""
        return max(game_state.foe_pods, key=lambda pod: pod.get_race_progress(game_state.checkpoints))

    def choose_base_command(self, game_state: GameState, enemy: BasePod) -> tuple[float, float]:
        """Chooses the brute direction and thrust before racer avoidance and shield override."""
        if self.is_attackable(enemy):
            log(f"Brute: attacking foe {enemy.ind}")
            return get_segment_direction(self.position, self.get_attack_target(enemy)), 100
        log(f"Brute: ambushing foe {enemy.ind}")
        return self.choose_ambush_command(game_state, enemy)

    def is_attackable(self, enemy: BasePod) -> bool:
        """Checks current attack angles and keeps predicting until either impact is found or future attack angles break."""
        if not self.has_attack_angles(enemy):
            return False
        brute = self
        foe = enemy
        for turn_ind in range(BRUTE_PREDICT_TURNS):
            next_direction = get_segment_direction(brute.position, brute.get_attack_target(foe))
            next_brute = predict_next(brute, None, normalize_angle(next_direction - brute.direction), 100).pod
            next_foe = predict_next(foe, None, 0, 100).pod
            if self.get_min_approach_distance(brute.position, next_brute.position, foe.position, next_foe.position) <= COLLISION_RADIUS:
                return True
            brute = next_brute
            foe = next_foe
            if not brute.has_attack_angles(foe):
                return False
        return True

    def has_attack_angles(self, enemy: BasePod) -> bool:
        """Checks whether enemy is in front of the brute and moving approximately opposite to it."""
        return abs(normalize_angle(get_segment_direction(self.position, enemy.position) - self.direction)) < MAX_POSITION_ANGLE and \
            abs(normalize_angle(enemy.direction - self.direction)) >= 180 - MAX_DIRECTION_ANGLE

    def get_attack_target(self, enemy: BasePod) -> NDArray[float]:
        """Returns the attack point between enemy and the brute projection onto the enemy direction line."""
        enemy_direction = math.radians(enemy.direction)
        enemy_direction_vector = np.array((math.cos(enemy_direction), -math.sin(enemy_direction)))
        return enemy.position + enemy_direction_vector * np.dot(self.position - enemy.position, enemy_direction_vector) * ATTACK_DIST_FRAC

    def choose_ambush_command(self, game_state: GameState, enemy: BasePod) -> tuple[float, float]:
        """Chooses a fallback command toward the end of enemy next active segment, or coasts while turning back along it."""
        segment_start = game_state.checkpoints[enemy.next_checkpoint_ind]
        segment_end = game_state.checkpoints[(enemy.next_checkpoint_ind + 1) % len(game_state.checkpoints)]
        segment_back_direction = get_segment_direction(segment_end, segment_start)
        if linalg.norm(self.position - segment_end) <= PARKING_DIST:
            log(f"Brute: parked foe {enemy.ind}")
            return segment_back_direction, 0
        if self.should_coast_to_turn(segment_end, segment_back_direction):
            log(f"Brute: coast segment end foe {enemy.ind}")
            return segment_back_direction, 0
        return get_segment_direction(self.position, segment_end), 100

    def should_coast_to_turn(self, target_pos: NDArray[float], target_direction: float) -> bool:
        """Checks whether the velocity ray crosses the parking spot and zero-thrust prediction ends inside it after turning time."""
        if not np.any(self.velocity):
            return False
        velocity_pos = max(0, np.dot(target_pos - self.position, self.velocity) / np.dot(self.velocity, self.velocity))
        if linalg.norm(self.position + self.velocity * velocity_pos - target_pos) > PARKING_DIST:
            return False
        turn_count = math.ceil(abs(normalize_angle(target_direction - self.direction)) / MAX_TURN_DEG)
        if turn_count == 0:
            return False
        future_states = predict_turns(self, None, np.tile(np.array((0, 0), dtype=float), turn_count))
        return linalg.norm(future_states[-1].pod.position - target_pos) <= PARKING_DIST

    def avoid_racer(self, game_state: GameState, direction: float, racer_command: tuple[float, float | str]) -> tuple[float, float | None]:
        """Returns base direction unless it threatens the racer, then tries max right and max left, else zeroes thrust."""
        racer_moves = self.predict_moves(game_state.my_pods[0], racer_command[0])
        racer_segment_end = predict_turns(game_state.my_pods[0], None, racer_moves)[-1].pod.position
        brute_moves = self.predict_moves(self, direction)
        brute_segment_end = predict_turns(self, None, brute_moves)[-1].pod.position
        distance = self.get_min_approach_distance(self.position, brute_segment_end, game_state.my_pods[0].position, racer_segment_end)
        if distance >= RACER_AVOID_RADIUS:
            return direction, None
        for candidate in (normalize_angle(self.direction - MAX_TURN_DEG), normalize_angle(self.direction + MAX_TURN_DEG)):
            brute_moves = self.predict_moves(self, candidate)
            brute_segment_end = predict_turns(self, None, brute_moves)[-1].pod.position
            distance = self.get_min_approach_distance(self.position, brute_segment_end, game_state.my_pods[0].position, racer_segment_end)
            if distance >= RACER_AVOID_RADIUS:
                return candidate, None
        log("Brute: no racer-safe direction found")
        return direction, 0

    @staticmethod
    def predict_moves(pod: BasePod, direction: float) -> list[float]:
        """Returns moves that keep aiming at the same absolute target direction for the avoidance horizon."""
        return [normalize_angle(direction - pod.direction), 100] + [0, 100] * (BRUTE_PREDICT_TURNS - 1)

    @staticmethod
    def get_point_segment_distance(point: NDArray[float], start: NDArray[float], end: NDArray[float]) -> float:
        """Returns the shortest geometric distance from point to the finite segment."""
        segment = end - start
        if not np.any(segment):
            return linalg.norm(point - start)
        segment_pos = np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0, 1)
        return linalg.norm(start + segment * segment_pos - point)

    def does_next_motion_collide(self, game_state: GameState, my_command: tuple[float, float | str], pod: BasePod,
                                 pod_command: tuple[float, float | str] | None = None) -> bool:
        """Checks whether the brute planned next-turn segment comes within collision distance of another pod next-turn segment.
        The racer uses its planned command; opponents are predicted as if they keep direction and use thrust 100.
        """
        if pod_command is None:
            pod_end = predict_next(pod, game_state.checkpoints, 0, 100).pod.position
        else:
            pod_end = predict_next(pod, game_state.checkpoints, normalize_angle(pod_command[0] - pod.direction), pod_command[1]).pod.position
        my_end = predict_next(self, game_state.checkpoints, normalize_angle(my_command[0] - self.direction), my_command[1]).pod.position
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
        reset_timer()
        game_state = update_game_state(game_state)
        log_time("read turn")
        game_state.log()
        log_time("state log")

        commands = choose_move(game_state)
        for pod, (direction, thrust) in zip(game_state.my_pods, commands):
            target_pos = get_command_target(pod.position, direction)
            print(round(target_pos[0]), round(target_pos[1]), thrust if isinstance(thrust, str) else round(thrust))
        log_time("output commands")


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


def choose_move(game_state: GameState) -> list[tuple[float, float | str]]:
    """Chooses both pod commands and applies command side effects to shared boosts."""
    commands = [game_state.my_pods[0].choose_move(game_state)[:2]]
    log_time("choose racer")
    commands.append(game_state.my_pods[1].choose_command(game_state, commands[0]))
    log_time("choose brute")
    for _, thrust in commands:
        if thrust == "BOOST":
            game_state.boosts -= 1
    return commands


def predict_turns(current: BasePod, checkpoints: list[NDArray[int]] | None, moves: list[float] | NDArray[float]) -> list[FutureState]:
    """Applies alternating direction delta and thrust pairs one turn at a time.
    The returned list contains the FutureState after each predicted turn, with cumulative moves and checkpoint passes.
    """
    future_states = []
    for move_ind in range(0, len(moves), 2):
        next_state = predict_next(future_states[-1].pod if future_states else current, checkpoints, moves[move_ind], moves[move_ind + 1],
                                  OPTIMIZER_CHECKPOINT_RADIUS)
        future_states.append(FutureState((future_states[-1].moves if future_states else []) + next_state.moves, next_state.pod,
                                         (future_states[-1].passed_checkpoints if future_states else 0) + next_state.passed_checkpoints))
    return future_states


def predict_next(current: BasePod, checkpoints: list[NDArray[int]] | None, direction_delta: float, thrust: float | str,
                 checkpoint_radius: int = CHECKPOINT_RADIUS) -> FutureState:
    """Predicts one turn without collisions using the Codingame movement order.
    The move is constrained, direction is updated, acceleration is added to velocity, position advances, checkpoints are counted at
    the final position when checkpoints are given, and drag is applied to velocity. BOOST is 650 acceleration, and SHIELD is 0 acceleration.
    """
    direction_delta, thrust = constrain_moves([direction_delta, thrust])
    thrust = BOOST_THRUST if thrust == "BOOST" else 0 if thrust == "SHIELD" else thrust
    next_direction = normalize_angle(current.direction + direction_delta)
    next_direction_rad = math.radians(next_direction)
    acceleration = np.array((math.cos(next_direction_rad), -math.sin(next_direction_rad))) * thrust
    velocity = current.velocity + acceleration
    segment_end = current.position + velocity
    passed_checkpoints = 0
    next_checkpoint_ind = current.next_checkpoint_ind
    if checkpoints is not None:
        while passed_checkpoints < len(checkpoints) \
                and linalg.norm(checkpoints[(current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)] - segment_end) <= checkpoint_radius:
            passed_checkpoints += 1
        next_checkpoint_ind = (current.next_checkpoint_ind + passed_checkpoints) % len(checkpoints)

    velocity = velocity * DRAG
    pod = type(current)(current.ind, segment_end, velocity, next_direction, next_checkpoint_ind, current.passed_checkpoints + passed_checkpoints)
    return FutureState([direction_delta, thrust], pod, passed_checkpoints)


def constrain_moves(moves: list[float | str] | NDArray[float]) -> list[float | str] | NDArray[float]:
    """Clips a move vector to model limits.
    Direction deltas are clipped to +/-18 degrees, base thrust coordinates are clipped to 0..100, and command thrust strings are
    preserved for predict_next to interpret.
    """
    moves[0] = np.clip(moves[0], -MAX_TURN_DEG, MAX_TURN_DEG)
    for move_ind in range(1, len(moves), 2):
        if not isinstance(moves[move_ind], str):
            moves[move_ind] = np.clip(moves[move_ind], 0, 100)
    for move_ind in range(2, len(moves), 2):
        moves[move_ind] = np.clip(moves[move_ind], -MAX_TURN_DEG, MAX_TURN_DEG)
    return moves


def normalize_angle(angle: float | NDArray[float]) -> float | NDArray[float]:
    """Maps degrees into the half-open range [-180, 180), preserving numpy arrays elementwise."""
    return (angle + 180) % 360 - 180


def get_segment_direction(start: NDArray[float], end: NDArray[float]) -> float:
    """Returns the bot direction angle of the vector from start to end."""
    segment = end - start
    return -math.degrees(math.atan2(segment[1], segment[0]))


def get_command_target(position: NDArray[float], direction: float) -> NDArray[float]:
    """Converts a command direction into the far target point required by Codingame output."""
    direction_rad = math.radians(direction)
    return position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * TARGET_DISTANCE


def reset_timer():
    """Starts the global per-turn debug timer."""
    global TIMER_START, TIMER_LAST
    TIMER_START = time.perf_counter()
    TIMER_LAST = TIMER_START


def log_time(msg: str):
    """Prints elapsed milliseconds since the previous timing mark and since the turn timer started."""
    global TIMER_LAST
    if TIMING:
        now = time.perf_counter()
        log(f"TIME {msg}: step={(now - TIMER_LAST) * 1000:.3g}ms total={(now - TIMER_START) * 1000:.3g}ms")
        TIMER_LAST = now


def log(msg: str):
    """Writes a debug line to stderr so stdout stays reserved for game commands."""
    if DEBUG:
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
