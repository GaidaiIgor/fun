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
        """Returns Euclidean distance from self.position to checkpoints[self.next_checkpoint_ind]."""
        return linalg.norm(checkpoints[self.next_checkpoint_ind] - self.position)

    def get_next_checkpoint_direction(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns the command direction from self.position to checkpoints[self.next_checkpoint_ind]."""
        return self.get_target_direction(checkpoints[self.next_checkpoint_ind])

    def get_race_progress(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns a monotonic progress estimate from self.passed_checkpoints and distance left in checkpoints."""
        return self.passed_checkpoints * CHECKPOINT_BONUS - self.get_next_checkpoint_distance(checkpoints)

    def get_target_direction(self, target_pos: NDArray[float]) -> float:
        """Returns the command direction from self.position to target_pos."""
        return get_segment_direction(self.position, target_pos)

    def log(self):
        """Prints index, position, velocity, direction and next checkpoint for debugging."""
        log(f"{self.ind}: pos=({self.position[0]}, {self.position[1]}); vel=({self.velocity[0]}, {self.velocity[1]}); "
            f"dir={self.direction:g}; CP ind={self.next_checkpoint_ind}; CP passed={self.passed_checkpoints}")


@dataclass(slots=True)
class RacerPod(BasePod):
    """Represents our racing pod, which follows checkpoints in order using continuous move optimization."""

    def choose_command(self, game_state: GameState) -> tuple[float, float | str, list[FutureState]]:
        """Chooses one racer command and the optimized future move sequence that produced it.
        game_state supplies checkpoints, turn index and boost availability. The first turn aims straight at the next checkpoint.
        Later turns optimize direction delta and thrust pairs, unless the pre-optimization BOOST check selects a direct boost.
        Returns command direction, thrust and racer trajectory.
        """
        checkpoint_direction = self.get_next_checkpoint_direction(game_state.checkpoints)
        if game_state.turn_ind == 0:
            return checkpoint_direction, "BOOST" if self.should_boost(game_state, checkpoint_direction) else 100, [FutureState([], self)]
        if self.should_boost(game_state, checkpoint_direction):
            next_state = predict_next(self, game_state.checkpoints, normalize_angle(checkpoint_direction - self.direction), "BOOST")
            return checkpoint_direction, "BOOST", [FutureState([], self), next_state]

        log_time("Racer: begin optimization")
        result = self.optimize_moves(game_state.checkpoints)
        log_time("Racer: end optimization")
        future_states = predict_turns(self, game_state.checkpoints, result.x)

        log(f"Pod {self.ind} move:")
        opt_moves = ", ".join(f"{value:.3g}" for value in result.x)
        log(f"opt moves=[{opt_moves}]; score={round(result.fun)}")
        log(f"opt success={result.success}; nfev={result.nfev}; message={result.message}")
        log("Predicted:")
        for next_state in future_states:
            log(f"pos={next_state.pod.position}; CP={next_state.pod.next_checkpoint_ind}")

        return normalize_angle(self.direction + result.x[0]), result.x[1], [FutureState([], self)] + future_states

    def should_boost(self, game_state: GameState, checkpoint_direction: float) -> bool:
        """Returns true when game_state and checkpoint_direction make this racer spend boost on a long checkpoint approach."""
        return self.get_next_checkpoint_distance(game_state.checkpoints) > BOOST_MIN_DIST and game_state.boosts and \
            (game_state.turn_ind == 0 or abs(normalize_angle(self.direction - checkpoint_direction)) <= BOOST_ANGLE_TOL)

    def optimize_moves(self, checkpoints: list[NDArray[int]]) -> OptimizeResult:
        """Optimizes racer moves for checkpoints using the fast scalar objective.
        :param checkpoints: Race checkpoints used to score predicted progress.
        :return: Scipy optimization result with result.x constrained to legal model coordinates.
        """
        move_bounds = optimize.Bounds(np.tile(np.array((-MAX_TURN_DEG, 0), dtype=float), RACER_PREDICT_TURNS),
                                      np.tile(np.array((MAX_TURN_DEG, 100), dtype=float), RACER_PREDICT_TURNS))
        result = optimize.minimize(lambda moves: get_optimizer_score(self, checkpoints, moves), self.get_optimizer_guess_moves(),
                                   method="L-BFGS-B", bounds=move_bounds, options={"maxiter": np.iinfo(np.int32).max})
        result.x = constrain_moves(result.x)
        return result

    @staticmethod
    def get_optimizer_guess_moves() -> NDArray[float]:
        """Returns the neutral optimizer seed: zero direction change and full base thrust for each predicted turn."""
        return np.tile(np.array((0, 100), dtype=float), RACER_PREDICT_TURNS)


@dataclass(slots=True)
class BrutePod(BasePod):
    """Represents our disruptive pod, which attacks the lead enemy or waits counterflow at the end of that enemy next segment."""

    def choose_command(self, game_state: GameState, racer_trajectory: list[FutureState] | None = None) -> tuple[float, float | str]:
        """Chooses the brute command from game_state, using racer_trajectory for avoidance when provided. Returns command direction and thrust."""
        enemy = self.get_lead_enemy(game_state)
        if game_state.turn_ind == 0:
            return self.get_target_direction(game_state.checkpoints[(enemy.next_checkpoint_ind + 1) % len(game_state.checkpoints)]), 100
        brute_trajectory, enemy_trajectory = self.choose_base_command(game_state, enemy)
        if racer_trajectory is not None:
            brute_trajectory = self.avoid_racer(game_state, brute_trajectory, enemy_trajectory, racer_trajectory)
        direction = normalize_angle(self.direction + brute_trajectory[1].moves[0])
        thrust = brute_trajectory[1].moves[1]
        shield = self.get_collision_time(brute_trajectory, enemy_trajectory, COLLISION_RADIUS) <= 1
        return direction, "SHIELD" if shield else thrust

    @staticmethod
    def get_lead_enemy(game_state: GameState) -> BasePod:
        """Returns the opponent pod in game_state with the greatest observed race progress."""
        return max(game_state.foe_pods, key=lambda pod: pod.get_race_progress(game_state.checkpoints))

    def choose_base_command(self, game_state: GameState, enemy: BasePod) -> tuple[list[FutureState], list[FutureState]]:
        """Chooses brute trajectory from game_state against enemy before racer avoidance.
        Returns brute trajectory and foe trajectory.
        """
        foe_trajectory = extend_checkpoint_trajectory([FutureState([], enemy)], game_state.checkpoints, BRUTE_PREDICT_TURNS)
        brute_trajectory = self.get_attack_trajectory(foe_trajectory)
        if self.is_attackable(foe_trajectory, brute_trajectory):
            log(f"Brute: attacking foe {enemy.ind}")
            return brute_trajectory, foe_trajectory
        log(f"Brute: ambushing foe {enemy.ind}")
        return self.get_ambush_trajectory(game_state, enemy), foe_trajectory

    def get_attack_trajectory(self, enemy_trajectory: list[FutureState]) -> list[FutureState]:
        """Builds brute attack states from enemy_trajectory. Returns predicted states including the start."""
        trajectory = [FutureState([], self)]
        for turn_ind in range(BRUTE_PREDICT_TURNS):
            pod = trajectory[-1].pod
            direction = pod.get_target_direction(pod.get_attack_target(enemy_trajectory[turn_ind].pod))
            next_state = predict_next(pod, None, normalize_angle(direction - pod.direction), 100)
            next_state.moves = trajectory[-1].moves + next_state.moves
            trajectory.append(next_state)
        return trajectory

    def is_attackable(self, foe_trajectory: list[FutureState], brute_trajectory: list[FutureState]) -> bool:
        """Checks target angles using foe_trajectory and brute_trajectory predictions. Returns true when brute should attack."""
        for turn_ind in range(BRUTE_PREDICT_TURNS):
            if self.get_min_approach_distance(brute_trajectory[turn_ind].pod.position, brute_trajectory[turn_ind + 1].pod.position,
                                              foe_trajectory[turn_ind].pod.position, foe_trajectory[turn_ind + 1].pod.position) <= COLLISION_RADIUS:
                return True
        return brute_trajectory[-1].pod.has_attack_angles(foe_trajectory[-1].pod)

    def has_attack_angles(self, enemy: BasePod) -> bool:
        """Returns true when enemy is in front of the brute and moving approximately opposite to it."""
        return abs(normalize_angle(self.get_target_direction(enemy.position) - self.direction)) < MAX_POSITION_ANGLE and \
            abs(normalize_angle(enemy.direction - self.direction)) >= 180 - MAX_DIRECTION_ANGLE

    def get_attack_target(self, enemy: BasePod) -> NDArray[float]:
        """Returns the attack point between enemy and the brute projection onto the enemy direction line."""
        enemy_direction = math.radians(enemy.direction)
        enemy_direction_vector = np.array((math.cos(enemy_direction), -math.sin(enemy_direction)))
        return enemy.position + enemy_direction_vector * max(0, np.dot(self.position - enemy.position, enemy_direction_vector)) * ATTACK_DIST_FRAC

    def get_ambush_trajectory(self, game_state: GameState, enemy: BasePod) -> list[FutureState]:
        """Chooses fallback trajectory from game_state against enemy segment. Returns full trajectory."""
        segment_start = game_state.checkpoints[enemy.next_checkpoint_ind]
        segment_end = game_state.checkpoints[(enemy.next_checkpoint_ind + 1) % len(game_state.checkpoints)]
        trajectory = [FutureState([], self)]
        for turn_ind in range(BRUTE_PREDICT_TURNS):
            pod = trajectory[-1].pod
            segment_start_direction = pod.get_target_direction(segment_start)
            if linalg.norm(pod.position - segment_end) <= PARKING_DIST or pod.should_coast_to_turn(segment_end, segment_start_direction):
                next_direction = segment_start_direction
                next_thrust = 0
            else:
                next_direction = pod.get_target_direction(segment_end)
                next_thrust = 100
            next_state = predict_next(pod, None, normalize_angle(next_direction - pod.direction), next_thrust)
            next_state.moves = trajectory[-1].moves + next_state.moves
            trajectory.append(next_state)
        return trajectory

    def should_coast_to_turn(self, target_pos: NDArray[float], target_direction: float) -> bool:
        """Checks whether velocity crosses target_pos and zero-thrust prediction ends there after turning to target_direction.
        Returns true when brute should coast.
        """
        turn_count = math.ceil(abs(normalize_angle(target_direction - self.direction)) / MAX_TURN_DEG)
        if turn_count == 0:
            return False
        future_states = predict_turns(self, None, np.tile(np.array((0, 0), dtype=float), turn_count))
        return linalg.norm(future_states[-1].pod.position - target_pos) <= PARKING_DIST

    def avoid_racer(self, game_state: GameState, brute_trajectory: list[FutureState], foe_trajectory: list[FutureState], racer_trajectory: list[FutureState]) \
        -> list[FutureState]:
        """Checks brute_trajectory against racer_trajectory and foe_trajectory from game_state.
        Returns the chosen brute trajectory after trying racer avoidance trajectories.
        """
        racer_trajectory = extend_checkpoint_trajectory(racer_trajectory.copy(), game_state.checkpoints, BRUTE_PREDICT_TURNS)
        racer_collision_time = self.get_collision_time(brute_trajectory, racer_trajectory, RACER_AVOID_RADIUS)
        if math.isinf(racer_collision_time):
            return brute_trajectory
        enemy_collision_time = self.get_collision_time(brute_trajectory, foe_trajectory, COLLISION_RADIUS)
        if enemy_collision_time < racer_collision_time:
            return brute_trajectory

        log("Brute: avoiding racer")
        thrust = brute_trajectory[1].moves[1]
        for direction_delta in (-MAX_TURN_DEG, MAX_TURN_DEG):
            candidate_trajectory = [FutureState([], self)] + predict_turns(self, None, [direction_delta, thrust] * BRUTE_PREDICT_TURNS)
            if math.isinf(self.get_collision_time(candidate_trajectory, racer_trajectory, RACER_AVOID_RADIUS)):
                return candidate_trajectory
        log("Brute: no racer-safe direction found")
        return [FutureState([], self)] + predict_turns(self, None, [brute_trajectory[1].moves[0], 0] + [0, 0] * (BRUTE_PREDICT_TURNS - 1))

    @staticmethod
    def get_collision_time(trajectory_1: list[FutureState], trajectory_2: list[FutureState], radius: float) -> float:
        """Returns the first fractional turn time when trajectory_1 and trajectory_2 enter radius, or infinity if they never do."""
        for turn_ind in range(len(trajectory_1) - 1):
            relative_position = trajectory_1[turn_ind].pod.position - trajectory_2[turn_ind].pod.position
            relative_velocity = trajectory_1[turn_ind + 1].pod.position - trajectory_1[turn_ind].pod.position \
                - trajectory_2[turn_ind + 1].pod.position + trajectory_2[turn_ind].pod.position
            a = np.dot(relative_velocity, relative_velocity)
            c = np.dot(relative_position, relative_position) - radius ** 2
            if c <= 0:
                return turn_ind
            if a:
                b = 2 * np.dot(relative_position, relative_velocity)
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    collision_time = (-b - math.sqrt(discriminant)) / (2 * a)
                    if 0 <= collision_time <= 1:
                        return turn_ind + collision_time
        return math.inf

    @staticmethod
    def get_point_segment_distance(point: NDArray[float], start: NDArray[float], end: NDArray[float]) -> float:
        """Returns the shortest geometric distance from point to the finite segment from start to end."""
        segment = end - start
        if not np.any(segment):
            return linalg.norm(point - start)
        segment_pos = np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0, 1)
        return linalg.norm(start + segment * segment_pos - point)

    @staticmethod
    def get_min_approach_distance(start_1: NDArray[float], end_1: NDArray[float], start_2: NDArray[float], end_2: NDArray[float]) -> float:
        """Returns closest synchronized distance between points moving from start_1 to end_1 and from start_2 to end_2 over one turn."""
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
    """
    moves: list[float | str]
    pod: BasePod

    def get_score(self, checkpoints: list[NDArray[int]]) -> float:
        """Returns the optimizer score from checkpoints: remaining distance minus progress bonuses, with lower values better."""
        return self.pod.get_next_checkpoint_distance(checkpoints) - self.pod.passed_checkpoints * CHECKPOINT_BONUS


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

        commands = choose_pods_move(game_state)
        for pod, (direction, thrust) in zip(game_state.my_pods, commands):
            target_pos = get_command_target(pod.position, direction)
            print(round(target_pos[0]), round(target_pos[1]), thrust if isinstance(thrust, str) else round(thrust))
        log_time("output commands")


def read_initial_game_state() -> GameState:
    """Reads race constants sent before the turn loop. Returns the empty turn -1 game state."""
    laps = int(input())
    checkpoint_count = int(input())
    checkpoints = [np.array(tuple(map(int, input().split()))) for _ in range(checkpoint_count)]
    return GameState(-1, laps, checkpoints, [], [], 1)


def update_game_state(prev_game_state: GameState) -> GameState:
    """Reads server pod lines after prev_game_state while preserving race constants and boost count. Returns the updated game state."""
    if prev_game_state.turn_ind == -1:
        our_pods = [read_pod(0, RacerPod, prev_game_state), read_pod(1, BrutePod, prev_game_state)]
        foe_pods = [read_pod(pod_ind, BasePod, prev_game_state) for pod_ind in range(2)]
    else:
        our_pods = [read_pod(0, RacerPod, prev_game_state), read_pod(1, BrutePod, prev_game_state)]
        foe_pods = [read_pod(pod_ind, BasePod, prev_game_state) for pod_ind in range(2)]
    return GameState(prev_game_state.turn_ind + 1, prev_game_state.laps, prev_game_state.checkpoints, our_pods, foe_pods, prev_game_state.boosts)


def read_pod(pod_ind: int, pod_type: type[BasePod], prev_game_state: GameState) -> BasePod:
    """Parses pod_ind as pod_type using prev_game_state to update checkpoint passes. Returns the parsed pod state."""
    x, y, vx, vy, angle, next_checkpoint_ind = map(int, input().split())
    if prev_game_state.turn_ind == -1:
        checkpoint_passes = 0
    else:
        prev_pods = prev_game_state.my_pods if pod_type is not BasePod else prev_game_state.foe_pods
        passed_checkpoints = (next_checkpoint_ind - prev_pods[pod_ind].next_checkpoint_ind) % len(prev_game_state.checkpoints)
        checkpoint_passes = prev_pods[pod_ind].passed_checkpoints + passed_checkpoints
    return pod_type(pod_ind, np.array((x, y), dtype=float), np.array((vx, vy), dtype=float), normalize_angle(-angle), next_checkpoint_ind, checkpoint_passes)


def choose_pods_move(game_state: GameState) -> list[tuple[float, float | str]]:
    """Chooses both pod commands from game_state and applies boost side effects. Returns one command per controlled pod."""
    racer_response = game_state.my_pods[0].choose_command(game_state)
    commands = [racer_response[:2]]
    if racer_response[1] == "BOOST":
        game_state.boosts -= 1
    log_time("choose racer")

    brute_command = game_state.my_pods[1].choose_command(game_state, racer_response[2])
    commands.append(brute_command)
    log_time("choose brute")
    return commands


def predict_turns(current: BasePod, checkpoints: list[NDArray[int]] | None, moves: list[float] | NDArray[float]) -> list[FutureState]:
    """Applies alternating direction delta and thrust pairs one turn at a time.
    current is the start pod, checkpoints may be None, and moves is the command vector. Returns FutureState after each predicted turn.
    """
    future_states = []
    for move_ind in range(0, len(moves), 2):
        next_state = predict_next(future_states[-1].pod if future_states else current, checkpoints, moves[move_ind], moves[move_ind + 1],
                                  OPTIMIZER_CHECKPOINT_RADIUS)
        if future_states:
            next_state.moves = future_states[-1].moves + next_state.moves
        future_states.append(next_state)
    return future_states


def get_optimizer_score(current: BasePod, checkpoints: list[NDArray[int]], moves: list[float] | NDArray[float]) -> float:
    """Scores moves from current against checkpoints without allocating FutureState objects.
    :param current: Pod state at the start of prediction.
    :param checkpoints: Race checkpoints used for checkpoint progress and final distance.
    :param moves: Alternating direction delta and thrust coordinates to score.
    :return: Optimizer score matching FutureState.get_score for the final predicted state.
    """
    x, y = current.position
    vx, vy = current.velocity
    direction = current.direction
    next_checkpoint_ind = current.next_checkpoint_ind
    passed_checkpoints = current.passed_checkpoints
    checkpoint_radius_sq = OPTIMIZER_CHECKPOINT_RADIUS ** 2
    for move_ind in range(0, len(moves), 2):
        direction_delta = np.clip(moves[move_ind], -MAX_TURN_DEG, MAX_TURN_DEG)
        thrust = np.clip(moves[move_ind + 1], 0, 100)
        direction = normalize_angle(direction + direction_delta)
        direction_rad = math.radians(direction)
        vx += math.cos(direction_rad) * thrust
        vy -= math.sin(direction_rad) * thrust
        x += vx
        y += vy
        passed_this_turn = 0
        while passed_this_turn < len(checkpoints):
            checkpoint = checkpoints[(next_checkpoint_ind + passed_this_turn) % len(checkpoints)]
            dx = checkpoint[0] - x
            dy = checkpoint[1] - y
            if dx * dx + dy * dy > checkpoint_radius_sq:
                break
            passed_this_turn += 1
        passed_checkpoints += passed_this_turn
        next_checkpoint_ind = (next_checkpoint_ind + passed_this_turn) % len(checkpoints)
        vx *= DRAG
        vy *= DRAG
    dx = checkpoints[next_checkpoint_ind][0] - x
    dy = checkpoints[next_checkpoint_ind][1] - y
    return math.sqrt(dx * dx + dy * dy) - passed_checkpoints * CHECKPOINT_BONUS


def extend_checkpoint_trajectory(trajectory: list[FutureState], checkpoints: list[NDArray[int]], turn_count: int) -> list[FutureState]:
    """Extends trajectory through turn_count by aiming at checkpoints with full thrust. Returns the extended trajectory."""
    while len(trajectory) <= turn_count:
        pod = trajectory[-1].pod
        direction = pod.get_next_checkpoint_direction(checkpoints)
        next_state = predict_next(pod, checkpoints, normalize_angle(direction - pod.direction), 100)
        next_state.moves = trajectory[-1].moves + next_state.moves
        trajectory.append(next_state)
    return trajectory


def predict_next(current: BasePod, checkpoints: list[NDArray[int]] | None, direction_delta: float, thrust: float | str,
                 checkpoint_radius: int = CHECKPOINT_RADIUS) -> FutureState:
    """Predicts one turn without collisions using the Codingame movement order.
    current supplies the start state, checkpoints may be None, direction_delta and thrust form the command, and checkpoint_radius controls passage.
    Returns the predicted one-turn future state.
    """
    direction_delta, thrust = constrain_moves([direction_delta, thrust])
    command_thrust = thrust
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
    return FutureState([direction_delta, command_thrust], pod)


def constrain_moves(moves: list[float | str] | NDArray[float]) -> list[float | str] | NDArray[float]:
    """Clips a move vector to model limits.
    moves has direction deltas clipped to +/-18 degrees and numeric thrust clipped to 0..100. Returns moves after in-place clipping.
    """
    moves[0] = np.clip(moves[0], -MAX_TURN_DEG, MAX_TURN_DEG)
    for move_ind in range(1, len(moves), 2):
        if not isinstance(moves[move_ind], str):
            moves[move_ind] = np.clip(moves[move_ind], 0, 100)
    for move_ind in range(2, len(moves), 2):
        moves[move_ind] = np.clip(moves[move_ind], -MAX_TURN_DEG, MAX_TURN_DEG)
    return moves


def normalize_angle(angle: float | NDArray[float]) -> float | NDArray[float]:
    """Returns angle in degrees mapped into the half-open range [-180, 180), preserving numpy arrays elementwise."""
    return (angle + 180) % 360 - 180


def get_segment_direction(start: NDArray[float], end: NDArray[float]) -> float:
    """Returns the bot direction angle of the vector from start to end."""
    segment = end - start
    return -math.degrees(math.atan2(segment[1], segment[0]))


def get_command_target(position: NDArray[float], direction: float) -> NDArray[float]:
    """Converts position and direction into the far target point required by Codingame output. Returns target coordinates."""
    direction_rad = math.radians(direction)
    return position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * TARGET_DISTANCE


def reset_timer():
    """Starts the global per-turn debug timer."""
    global TIMER_START, TIMER_LAST
    TIMER_START = time.perf_counter()
    TIMER_LAST = TIMER_START


def log_time(msg: str):
    """Prints msg with elapsed milliseconds since the previous timing mark and since the turn timer started."""
    global TIMER_LAST
    if TIMING:
        now = time.perf_counter()
        log(f"TIME {msg}: step={(now - TIMER_LAST) * 1000:.3g}ms total={(now - TIMER_START) * 1000:.3g}ms")
        TIMER_LAST = now


def log(msg: str):
    """Writes msg to stderr so stdout stays reserved for game commands."""
    if DEBUG:
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
