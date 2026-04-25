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
COMMAND_TARGET_DIST = 10000
MAX_TURN_DEG = 18


@dataclass(slots=True)
class Player:
    """Stores one pod state.
    :var position: Pod center coordinates.
    :var velocity: Hidden post-friction velocity carried into this turn, or None when unknown.
    :var direction: Pod angle in degrees from the positive x-axis, positive toward negative y, or None when unknown.
    :var next_checkpoint_ind: Index of the current checkpoint in checkpoints, or None when unknown.
    :var boosts: Number of unused boosts, or None when unknown.
    """
    position: NDArray[int]
    velocity: NDArray[int] | None
    direction: float | None
    next_checkpoint_ind: int | None
    boosts: int | None


@dataclass(slots=True)
class GameState:
    """Stores the current turn state.
    :var player: Our pod state.
    :var opponent: The opponent pod state.
    :var checkpoints: Known checkpoint coordinates starting with the initial pod position.
    """
    player: Player
    opponent: Player
    checkpoints: list[NDArray[int]]


def main():
    """Runs the game loop."""
    game_state = None
    while True:
        prev_game_state = game_state
        game_state = update_game_state(prev_game_state)
        print(f"pos={game_state.player.position}, vel={game_state.player.velocity}, dir={game_state.player.direction}", file=sys.stderr)

        target_pos, thrust = choose_move(game_state)
        if thrust == "BOOST":
            game_state.player.boosts -= 1
        print(*target_pos, thrust)


def update_game_state(prev_game_state: GameState | None) -> GameState:
    """Updates game state.
    :param prev_game_state: Previous turn state carrying persistent data.
    :return: Parsed game state for the current turn.
    """
    x, y, next_checkpoint_x, next_checkpoint_y, _, next_checkpoint_angle = map(int, input().split())
    opponent_x, opponent_y = map(int, input().split())
    player_pos = np.array((x, y))
    next_checkpoint = np.array((next_checkpoint_x, next_checkpoint_y))
    opponent_pos = np.array((opponent_x, opponent_y))
    if prev_game_state is None:
        player_velocity = np.zeros(2, dtype=int)
        checkpoints = [player_pos]
        boosts = 1
    else:
        player_velocity = ((player_pos - prev_game_state.player.position) * DRAG).astype(int)
        checkpoints = prev_game_state.checkpoints
        boosts = prev_game_state.player.boosts

    checkpoint_direction = -math.degrees(math.atan2(next_checkpoint_y - y, next_checkpoint_x - x))
    player_direction = normalize_angle(checkpoint_direction + next_checkpoint_angle)
    checkpoints, next_checkpoint_ind = update_checkpoint_state(checkpoints, next_checkpoint)
    return GameState(Player(player_pos, player_velocity, player_direction, next_checkpoint_ind, boosts), Player(opponent_pos, None, None, None, None),
                     checkpoints)


def update_checkpoint_state(checkpoints: list[NDArray[int]], next_checkpoint: NDArray[int]) -> tuple[list[NDArray[int]], int]:
    """Updates known checkpoints with the current next checkpoint.
    :param checkpoints: Previously known checkpoint coordinates.
    :param next_checkpoint: Current next checkpoint coordinates.
    :return: Updated checkpoints and current checkpoint index.
    """
    next_checkpoint_ind = len(checkpoints)
    for checkpoint_ind, checkpoint in enumerate(checkpoints):
        if np.array_equal(next_checkpoint, checkpoint):
            next_checkpoint_ind = checkpoint_ind
            break
    if next_checkpoint_ind == len(checkpoints):
        checkpoints = [*checkpoints, next_checkpoint]
    return checkpoints, next_checkpoint_ind


def choose_move(game_state: GameState) -> tuple[NDArray[int], int | str]:
    """Chooses a command minimizing predicted distance to the next checkpoint.
    :param game_state: Current game state.
    :return: Target coordinates and thrust command.
    """
    checkpoint_pos = game_state.checkpoints[game_state.player.next_checkpoint_ind]
    checkpoint_delta = checkpoint_pos - game_state.player.position
    checkpoint_direction = -math.degrees(math.atan2(checkpoint_delta[1], checkpoint_delta[0]))
    direction_delta_guess = np.clip(normalize_angle(checkpoint_direction - game_state.player.direction), -MAX_TURN_DEG, MAX_TURN_DEG)
    direction_guess = normalize_angle(game_state.player.direction + direction_delta_guess)
    direction_bounds = (game_state.player.direction - MAX_TURN_DEG, game_state.player.direction + MAX_TURN_DEG)
    result = minimize(lambda move: linalg.norm(predict_next(game_state.player, move[0], move[1]).position - checkpoint_pos),
                      np.array((direction_guess, 100)), bounds=(direction_bounds, (0, 100)), method="Powell")
    print(f"guess=({direction_guess:g}, 100), optimized=({result.x[0]:g}, {result.x[1]:g})", file=sys.stderr)

    direction = normalize_angle(result.x[0])
    thrust = round(result.x[1])
    if game_state.player.direction == checkpoint_direction and linalg.norm(checkpoint_delta) > 5000 and game_state.player.boosts:
        thrust = "BOOST"
    direction_rad = math.radians(direction)
    target_pos = np.rint(game_state.player.position + np.array((math.cos(direction_rad), -math.sin(direction_rad))) * COMMAND_TARGET_DIST).astype(int)
    return target_pos, thrust


def predict_next(current: Player, direction: float, thrust: float) -> Player:
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
    return Player(position, velocity, next_direction, None, current.boosts)


def normalize_angle(angle: float) -> float:
    """Normalizes an angle to [-180, 180) degrees.
    :param angle: Angle in degrees.
    :return: Equivalent angle in the normalized range.
    """
    return (angle + 180) % 360 - 180


main()
