"""Runs a simple Mad Pod Racing bot."""

import math
import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


DRAG_FACTOR = 0.15
BOOST_THRUST = 650
MAX_TURN_DEG = 18


@dataclass(slots=True)
class Player:
    """Stores one pod state.
    :var position: Pod center coordinates.
    :var velocity: Pod velocity vector since the previous turn.
    :var direction: Pod angle in degrees from the positive x-axis, positive toward negative y, or None when unknown.
    """
    position: NDArray[int]
    velocity: NDArray[int]
    direction: float | None


@dataclass(slots=True)
class GameState:
    """Stores the current turn state.
    :var player: Our pod state.
    :var opponent: The opponent pod state.
    :var next_check_pos: Coordinates of the next checkpoint.
    :var next_check_dist: Distance to the next checkpoint.
    :var next_check_angle: Signed angle to the next checkpoint.
    :var boosts: Number of unused boosts.
    """
    player: Player
    opponent: Player
    next_check_pos: NDArray[int]
    next_check_dist: int
    next_check_angle: int
    boosts: int


def main():
    """Runs the game loop."""
    game_state = None
    while True:
        prev_game_state = game_state
        game_state = read_game_state(prev_game_state)
        print(f"pos={game_state.player.position}, vel={game_state.player.velocity}, dir={game_state.player.direction}", file=sys.stderr)

        thrust = 0 if abs(game_state.next_check_angle) > 90 else 100
        if game_state.next_check_angle == 0 and game_state.next_check_dist > 5000 and game_state.boosts:
            thrust = "BOOST"
            game_state.boosts -= 1
        print(*game_state.next_check_pos, thrust)


def read_game_state(prev_game_state: GameState | None) -> GameState:
    """Reads the current turn input.
    :param prev_game_state: Previous turn state carrying persistent data.
    :return: Parsed game state for the current turn.
    """
    x, y, next_check_x, next_check_y, next_check_dist, next_check_angle = map(int, input().split())
    opponent_x, opponent_y = map(int, input().split())
    player_pos = np.array((x, y))
    opponent_pos = np.array((opponent_x, opponent_y))
    player_velocity = np.zeros(2, dtype=int) if prev_game_state is None else player_pos - prev_game_state.player.position
    opponent_velocity = np.zeros(2, dtype=int) if prev_game_state is None else opponent_pos - prev_game_state.opponent.position
    target_direction = -math.degrees(math.atan2(next_check_y - y, next_check_x - x))
    player_direction = (target_direction + next_check_angle + 180) % 360 - 180
    boosts = 1 if prev_game_state is None else prev_game_state.boosts
    return GameState(Player(player_pos, player_velocity, player_direction), Player(opponent_pos, opponent_velocity, None),
                     np.array((next_check_x, next_check_y)), next_check_dist, next_check_angle, boosts)


main()
