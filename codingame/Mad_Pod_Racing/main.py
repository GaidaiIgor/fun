"""Runs a simple Mad Pod Racing bot."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Player:
    """Stores one pod state.
    :var position: Pod center coordinates.
    """
    position: NDArray[int]


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
        game_state = read_game_state(game_state)
        thrust = 0 if abs(game_state.next_check_angle) > 90 else 100
        if game_state.next_check_angle == 0 and game_state.next_check_dist > 5000 and game_state.boosts:
            thrust = "BOOST"
            game_state.boosts -= 1
        print(*game_state.next_check_pos, thrust)


def read_game_state(previous_game_state: GameState | None) -> GameState:
    """Reads the current turn input.
    :param previous_game_state: Previous turn state carrying persistent data.
    :return: Parsed game state for the current turn.
    """
    x, y, next_check_x, next_check_y, next_check_dist, next_check_angle = map(int, input().split())
    opponent_x, opponent_y = map(int, input().split())
    boosts = 1 if previous_game_state is None else previous_game_state.boosts
    return GameState(player=Player(np.array((x, y))), opponent=Player(np.array((opponent_x, opponent_y))),
                     next_check_pos=np.array((next_check_x, next_check_y)), next_check_dist=next_check_dist, next_check_angle=next_check_angle, boosts=boosts)


main()
