"""Runs a simple Mad Pod Racing bot."""

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
    """
    position: NDArray[int]
    velocity: NDArray[int]


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

        abs_velocity_change = np.linalg.norm(game_state.player.velocity - prev_game_state.player.velocity) if prev_game_state else 0
        print(f"pos={game_state.player.position} vel={game_state.player.velocity} d|v|={abs_velocity_change:g}", file=sys.stderr)

        # if np.linalg.norm(game_state.player.velocity) < 300:
        #     print(game_state.player.position[0], 0, 100)
        # else:
        #     print(game_state.player.position[0], 0, 0)

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
    boosts = 1 if prev_game_state is None else prev_game_state.boosts
    return GameState(Player(player_pos, player_velocity), Player(opponent_pos, opponent_velocity), np.array((next_check_x, next_check_y)),
                     next_check_dist, next_check_angle, boosts)


main()
