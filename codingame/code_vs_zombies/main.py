import copy
import sys
from dataclasses import dataclass

import numpy as np
import numpy.linalg as linalg
from numpy import ndarray

limits = [16000, 9000]
zombie_speed = 400
ash_speed = 1000
shooting_range = 2000
target_range = 100
max_zombies = 99
max_score = 99999
multipliers = [1, 1]
for i in range(max_zombies - 1):
    multipliers.append(multipliers[i] + multipliers[i + 1])


@dataclass
class GameState:
    humans: ndarray
    zombies: ndarray
    score: int


def read_input() -> GameState:
    humans = [[int(i) for i in input().split()]]
    human_count = int(input())
    for i in range(human_count):
        human_id, human_x, human_y = [int(j) for j in input().split()]
        humans.append([human_x, human_y])
    zombie_count = int(input())
    zombies = []
    for i in range(zombie_count):
        zombie_id, zombie_x, zombie_y, zombie_xnext, zombie_ynext = [int(j) for j in input().split()]
        zombies.append([zombie_x, zombie_y])
    game_state_read = GameState(np.array(humans), np.array(zombies), 0)
    return game_state_read


def debug_input() -> GameState:
    # Simple
    # humans = [[0, 0], [8250, 4500]]
    # zombies = [[8250, 8999]]

    # 2 zombies
    # humans = [[5000, 0], [950, 6000], [8000, 6100]]
    # zombies = [[3100, 7000], [11500, 7100]]

    # 3 vs 3
    # humans = [[7500, 2000], [9000, 1200], [400, 6000]]
    # zombies = [[2000, 1500], [13900, 6500], [7000, 7500]]

    # Rows to defend
    # humans = [[0, 4000], [0, 1000], [0, 8000]]
    # zombies = [[5000, 1000], [5000, 8000], [7000, 1000], [7000, 8000], [9000, 1000], [9000, 8000], [11000, 1000], [11000, 8000], [13000, 1000], [13000, 8000], [14000, 1000],
    #            [14000, 8000], [14500, 1000], [14500, 8000], [15000, 1000], [15000, 8000]]

    # Columns of death
    humans = [[8000, 4000], [0, 4000], [15000, 4000]]
    zombies = [[4333, 1800], [4333, 3600], [4333, 5400], [4333, 7200], [10666, 1800], [10666, 3600], [10666, 5400], [10666, 7200], [0, 7200]]

    game_state_read = GameState(np.array(humans), np.array(zombies), 0)
    return game_state_read


def validate_state(state1: GameState, state2: GameState):
    if not np.all(state1.humans == state2.humans):
        print('Wrong humans state', file=sys.stderr)
        print(state1.humans, file=sys.stderr)
        print(state2.humans, file=sys.stderr)
        raise Exception()
    assert np.all(state1.zombies == state2.zombies), 'Wrong zombie state'


def get_zombie_target(zombie: ndarray, humans: ndarray) -> ndarray:
    distances = np.array([linalg.norm(zombie - human) for human in humans])
    min_dist_ind = np.argmin(distances)
    return humans[min_dist_ind, :]


def update_coords(coords: ndarray, target_coords: ndarray, speed: int) -> ndarray:
    shift = target_coords - coords
    distance = linalg.norm(shift)
    if distance < speed:
        return target_coords
    angle = np.arctan2(shift[1], shift[0])
    new_coords = (coords + np.array([np.cos(angle), np.sin(angle)]) * speed).astype(int)
    return new_coords


def get_score(zombies_killed: int, humans_alive: int) -> int:
    scores = [multipliers[i] * humans_alive ** 2 * 10 for i in range(zombies_killed)]
    return sum(scores)


def simulate_action(state: GameState, action: ndarray) -> GameState:
    state = copy.deepcopy(state)
    zombie_targets = [get_zombie_target(zombie, state.humans) for zombie in state.zombies]
    state.zombies = np.array([update_coords(zombie, target, zombie_speed) for zombie, target in zip(state.zombies, zombie_targets)])
    state.humans[0, :] = update_coords(state.humans[0, :], action, ash_speed)
    zombie_distances = np.array([linalg.norm(state.humans[0, :] - zombie) for zombie in state.zombies])
    alive_zombies = state.zombies[zombie_distances > shooting_range, :]
    zombies_killed = state.zombies.shape[0] - alive_zombies.shape[0]
    state.zombies = alive_zombies
    alive_humans = [not np.any(np.all(human == state.zombies, 1)) for human in state.humans]
    state.humans = state.humans[alive_humans, :]
    if state.zombies.shape[0] == 0:
        state.score = max_score + 1
    elif state.humans.shape[0] == 1:
        state.score = -1
    else:
        state.score += get_score(zombies_killed, state.humans.shape[0] - 1)
    return state


def simulate_protect_target(state: GameState, target: ndarray) -> (ndarray, list[GameState]):
    states = [state]
    while True:
        if not np.any(np.all(states[-1].humans == target, 1)):
            states[-1].score = -1
        if states[-1].score < 0 or states[-1].score > max_score:
            break
        zombie_targets = np.array([get_zombie_target(zombie, states[-1].humans[1:, :]) for zombie in states[-1].zombies])
        threat_exists = np.any(np.all(zombie_targets == target, 1))
        if not threat_exists:
            break
        states.append(simulate_action(states[-1], target))
    return target, states


def find_best_action(state: GameState) -> (ndarray, GameState):
    simulation_results = [simulate_protect_target(state, target) for target in state.humans[1:, :]]
    simulation_results = [result for result in simulation_results if len(result[1]) > 1 and result[1][-1].score >= 0]
    if len(simulation_results) == 0:
        return state.humans[0, :], simulate_action(state, state.humans[0, :])
    best_result = max(simulation_results, key=lambda x: (x[1][-1].humans.shape[0], -len(x[1])))
    return best_result[0], best_result[1][1]


def main_loop():
    game_state = None
    while True:
        game_state_read = read_input()
        # game_state_read = debug_input()

        if game_state is None:
            game_state = game_state_read
        else:
            validate_state(game_state, game_state_read)
        best_action, game_state = find_best_action(game_state)
        print(f'{best_action[0]} {best_action[1]}')


main_loop()
