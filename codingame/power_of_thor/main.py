from __future__ import annotations

import sys
from collections import deque

import numpy as np
from numpy import ndarray


moves = {'WAIT': [0, 0], 'N': [0, -1], 'NE': [1, -1], 'E': [1, 0], 'SE': [1, 1], 'S': [0, 1], 'SW': [-1, 1], 'W': [-1, 0], 'NW': [-1, -1]}
map_size = [40, 18]
inverse_moves = {tuple(v): k for k, v in moves.items()}


class Position:
    def __init__(self, coords: ndarray, previous: Position | None, giants: ndarray):
        self.coords = coords
        self.previous = previous
        self.distance = 0 if previous is None else previous.distance + 1
        giant_distances = np.array([get_distance(coords, giant) for giant in giants])
        self.feasible = min(giant_distances) > 1
        self.score = max(giant_distances) - min(giant_distances)

    def __str__(self):
        return f'coords={self.coords}, score={self.score}'


def get_distance(thor_coords: ndarray, giant_coords: ndarray) -> int:
    """ Returns number of turns until given giant reaches Thor. """
    coord_distances = np.abs(thor_coords - giant_coords)
    return max(coord_distances)


def read_giant_coords():
    num_strikes, num_giants = [int(i) for i in input().split()]
    giant_coords = []
    for i in range(num_giants):
        giant_coords.append([int(j) for j in input().split()])
    giant_coords = np.array(giant_coords)
    return giant_coords


def explore_pathways(thor: ndarray, giants: ndarray) -> ndarray:
    map = np.full(map_size, None, dtype=object)
    map[*thor] = Position(thor, None, giants)
    exploration_queue = deque()
    exploration_queue.append(map[*thor])
    while len(exploration_queue) > 0:
        position = exploration_queue.popleft()
        for step in moves.values():
            if step == [0, 0]:
                continue
            next_coords = position.coords + step
            if any(next_coords < 0) or any(next_coords >= map.shape):
                continue
            if map[*next_coords] is not None:
                if map[*next_coords].previous is not None and map[*next_coords].distance == position.distance + 1 and map[*next_coords].previous.score > position.score:
                    map[*next_coords].previous = position
            else:
                map[*next_coords] = Position(next_coords, position, giants)
                if map[*next_coords].feasible:
                    exploration_queue.append(map[*next_coords])
    return map


def backtrack(position: Position) -> Position:
    if position.previous is None:
        return position
    while position.previous.previous is not None:
        position = position.previous
    return position


def print_map(thor: ndarray, giants: ndarray, map: ndarray):
    for i in range(map.shape[0]):
        print(f'{i:<4} ', end='', file=sys.stderr, flush=True)
        for j in range(map.shape[1]):
            if all([i, j] == thor):
                sym = 'T'
            elif any(np.all([i, j] == giants, axis=1)):
                sym = 'G'
            elif map[i, j] is None:
                sym = 'N'
            elif not map[i, j].feasible:
                sym = 'X'
            else:
                sym = str(map[i, j].score)
            print(f'{sym:3}', end='', file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)


thor = np.array([int(i) for i in input().split()])
while True:
    giants = read_giant_coords()
    map = explore_pathways(thor, giants)

    print_map(thor, giants, map)

    feasible_positions = [position for position in map.flatten() if position is not None and position.feasible]
    best_feasible_position = min(feasible_positions, key=lambda position: (position.score, get_distance(thor, position.coords)), default=None)

    if not map[*thor].feasible and (best_feasible_position is None or best_feasible_position.score > map[*thor].score):
        print('STRIKE')
    else:

        print(best_feasible_position.coords, file=sys.stderr, flush=True)

        next_position = backtrack(best_feasible_position)
        direction = np.sign(next_position.coords - thor)
        print(inverse_moves[tuple(direction)])
        thor += direction
