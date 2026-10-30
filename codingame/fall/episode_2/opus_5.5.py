"""Guides Indy to the exit in The Fall - Episode 2 by rotating the rooms on his way and diverting the rocks that would hit him."""
from collections.abc import Iterator
from itertools import count, islice
from math import inf

# room type -> entrance point -> (dx, dy) of the next room
MOVES = {1: {"TOP": (0, 1), "LEFT": (0, 1), "RIGHT": (0, 1)}, 2: {"LEFT": (1, 0), "RIGHT": (-1, 0)}, 3: {"TOP": (0, 1)},
         4: {"TOP": (-1, 0), "RIGHT": (0, 1)}, 5: {"TOP": (1, 0), "LEFT": (0, 1)}, 6: {"LEFT": (1, 0), "RIGHT": (-1, 0)},
         7: {"TOP": (0, 1), "RIGHT": (0, 1)}, 8: {"LEFT": (0, 1), "RIGHT": (0, 1)}, 9: {"TOP": (0, 1), "LEFT": (0, 1)},
         10: {"TOP": (-1, 0)}, 11: {"TOP": (1, 0)}, 12: {"RIGHT": (0, 1)}, 13: {"LEFT": (0, 1)}}
CLOCKWISE = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 8, 8: 9, 9: 6, 10: 11, 11: 12, 12: 13, 13: 10}
ROTATIONS = {"RIGHT": CLOCKWISE, "LEFT": {after: before for before, after in CLOCKWISE.items()}}
ENTRANCES = {(0, 1): "TOP", (1, 0): "LEFT", (-1, 0): "RIGHT"}
# room type -> entrance point that is open but leads nowhere, so that a rock gets stuck in the room instead of hitting a wall
STUCK = {4: "LEFT", 5: "RIGHT", 6: "TOP", 10: "LEFT", 11: "RIGHT"}
ROUTES = 20  # routes of Indy tried per grid when avoiding rocks

Grid = dict[tuple[int, int], int]
Mover = tuple[int, int, str]
Action = tuple[int, int, str]

def main():
    """Reads the tunnel, then each turn reads Indy and the rocks and prints the rotation to do."""
    _, height = map(int, input().split())
    types = {(x, y): int(kind) for y in range(height) for x, kind in enumerate(input().split())}
    grid = {room: abs(kind) for room, kind in types.items()}
    exit_room = (int(input()), height - 1)
    locked = {room for room, kind in types.items() if kind < 0} | {exit_room}
    while True:
        x, y, entrance = input().split()
        indy = (int(x), int(y), entrance)
        rocks = [(int(x), int(y), entrance) for x, y, entrance in (input().split() for _ in range(int(input())))]
        actions = choose(grid, locked, exit_room, indy, rocks)
        if actions:
            x, y, command = actions[0]
            grid[x, y] = ROTATIONS[command][grid[x, y]]
            print(x, y, command)
        else:
            print("WAIT")

def choose(grid: Grid, locked: set[tuple[int, int]], exit_room: tuple[int, int], indy: Mover, rocks: list[Mover]) -> list[Action]:
    """Returns the rotations to do on this and the following turns, one per turn, to lead Indy at indy to exit_room in the tunnel grid
    while keeping away from the rocks. Rooms in locked cannot be rotated. If a rock would hit Indy, rotations of the rooms on the rock
    routes are tried first, keeping the one that delays the first hit the most."""
    best, best_hit, visited = [], -inf, set()
    for actions in islice(routes(grid, locked, exit_room, indy, 0), ROUTES):
        hit, seen = collision(grid, exit_room, indy, rocks, actions)
        if hit == inf:
            return actions
        if hit > best_hit:
            best, best_hit = actions, hit
        visited |= seen
    for x, y in visited - {(x, y) for x, y, _ in rocks} - {indy[:2]} - locked:
        for command, rotation in ROTATIONS.items():
            for rest in islice(routes(grid | {(x, y): rotation[grid[x, y]]}, locked, exit_room, indy, 1), ROUTES):
                if (hit := collision(grid, exit_room, indy, rocks, [(x, y, command)] + rest)[0]) == inf:
                    return [(x, y, command)] + rest
                if hit > best_hit:
                    best, best_hit = [(x, y, command)] + rest, hit
    return best

def routes(grid: Grid, locked: set[tuple[int, int]], exit_room: tuple[int, int], indy: Mover, reserved: int) -> Iterator[list[Action]]:
    """Generates the safe routes of Indy at indy to exit_room in the tunnel grid, where the rooms in locked cannot be rotated and
    the first reserved turns are spent elsewhere. Each route is given by the rotations to do, one per turn, in the order Indy
    reaches the rooms."""
    def search(x: int, y: int, entrance: str, kind: int, slack: int) -> Iterator[list[Action]]:
        """Generates the rotations of the rest of the route from room (x, y) of type kind entered through entrance, given the
        number slack of spare turns left after the rotations of the rooms before."""
        if (x, y) == exit_room:
            yield []
            return
        if failed.get((x, y, entrance, kind), -inf) >= slack:
            return
        dx, dy = MOVES[kind][entrance]
        found = False
        if (x + dx, y + dy) in grid:
            current = grid[x + dx, y + dy]
            variants = [(current, [])] + ([] if (x + dx, y + dy) in locked else [(CLOCKWISE[current], ["RIGHT"]),
                        (ROTATIONS["LEFT"][current], ["LEFT"]), (CLOCKWISE[CLOCKWISE[current]], ["RIGHT", "RIGHT"])])
            # reversing keeps the cheapest rotation leading to each type
            for option, commands in sorted(dict(reversed(variants)).items(), key=lambda variant: len(variant[1])):
                if ENTRANCES[dx, dy] in MOVES.get(option, {}) and len(commands) <= slack + 1:
                    for rest in search(x + dx, y + dy, ENTRANCES[dx, dy], option, slack + 1 - len(commands)):
                        found = True
                        yield [(x + dx, y + dy, command) for command in commands] + rest
        if not found:
            failed[x, y, entrance, kind] = slack

    failed = {}
    return search(*indy, grid[indy[:2]], -reserved)

def collision(grid: Grid, exit_room: tuple[int, int], indy: Mover, rocks: list[Mover], actions: list[Action]) -> tuple[float, set[tuple[int, int]]]:
    """Simulates the tunnel grid while the rotations actions are done one per turn, until Indy at indy reaches exit_room or meets
    one of the rocks. Returns the turn of the first hit (inf if there is none) and the rooms the rocks enter until then."""
    grid, visited = dict(grid), set()
    for turn in count(1):
        if turn <= len(actions):
            x, y, command = actions[turn - 1]
            grid[x, y] = ROTATIONS[command][grid[x, y]]
        indy = advance(grid, indy)
        rocks = [rock for rock in (advance(grid, rock) for rock in rocks) if rock]
        visited |= {rock[:2] for rock in rocks}
        if indy is None or any(rock[:2] == indy[:2] for rock in rocks):
            return turn, visited
        if indy[:2] == exit_room:
            return inf, visited

def advance(grid: Grid, mover: Mover) -> Mover:
    """Returns where mover gets to in the tunnel grid on the next turn, or None if it disappears. A mover that hits a wall stays
    in place without an entrance for one more turn."""
    x, y, entrance = mover
    if (move := MOVES.get(grid[x, y], {}).get(entrance)) is None or (x + move[0], y + move[1]) not in grid:
        return None
    if ENTRANCES[move] in MOVES.get(kind := grid[x + move[0], y + move[1]], {}) or STUCK.get(kind) == ENTRANCES[move]:
        return x + move[0], y + move[1], ENTRANCES[move]
    return x, y, None

main()
