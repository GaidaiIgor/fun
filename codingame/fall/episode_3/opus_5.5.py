"""Guides Indy to the exit in The Fall - Episode 3 by rotating the rooms on his way and stopping the rocks that would hit him,
either against walls or against each other."""
from collections import Counter, deque
from collections.abc import Iterator
from itertools import count, islice
from math import inf
from time import perf_counter

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
ROUTES = 20  # routes of Indy tried when no rock is stopped
STOPPED_ROUTES = 3  # routes of Indy tried per set of rotations stopping rocks
TIME_LIMIT = 0.1  # seconds of search per turn

Grid = dict[tuple[int, int], int]
Mover = tuple[int, int, str]
Task = tuple[int, int, int, str, bool]  # last turn to do it, room x, room y, rotation command, whether Indy's route needs it

def main():
    """Reads the tunnel, then each turn reads Indy and the rocks and prints the rotation to do."""
    _, height = map(int, input().split())
    types = {(x, y): int(kind) for y in range(height) for x, kind in enumerate(input().split())}
    grid = {room: abs(kind) for room, kind in types.items()}
    exit_room = (int(input()), height - 1)
    locked = {room for room, kind in types.items() if kind < 0} | {exit_room}
    while True:
        x, y, entrance = input().split()
        end = perf_counter() + TIME_LIMIT
        indy = (int(x), int(y), entrance)
        rocks = [(int(x), int(y), entrance) for x, y, entrance in (input().split() for _ in range(int(input())))]
        if task := choose(grid, locked, exit_room, indy, rocks, end):
            _, x, y, command, _ = task
            grid[x, y] = ROTATIONS[command][grid[x, y]]
            print(x, y, command)
        else:
            print("WAIT")

def choose(grid: Grid, locked: set[tuple[int, int]], exit_room: tuple[int, int], indy: Mover, rocks: list[Mover], end: float) -> Task:
    """Returns the rotation to do this turn (None to wait) so that Indy at indy reaches exit_room in the tunnel grid without meeting
    the rocks. Rooms in locked cannot be rotated. Searches breadth first over sets of rotations stopping rocks, each tried with
    several routes of Indy, until a safe plan is found or the moment end passes. Falls back on the plan whose first hit is the latest."""
    queue, seen, best_hit, best = deque([()]), {frozenset()}, -inf, None
    while queue and perf_counter() < end:
        stops = queue.popleft()
        tasks = [(turn, x, y, command, False) for turn, x, y, commands in stops for command in commands]
        stopped = {(x, y) for _, x, y, _ in stops}
        for route in islice(routes(grid, locked, exit_room, indy, stopped, Counter(task[0] for task in tasks)), STOPPED_ROUTES if stops else ROUTES):
            hit, task, entered = simulate(grid, exit_room, indy, rocks, route + tasks)
            if hit == inf:
                return task
            if hit > best_hit:
                best_hit, best = hit, task
            # the latest stops are tried first, so that rocks arriving later can still destroy the others instead
            for (x, y), turn in sorted(entered.items(), key=lambda item: -item[1]):
                if (x, y) in locked or (x, y) in stopped or (x, y) == indy[:2]:
                    continue
                for _, commands in variants(grid[x, y])[1:]:
                    if (key := frozenset(stop[1:] for stop in stops) | {(x, y, commands)}) not in seen:
                        seen.add(key)
                        queue.append(stops + ((turn - 1, x, y, commands),))
    return best

def routes(grid: Grid, locked: set[tuple[int, int]], exit_room: tuple[int, int], indy: Mover, avoided: set[tuple[int, int]],
           stops: Counter[int]) -> Iterator[list[Task]]:
    """Generates the safe routes of Indy at indy to exit_room in the tunnel grid that do not pass the rooms in avoided, where the
    rooms in locked cannot be rotated and stops counts the other rotations due by each turn. Each route is given by its rotations."""
    def search(x: int, y: int, entrance: str, kind: int, turn: int, spare: int) -> Iterator[list[Task]]:
        """Generates the rotations of the rest of the route from room (x, y) of type kind entered through entrance on turn, given
        the number spare of turns left after the rotations due before."""
        if (x, y) == exit_room:
            yield []
            return
        if failed.get((x, y, entrance, kind, turn), -inf) >= spare:
            return
        dx, dy = MOVES[kind][entrance]
        found = False
        if (x + dx, y + dy) in grid and (x + dx, y + dy) not in avoided:
            for option, commands in variants(grid[x + dx, y + dy])[:1 if (x + dx, y + dy) in locked else None]:
                if ENTRANCES[dx, dy] in MOVES.get(option, {}) and (left := spare + 1 - len(commands) - stops[turn]) >= 0:
                    for rest in search(x + dx, y + dy, ENTRANCES[dx, dy], option, turn + 1, left):
                        found = True
                        yield [(turn, x + dx, y + dy, command, True) for command in commands] + rest
        if not found:
            failed[x, y, entrance, kind, turn] = spare

    failed = {}
    return search(*indy, grid[indy[:2]], 0, 0)

def variants(kind: int) -> list[tuple[int, tuple[str, ...]]]:
    """Returns the types a room of type kind can be turned into with the fewest rotation commands doing it, cheapest first."""
    options = {}
    for option, commands in (kind, ()), (CLOCKWISE[kind], ("RIGHT",)), (ROTATIONS["LEFT"][kind], ("LEFT",)), \
            (CLOCKWISE[CLOCKWISE[kind]], ("RIGHT", "RIGHT")):
        options.setdefault(option, commands)
    return list(options.items())

def simulate(grid: Grid, exit_room: tuple[int, int], indy: Mover, rocks: list[Mover], tasks: list[Task]) -> tuple[float, Task, dict[tuple[int, int], int]]:
    """Plays the rotations tasks in the tunnel grid until Indy at indy reaches exit_room, crashes or meets one of the rocks.
    Returns the turn Indy dies on (inf if he escapes), the task done on the first turn (None for waiting) and the turn each room
    is first entered by a rock."""
    grid, pending, entered, first = dict(grid), sorted(tasks), {}, None
    for turn in count():
        task = pick(pending, {rock[:2] for rock in rocks} | {indy[:2]}, turn)
        if task:
            pending.remove(task)
            grid[task[1], task[2]] = ROTATIONS[task[3]][grid[task[1], task[2]]]
        first = task if turn == 0 else first
        indy = advance(grid, indy)
        rocks = [rock for rock in (advance(grid, rock) for rock in rocks) if rock]
        for rock in rocks:
            entered.setdefault(rock[:2], turn + 1)
        if indy is None or indy[2] not in MOVES.get(grid[indy[:2]], {}) or any(rock[:2] == indy[:2] for rock in rocks):
            return turn + 1, first, entered
        if indy[:2] == exit_room:
            return inf, first, entered
        crowded = Counter(rock[:2] for rock in rocks)
        rocks = [rock for rock in rocks if crowded[rock[:2]] == 1]

def pick(pending: list[Task], busy: set[tuple[int, int]], turn: int) -> Task:
    """Returns which of the pending tasks, sorted by deadline, to do on turn (None to wait), given the busy rooms that cannot be
    rotated now. Indy's next rotation is done as soon as possible, other rotations as late as the deadlines allow."""
    free = [task for task in pending if task[1:3] not in busy]
    for task in [task for task in free if task[4]][:1] + [None] + free:
        if all(turn + 1 + index <= other[0] for index, other in enumerate(other for other in pending if other is not task)):
            return task
    return free[0] if free else None

def advance(grid: Grid, mover: Mover) -> Mover:
    """Returns where mover gets to in the tunnel grid on the next turn, or None if it leaves the grid or crashes."""
    x, y, entrance = mover
    if (move := MOVES.get(grid[x, y], {}).get(entrance)) is None or (x + move[0], y + move[1]) not in grid:
        return None
    if ENTRANCES[move] in MOVES.get(kind := grid[x + move[0], y + move[1]], {}) or STUCK.get(kind) == ENTRANCES[move]:
        return x + move[0], y + move[1], ENTRANCES[move]
    return None

main()
