"""Solves CodinGame Vox Codei episode 2 by tracking moving surveillance nodes and searching bomb placements destroying all of them in time.
Nodes whose direction is still ambiguous after the observed frames split the game into several sampled worlds fixing one state per node.
A plan destroying every node in all of its possible states is preferred, and otherwise an action from which most worlds can still be won."""

from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from itertools import count
from operator import and_, or_
from random import Random
from time import perf_counter

RANGE = 3
FUSE = 3
DIRECTIONS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
WORLDS = 8
LIMIT = 0.05
SLICE = 0.01
RNG = Random(0)


@dataclass(slots=True)
class Firewall:
    """Stores the static geometry of the grid, with cells indexed row by row.
    reach holds for each cell the bit mask of cells covered by a blast there, which is also the mask of cells from which a blast covers it,
    and moves holds for each cell and direction index the cell and direction index a node has one turn later."""
    reach: list[int]
    moves: list[list[tuple[int, int]]]

    def step(self, states: set[tuple[int, int]], blasts: int) -> tuple[set[tuple[int, int]], bool]:
        """Moves node states one turn and then removes those hit by blasts.
        states is the set of (cell, direction index) states and blasts is the mask of cells hit at the end of the turn.
        Returns the surviving states and whether any state was hit."""
        moved = set()
        hit = False
        for cell, direction in states:
            new = self.moves[cell][direction]
            if blasts >> new[0] & 1:
                hit = True
            else:
                moved.add(new)
        return moved, hit


@dataclass(slots=True)
class Tracker:
    """Tracks surveillance nodes across frames.
    tracks holds for each node the set of (cell, direction index) states it may currently have, empty once it is destroyed,
    and doomed marks nodes that may have been destroyed, so that their disappearance is not a contradiction."""
    tracks: list[set[tuple[int, int]]]
    doomed: list[bool]

    def advance(self, firewall: Firewall, blasts: int, nodes: set[int]) -> bool:
        """Moves the tracked nodes to the next frame and keeps only the states matching it.
        firewall is the grid geometry, blasts is the mask of cells hit by bombs exploding during the turn
        and nodes is the set of cells showing surveillance nodes on the new frame.
        Returns whether the new frame is consistent with the tracked nodes."""
        for i, states in enumerate(self.tracks):
            moved, hit = firewall.step(states, blasts)
            self.doomed[i] |= hit
            self.tracks[i] = {state for state in moved if state[0] in nodes}
            if moved and not self.tracks[i] and not self.doomed[i]:
                return False
        changed = True
        while changed:
            changed = False
            for cell in nodes:
                owners = [states for states in self.tracks if any(state[0] == cell for state in states)]
                if not owners:
                    return False
                if len(owners) == 1 and any(state[0] != cell for state in owners[0]):
                    owners[0] -= {state for state in owners[0] if state[0] != cell}
                    changed = True
        return True

    def sample_world(self, nodes: set[int]) -> list[set[tuple[int, int]]]:
        """Picks a random assignment of at most one state to each node that explains every cell showing a surveillance node.
        nodes is the set of cells currently showing surveillance nodes.
        Returns the assignment as a list holding a set with the single state of each node, empty for destroyed nodes."""
        def assign(i: int, covered: set[int]) -> bool:
            if i == len(self.tracks):
                return True
            for state in RNG.sample(sorted(self.tracks[i]), len(self.tracks[i])) or [None]:
                cells = covered | {state[0]} if state else covered
                if closing[i] <= cells:
                    chosen.append({state} if state else set())
                    if assign(i + 1, cells):
                        return True
                    chosen.pop()
            return False

        last = {cell: i for i, states in enumerate(self.tracks) for cell, _ in states}
        closing = [set() for _ in self.tracks]
        for cell in nodes:
            closing[last[cell]].add(cell)
        chosen = []
        assign(0, set())
        return chosen

    def trajectories(self, firewall: Firewall, blasts: dict[int, int], turn: int, frames: int) -> list[list[set[int]]]:
        """Predicts the cells each tracked node may occupy, accounting for the bombs already placed.
        firewall is the grid geometry, blasts maps turns to the masks of cells hit by bombs exploding right before them,
        turn is the current turn and frames is the number of following frames to predict.
        Returns for each node the sets of its possible cells on the current and following frames, empty once it is surely destroyed."""
        paths = []
        for states in self.tracks:
            path = [{cell for cell, _ in states}]
            for frame in range(turn + 1, turn + frames + 1):
                states = firewall.step(states, blasts.get(frame, 0))[0]
                path.append({cell for cell, _ in states})
            paths.append(path)
        return paths


@dataclass(slots=True)
class Problem:
    """Stores the set cover problem of destroying nodes with bombs placed on distinct turns.
    options holds per relative turn the node masks covered by bombs on usable cells, full is the mask of all nodes to destroy,
    places maps each distinct mask of options to the (relative turn, cell) placements covering exactly those nodes, earliest first,
    and by_item holds for each node the masks of places covering it, largest first."""
    options: list[dict[int, int]]
    full: int
    places: dict[int, list[tuple[int, int]]] = field(init=False)
    by_item: list[list[int]] = field(init=False)

    def __post_init__(self):
        """Groups the options by the nodes they cover."""
        self.places = {}
        for t, cells in enumerate(self.options):
            for cell, mask in cells.items():
                self.places.setdefault(mask, []).append((t, cell))
        self.by_item = [[] for _ in range(self.full.bit_length())]
        for mask in sorted(self.places, key=int.bit_count, reverse=True):
            bits = mask
            while bits:
                self.by_item[(bits & -bits).bit_length() - 1].append(mask)
                bits &= bits - 1

    def solve(self, bombs: int, reach: list[int], deadline: float, fixed: dict[int, int], used: int) -> tuple[dict[int, int], bool]:
        """Finds bomb placements covering all nodes. A depth-first search picks masks of places, branching on the uncovered node covered
        by the fewest masks and pruned by the most nodes the remaining bombs can cover. The picked masks must always remain assignable
        to distinct turns, found by backtracking earliest first, on which no bombs trigger each other. Cover failures are remembered
        per covered mask unless a turn assignment failed below them.
        bombs is the number of bombs left, reach holds blast masks of cells, deadline is the perf_counter time at which the search gives up,
        fixed maps relative turns to cells of imposed placements and used is the mask of relative turns that are taken without a placement.
        Returns placements mapping relative turns to cells, the best partial ones if no complete one was found, and whether they are complete."""
        def dfs(covered: int) -> bool:
            nonlocal best, best_score, tainted
            plan = schedule()
            if plan is None:
                tainted = True
                return False
            if (covered.bit_count(), -len(masks)) > best_score:
                best, best_score = plan, (covered.bit_count(), -len(masks))
            if covered == self.full:
                return True
            left = bombs - len(fixed) - len(masks)
            if failed.get(covered, -1) >= left or perf_counter() > deadline:
                return False
            uncovered = self.full & ~covered
            outer, tainted = tainted, False
            if sum(sorted(((mask & uncovered).bit_count() for mask in self.places), reverse=True)[:left]) >= uncovered.bit_count():
                item = min((i for i in range(self.full.bit_length()) if uncovered >> i & 1), key=lambda i: len(self.by_item[i]))
                for mask in sorted(self.by_item[item], key=lambda mask: -(mask & uncovered).bit_count()):
                    masks.append(mask)
                    if dfs(covered | mask):
                        return True
                    masks.pop()
            if not tainted:
                failed[covered] = max(failed.get(covered, -1), left)
            tainted |= outer
            return False

        def schedule() -> dict[int, int]:
            def assign(i: int) -> bool:
                if i == len(order):
                    return True
                if perf_counter() > deadline:
                    return False
                for t, cell in self.places[order[i]]:
                    if not used >> t & 1 and t not in plan and not any(abs(t - other) < FUSE and reach[cell] >> plan[other] & 1 for other in plan):
                        plan[t] = cell
                        if assign(i + 1):
                            return True
                        del plan[t]
                return False

            order = sorted(masks, key=lambda mask: len(self.places[mask]))
            plan = dict(fixed)
            return plan if assign(0) else None

        failed = {}
        masks = []
        best = {}
        best_score = (-1, 0)
        tainted = False
        complete = dfs(reduce(or_, (self.options[t].get(cell, 0) for t, cell in fixed.items()), 0))
        return best, complete


def main():
    """Runs the interactive loop: observes the nodes, updates the tracker and places bombs."""
    width, height = map(int, input().split())
    placed = []
    for turn in count():
        rounds, bombs = map(int, input().split())
        rows = [input() for _ in range(height)]
        start = perf_counter()
        nodes = {y * width + x for y, row in enumerate(rows) for x, char in enumerate(row) if char == "@"}
        if not turn:
            firewall = build_firewall(width, height, rows)
        blasts = {}
        for cell, time in placed:
            blasts[time + FUSE] = blasts.get(time + FUSE, 0) | firewall.reach[cell]
        if not turn or not tracker.advance(firewall, blasts.get(turn, 0), nodes):
            tracker = Tracker([{(cell, direction) for direction in range(len(DIRECTIONS))} for cell in nodes], [False] * len(nodes))
            fresh = turn
        cell = choose_action(firewall, tracker, placed, blasts, turn, rounds, bombs, nodes, start) if turn > fresh else None
        print("WAIT" if cell is None else f"{cell % width} {cell // width}", flush=True)
        if cell is not None:
            placed.append((cell, turn))


def build_firewall(width: int, height: int, rows: list[str]) -> Firewall:
    """Precomputes blast reaches and node moves of the grid.
    width and height are the grid dimensions and rows holds the grid rows.
    Returns the grid geometry."""
    def is_open(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and rows[y][x] != "#"

    reach = []
    moves = []
    for y in range(height):
        for x in range(width):
            mask = 1 << y * width + x
            cell_moves = [(y * width + x, 0)]
            for direction, (dx, dy) in enumerate(DIRECTIONS[1:], 1):
                for step in range(1, RANGE + 1):
                    if not is_open(x + dx * step, y + dy * step):
                        break
                    mask |= 1 << (y + dy * step) * width + x + dx * step
                back = direction + 1 if direction % 2 else direction - 1
                cell_moves.append(((y + dy) * width + x + dx, direction) if is_open(x + dx, y + dy) else
                                  ((y - dy) * width + x - dx, back) if is_open(x - dx, y - dy) else (y * width + x, back))
            reach.append(mask)
            moves.append(cell_moves)
    return Firewall(reach, moves)


def choose_action(firewall: Firewall, tracker: Tracker, placed: list[tuple[int, int]], blasts: dict[int, int], turn: int, rounds: int, bombs: int,
                  nodes: set[int], start: float) -> int:
    """Chooses the bomb placement of the current turn. The first action of a plan destroying every node in all of its possible states
    is taken if one is found. Otherwise the game is split into sampled worlds fixing one state per node, and the first actions of plans
    for single worlds, most common first, the first action of the partial plan for all states and waiting are compared by the number of worlds
    that can still be won after them. Waiting wins ties unless no world or all of them can still be won, as the next frame may tell the worlds apart.
    firewall is the grid geometry, tracker holds the possible node states, placed holds (cell, turn) pairs of all placed bombs,
    blasts maps turns to the masks of cells hit right before them, turn is the current turn, rounds and bombs are the remaining rounds and bombs,
    nodes is the set of cells currently showing surveillance nodes and start is the perf_counter time at which the turn began.
    Returns the cell to place a bomb on, or None to wait."""
    def attempt(problem: Problem, fixed: dict[int, int], used: int) -> tuple[dict[int, int], bool]:
        return problem.solve(bombs, firewall.reach, min(start + LIMIT, perf_counter() + SLICE), fixed, used)

    base = [reduce(or_, (1 << cell for cell in nodes), 0) if not t else 0 for t in range(rounds - FUSE + 1)]
    for cell, time in placed:
        for t in range(min(time + FUSE - turn, len(base))):
            base[t] |= firewall.reach[cell]
    plan, complete = build_problem(firewall, tracker.trajectories(firewall, blasts, turn, rounds), base).solve(bombs, firewall.reach, start + LIMIT / 3, {}, 0)
    if complete:
        return plan.get(0)
    samples = {}
    for _ in range(WORLDS * WORLDS):
        tracks = tracker.sample_world(nodes)
        samples[tuple(next(iter(states), None) for states in tracks)] = tracks
        if len(samples) == WORLDS:
            break
    worlds = [build_problem(firewall, Tracker(tracks, tracker.doomed).trajectories(firewall, blasts, turn, rounds), base) for tracks in samples.values()]
    owns = [attempt(world, {}, 0) for world in worlds]
    firsts = [action for action, _ in Counter(own.get(0) for own, solved in owns if solved).most_common()] + [plan.get(0)] + [own.get(0) for own, _ in owns]
    candidates = [*dict.fromkeys(action for action in firsts if action is not None), None]
    best = candidates[0]
    best_score = (0, False)
    for action in candidates:
        score = sum(solved and own.get(0) == action or attempt(world, {} if action is None else {0: action}, 1)[1]
                    for world, (own, solved) in zip(worlds, owns))
        key = (score, action is None and 0 < score < len(worlds))
        if key > best_score:
            best, best_score = action, key
        if score == len(worlds):
            return action
    return best


def build_problem(firewall: Firewall, paths: list[list[set[int]]], base: list[int]) -> Problem:
    """Builds the problem of destroying the nodes that are not surely destroyed already, each in all of its possible states.
    firewall is the grid geometry, paths holds the possible cells of each node per frame
    and base holds for each relative turn the mask of cells on which a bomb cannot or should not be placed regardless of node moves.
    Returns the problem."""
    blocked = [reduce(or_, (1 << cell for path in paths for cell in path[t]), mask) for t, mask in enumerate(base)]
    options = [{} for _ in blocked]
    full = 0
    for path in paths:
        if not path[-1]:
            continue
        bit = full + 1
        full |= bit
        for t, cells in enumerate(options):
            candidates = reduce(and_, (firewall.reach[cell] for cell in path[t + FUSE])) & ~blocked[t]
            while candidates:
                cell = (candidates & -candidates).bit_length() - 1
                candidates &= candidates - 1
                cells[cell] = cells.get(cell, 0) | bit
    return Problem(options, full)


if __name__ == "__main__":
    main()
