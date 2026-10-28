"""Solves CodinGame Vox Codei episode 1 by searching a set of bomb cells covering every node and scheduling it within the rounds."""

from itertools import count
import sys

DELAY = 2
RANGE = 3


def main():
    """Reads the firewall, plans every bomb on the first round and replays the plan afterwards."""
    width, height = map(int, input().split())
    grid = [input() for _ in range(height)]
    rounds, bombs = map(int, input().split())
    node_at = {(x, y): i for i, (x, y) in enumerate((x, y) for y in range(height) for x in range(width) if grid[y][x] == "@")}
    covers = {}
    for y in range(height):
        for x in range(width):
            if grid[y][x] == "#":
                continue
            mask = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for step in range(1, RANGE + 1):
                    nx, ny = x + dx * step, y + dy * step
                    if not (0 <= nx < width and 0 <= ny < height) or grid[ny][nx] == "#":
                        break
                    mask |= 1 << node_at[nx, ny] if (nx, ny) in node_at else 0
            if mask:
                covers[x, y] = mask
    plan = search(covers, node_at, bombs, rounds)
    for turn in count():
        cell = plan.get(turn)
        print(f"{cell[0]} {cell[1]}" if cell else "WAIT", flush=True)
        if not sys.stdin.readline():
            break


def search(covers: dict[tuple[int, int], int], node_at: dict[tuple[int, int], int], bombs: int, rounds: int) -> dict[int, tuple[int, int]]:
    """Finds bomb cells covering every node with depth-first search, branching on the uncovered node with the fewest covering cells.
    covers maps each usable cell to the bit mask of nodes its blast reaches, node_at maps node cells to their bit indices,
    bombs is the number of available bombs and rounds is the number of remaining rounds.
    Returns the schedule mapping round indices to bomb cells."""
    def dfs(chosen: frozenset[tuple[int, int]], covered: int) -> dict[int, tuple[int, int]]:
        if covered == full:
            return schedule(chosen, covers, node_at, rounds)
        if len(chosen) == bombs or chosen in seen:
            return None
        seen.add(chosen)
        uncovered = full & ~covered
        if (bombs - len(chosen)) * max((mask & uncovered).bit_count() for mask in covers.values()) < uncovered.bit_count():
            return None
        node = min((i for i in range(len(node_at)) if uncovered >> i & 1), key=lambda i: len(cells_by_node[i]))
        for cell in sorted(cells_by_node[node], key=lambda cell: -(covers[cell] & uncovered).bit_count()):
            if plan := dfs(chosen | {cell}, covered | covers[cell]):
                return plan
        return None

    full = (1 << len(node_at)) - 1
    cells_by_node = [[cell for cell, mask in covers.items() if mask >> i & 1] for i in range(len(node_at))]
    seen = set()
    return dfs(frozenset(), 0)


def schedule(chosen: frozenset[tuple[int, int]], covers: dict[tuple[int, int], int], node_at: dict[tuple[int, int], int], rounds: int) \
        -> dict[int, tuple[int, int]]:
    """Places the chosen bomb cells as early as possible, waiting for nodes occupying chosen cells to be removed first,
    and preferring bombs whose blast frees more chosen cells. covers and node_at are as in search,
    rounds is the number of remaining rounds, so the last bomb has to explode on the last round at the latest.
    Returns the schedule mapping round indices to bomb cells, or None if chosen does not fit into rounds."""
    pending = set(chosen)
    removal = {}
    plan = {}
    for turn in range(rounds - DELAY):
        ready = [cell for cell in pending if cell not in node_at or removal.get(node_at[cell], rounds) <= turn]
        if not ready:
            continue
        cell = max(ready, key=lambda cell: sum(other in node_at and covers[cell] >> node_at[other] & 1 for other in pending))
        plan[turn] = cell
        pending.remove(cell)
        for i in range(len(node_at)):
            if covers[cell] >> i & 1:
                removal.setdefault(i, turn + DELAY + 1)
        if not pending:
            return plan
    return None


if __name__ == "__main__":
    main()
