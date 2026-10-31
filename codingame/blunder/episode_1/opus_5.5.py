"""Simulates Blunder's walk through the city map and prints his moves, or LOOP if he never reaches the booth."""
L, C = map(int, input().split())
grid = [list(input().ljust(C)) for _ in range(L)]
DIRS = {"SOUTH": (1, 0), "EAST": (0, 1), "NORTH": (-1, 0), "WEST": (0, -1)}
PRIORITY = list(DIRS)
MODIFIERS = {"S": "SOUTH", "E": "EAST", "N": "NORTH", "W": "WEST"}
r, c = next((i, row.index("@")) for i, row in enumerate(grid) if "@" in row)
teleports = [(i, j) for i in range(L) for j in range(C) if grid[i][j] == "T"]
direction, inverted, breaker, broken, moves, seen = "SOUTH", False, False, 0, [], set()
while grid[r][c] != "$":
    # Walls only ever get destroyed, so the broken count pins down the full map state along this trajectory
    if (state := (r, c, direction, inverted, breaker, broken)) in seen:
        moves = ["LOOP"]
        break
    seen.add(state)
    blocked = lambda d: grid[r + DIRS[d][0]][c + DIRS[d][1]] == "#" or grid[r + DIRS[d][0]][c + DIRS[d][1]] == "X" and not breaker
    if blocked(direction):
        direction = next(d for d in (PRIORITY[::-1] if inverted else PRIORITY) if not blocked(d))
    r, c = r + DIRS[direction][0], c + DIRS[direction][1]
    moves.append(direction)
    match grid[r][c]:
        case "X":
            grid[r][c] = " "
            broken += 1
        case m if m in MODIFIERS:
            direction = MODIFIERS[m]
        case "I":
            inverted = not inverted
        case "B":
            breaker = not breaker
        case "T":
            r, c = teleports[1 - teleports.index((r, c))]
print("\n".join(moves))
