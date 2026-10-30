"""Predicts Indy's next room in The Fall - Episode 1 from the room type and his entrance point."""

# room type -> entrance point -> (dx, dy) of the next room
MOVES = {1: {"TOP": (0, 1), "LEFT": (0, 1), "RIGHT": (0, 1)}, 2: {"LEFT": (1, 0), "RIGHT": (-1, 0)}, 3: {"TOP": (0, 1)},
         4: {"TOP": (-1, 0), "RIGHT": (0, 1)}, 5: {"TOP": (1, 0), "LEFT": (0, 1)}, 6: {"LEFT": (1, 0), "RIGHT": (-1, 0)},
         7: {"TOP": (0, 1), "RIGHT": (0, 1)}, 8: {"LEFT": (0, 1), "RIGHT": (0, 1)}, 9: {"TOP": (0, 1), "LEFT": (0, 1)},
         10: {"TOP": (-1, 0)}, 11: {"TOP": (1, 0)}, 12: {"RIGHT": (0, 1)}, 13: {"LEFT": (0, 1)}}

_, h = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(h)]
input()
while True:
    x, y, pos = input().split()
    x, y = int(x), int(y)
    dx, dy = MOVES[grid[y][x]][pos]
    print(x + dx, y + dy, flush=True)
