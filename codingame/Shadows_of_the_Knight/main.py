"""Solves CodinGame Shadows of the Knight with binary search over possible windows."""

w, h = [int(value) for value in input().split()]
input()
x, y = [int(value) for value in input().split()]

min_x, max_x = 0, w - 1
min_y, max_y = 0, h - 1

while True:
    direction = input()

    if "U" in direction:
        max_y = y - 1
    elif "D" in direction:
        min_y = y + 1
    else:
        min_y = max_y = y

    if "L" in direction:
        max_x = x - 1
    elif "R" in direction:
        min_x = x + 1
    else:
        min_x = max_x = x

    x = (min_x + max_x) // 2
    y = (min_y + max_y) // 2
    print(f"{x} {y}")
