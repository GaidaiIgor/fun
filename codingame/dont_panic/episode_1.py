"""Solves CodinGame Dont Panic by steering clones toward each floor target."""

_, _, _, exit_floor, exit_pos, _, _, nb_elevators = map(int, input().split())
targets_by_floor = {exit_floor: exit_pos}

for _ in range(nb_elevators):
    elevator_floor, elevator_pos = map(int, input().split())
    targets_by_floor[elevator_floor] = elevator_pos

while True:
    clone_floor_text, clone_pos_text, direction = input().split()
    clone_floor = int(clone_floor_text)
    clone_pos = int(clone_pos_text)

    if clone_floor == -1:
        print("WAIT")
        continue

    target_pos = targets_by_floor[clone_floor]
    should_block = direction == "LEFT" and clone_pos < target_pos or direction == "RIGHT" and clone_pos > target_pos
    print("BLOCK" if should_block else "WAIT")
