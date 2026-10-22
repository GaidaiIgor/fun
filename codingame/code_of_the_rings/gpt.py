"""Solves CodinGame Code of the Rings with beam search and repeated-rune loops."""

from __future__ import annotations

import sys


ZONE_COUNT = 30
RUNE_COUNT = 27
BEAM_WIDTH = 40
LOOP_MIN_REPEAT = 20
ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RUNE_VALUES = {rune: value for value, rune in enumerate(ALPHABET)}
State = tuple[int, int, tuple[int, ...], str]


def build_commands(size: int, forward: str, backward: str) -> list[list[str]]:
    """Builds shortest circular command strings for a ring size and returns them by source and target."""
    commands = []
    for source in range(size):
        row = []
        for target in range(size):
            distance = (target - source) % size
            row.append(forward * distance if distance <= size // 2 else backward * (size - distance))
        commands.append(row)
    return commands


MOVES = build_commands(ZONE_COUNT, ">", "<")
ROLLS = build_commands(RUNE_COUNT, "+", "-")
MOVE_LENGTHS = [[len(command) for command in row] for row in MOVES]
ROLL_LENGTHS = [[len(command) for command in row] for row in ROLLS]


def split_phrase(phrase: str) -> list[tuple[str, int]]:
    """Groups long repeated characters in the phrase and returns printable tokens with their repeat counts."""
    tokens = []
    index = 0
    while index < len(phrase):
        end = index + 1
        while end < len(phrase) and phrase[end] == phrase[index]:
            end += 1
        count = end - index
        if count >= LOOP_MIN_REPEAT:
            tokens.append((phrase[index], count))
        else:
            tokens.extend((phrase[index], 1) for _ in range(count))
        index = end
    return tokens


def update_best(best: dict[tuple[int, tuple[int, ...]], State], state: State):
    """Stores the cheapest state for its cursor and cells in the beam candidate map."""
    key = (state[1], state[2])
    old = best.get(key)
    if old is None or state[0] < old[0]:
        best[key] = state


def prune(best: dict[tuple[int, tuple[int, ...]], State]) -> list[State]:
    """Keeps the most compact states from a candidate map and returns the next beam."""
    return sorted(best.values(), key=lambda state: state[0])[:BEAM_WIDTH]


def set_cell(cells: tuple[int, ...], zone: int, value: int) -> tuple[int, ...]:
    """Replaces one zone value in the cell tuple and returns the updated forest state."""
    return cells if cells[zone] == value else cells[:zone] + (value,) + cells[zone + 1:]


def expand_single(states: list[State], target: int) -> list[State]:
    """Expands every beam state by printing one target rune and returns the pruned successors."""
    best = {}
    for cost, position, cells, code in states:
        for zone in range(ZONE_COUNT):
            move = MOVES[position][zone]
            roll = ROLLS[cells[zone]][target]
            new_cost = cost + MOVE_LENGTHS[position][zone] + ROLL_LENGTHS[cells[zone]][target] + 1
            update_best(best, (new_cost, zone, set_cell(cells, zone, target), code + move + roll + "."))
    return prune(best)


def build_loop(cells: tuple[int, ...], position: int, output_zone: int, counter_zone: int, target: int, count: int) -> tuple[int, int, tuple[int, ...], str]:
    """Builds commands using an output and counter zone and returns added cost, final cursor, cells, and code."""
    code = MOVES[position][output_zone] + ROLLS[cells[output_zone]][target]
    cost = MOVE_LENGTHS[position][output_zone] + ROLL_LENGTHS[cells[output_zone]][target]
    cursor = output_zone
    counter_value = cells[counter_zone]
    remaining = count
    while remaining >= LOOP_MIN_REPEAT:
        chunk = min(RUNE_COUNT - 1, remaining)
        if remaining >= 2 * LOOP_MIN_REPEAT and remaining - chunk < LOOP_MIN_REPEAT:
            chunk = remaining - LOOP_MIN_REPEAT
        down_start = chunk
        up_start = RUNE_COUNT - chunk
        if ROLL_LENGTHS[counter_value][down_start] <= ROLL_LENGTHS[counter_value][up_start]:
            start = down_start
            step = "-"
        else:
            start = up_start
            step = "+"
        loop_code = "[" + MOVES[counter_zone][output_zone] + "." + MOVES[output_zone][counter_zone] + step + "]"
        code += MOVES[cursor][counter_zone] + ROLLS[counter_value][start] + loop_code
        cost += MOVE_LENGTHS[cursor][counter_zone] + ROLL_LENGTHS[counter_value][start] + len(loop_code)
        cursor = counter_zone
        counter_value = 0
        remaining -= chunk
    if remaining:
        code += MOVES[cursor][output_zone] + "." * remaining
        cost += MOVE_LENGTHS[cursor][output_zone] + remaining
        cursor = output_zone
    cells = set_cell(set_cell(cells, output_zone, target), counter_zone, 0)
    return cost, cursor, cells, code


def expand_repeated(states: list[State], target: int, count: int) -> list[State]:
    """Expands beam states by printing a repeated rune and returns successors using direct output or loops."""
    best = {}
    for cost, position, cells, code in states:
        for output_zone in range(ZONE_COUNT):
            move = MOVES[position][output_zone]
            roll = ROLLS[cells[output_zone]][target]
            new_cost = cost + MOVE_LENGTHS[position][output_zone] + ROLL_LENGTHS[cells[output_zone]][target] + count
            update_best(best, (new_cost, output_zone, set_cell(cells, output_zone, target), code + move + roll + "." * count))
            for counter_zone in range(ZONE_COUNT):
                if counter_zone == output_zone:
                    continue
                loop_cost, cursor, new_cells, loop_code = build_loop(cells, position, output_zone, counter_zone, target, count)
                update_best(best, (cost + loop_cost, cursor, new_cells, code + loop_code))
    return prune(best)


def solve(phrase: str) -> str:
    """Finds a compact Blub program for the magic phrase and returns the command sequence."""
    states = [(0, 0, (0,) * ZONE_COUNT, "")]
    for char, count in split_phrase(phrase):
        target = RUNE_VALUES[char]
        states = expand_repeated(states, target, count) if count >= LOOP_MIN_REPEAT else expand_single(states, target)
    return min(states, key=lambda state: state[0])[3]


def main():
    """Reads the magic phrase, solves it, and prints the instruction sequence."""
    print(solve(sys.stdin.readline().rstrip("\n")))


if __name__ == "__main__":
    main()
