"""Reads initialization and first-turn data and echoes them to stderr."""

import sys


light_x, light_y, initial_thor_x, initial_thor_y = map(int, input().split())
remaining_turns = int(input())
print(light_x, light_y, initial_thor_x, initial_thor_y, file=sys.stderr)
print(remaining_turns, file=sys.stderr)
