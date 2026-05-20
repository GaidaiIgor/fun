"""Reads Selenia City turns and emits a no-op action skeleton."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


class GameState:
    """Stores the complete current game snapshot while reading monthly input.
    Buildings keeps every known building. Resources, magnetic tubes, pods, and teleports are refreshed each month."""
    month: int
    resources: int
    buildings: dict[int, Building]
    tubes: list[Tube]
    pods: dict[int, list[int]]

    def __init__(self):
        """Initializes empty city memory before the first month is parsed."""
        self.month = 0
        self.resources = 0
        self.buildings = {}
        self.tubes = []
        self.pods = {}


    def play(self):
        """Runs the interactive game loop until the input stream ends."""
        while True:
            self.read_month()
            print(self.choose_actions())
            self.month += 1

    def read_month(self):
        """Reads one complete monthly input snapshot into the game state."""
        self.resources = int(input())
        for building in self.buildings.values():
            building.tubes.clear()
            building.teleport = (-1, -1)
        self.tubes = self.read_tubes()
        self.pods = self.read_pods()
        self.read_buildings()

    def read_tubes(self) -> list[Tube]:
        """Reads current route input and returns only magnetic tubes."""
        tubes = []
        for _ in range(int(input())):
            start_id, end_id, capacity = map(int, input().split())
            if capacity == 0:
                self.buildings[start_id].teleport = (-1, end_id)
                self.buildings[end_id].teleport = (start_id, -1)
            else:
                tube = Tube((start_id, end_id), capacity)
                tubes.append(tube)
                self.buildings[start_id].tubes.append(tube)
                self.buildings[end_id].tubes.append(tube)
        return tubes

    def read_pods(self) -> dict[int, list[int]]:
        """Reads the currently existing transport pods.
        Returns pod paths keyed by pod identifier."""
        pods = {}
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            pods[values[0]] = values[2:2 + values[1]]
        return pods

    def read_buildings(self):
        """Reads buildings constructed for the current month."""
        for _ in range(int(input())):
            building = self.parse_building(list(map(int, input().split())))
            self.buildings[building.id] = building

    def parse_building(self, values: list[int]) -> Building:
        """Builds a typed building object from one input line.
        Values are integer tokens from a landing pad or module description. Returns the parsed building description."""
        if values[0] == 0:
            astronaut_types = dict(Counter(values[5:5 + values[4]]))
            return Building(values[1], 0, np.array((values[2], values[3]), dtype=int), astronaut_types=astronaut_types)
        return Building(values[1], values[0], np.array((values[2], values[3]), dtype=int))

    def choose_actions(self) -> str:
        """Chooses the action line for the current month.
        Returns a valid action string for CodinGame."""
        return "WAIT"


@dataclass(slots=True)
class Building:
    """Stores one landing pad or lunar module.
    Kind is zero for a landing pad, or the positive module type for a lunar module. Coordinates use a two-value integer array.
    Tubes stores current magnetic tube attachments. Teleport stores incoming and outgoing building ids, with -1 for none."""
    id: int
    kind: int
    coords: NDArray[int]
    tubes: list[Tube] = field(default_factory=list)
    teleport: tuple[int, int] = (-1, -1)
    astronaut_types: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class Tube:
    """Stores one existing magnetic tube.
    Buildings stores the two tube endpoints. Capacity is the simultaneous pod capacity."""
    buildings: tuple[int, int]
    capacity: int


if __name__ == "__main__":
    GameState().play()
