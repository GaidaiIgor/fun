"""Reads Selenia City turns and emits a no-op action skeleton."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class GameState:
    """Stores the complete current game snapshot while reading monthly input.
    Buildings keeps every known building. Resources, magnetic tubes, pods, and teleports are refreshed each month."""
    month: int = -1
    resources: int = 0
    buildings: dict[int, Building] = field(default_factory=dict)
    tubes: list[Tube] = field(default_factory=list)
    pods: dict[int, list[int]] = field(default_factory=dict)

    @staticmethod
    def read_month(previous: GameState) -> GameState:
        """Reads one complete monthly input snapshot into a fresh game state.
        Returns the parsed state for the next month."""
        state = GameState()
        state.month = previous.month + 1
        state.buildings = {id: building.copy_building_only() for id, building in previous.buildings.items()}
        state.resources = int(input())
        state.tubes = state.read_tubes()
        state.pods = state.read_pods()
        state.read_buildings()
        return state

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
            pods[values[0]] = values[2:]
        return pods

    def read_buildings(self):
        """Reads buildings constructed for the current month."""
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            astronaut_types = {}
            if values[0] == 0:
                astronaut_types = dict(Counter(values[5:]))
            building = Building(values[1], values[0], np.array((values[2], values[3]), dtype=int), astronaut_types=astronaut_types)
            self.buildings[building.id] = building

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

    def copy_building_only(self) -> Building:
        """Copies persistent building fields without current-month routes."""
        return Building(self.id, self.kind, self.coords, astronaut_types=self.astronaut_types)


@dataclass(slots=True)
class Tube:
    """Stores one existing magnetic tube.
    Buildings stores the two tube endpoints. Capacity is the simultaneous pod capacity."""
    buildings: tuple[int, int]
    capacity: int


def play():
    """Runs the interactive game loop."""
    state = GameState()
    while True:
        state = GameState.read_month(state)
        print(choose_action(state))


def choose_action(state: GameState) -> str:
    """Chooses the action line for the current month.
    Returns a valid action string for CodinGame."""
    return "WAIT"


if __name__ == "__main__":
    play()
