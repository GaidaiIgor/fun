"""Reads Selenia City turns and emits a no-op action skeleton."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

MONTH_DAYS = 20
POD_CAPACITY = 10


@dataclass(slots=True)
class GameState:
    """Stores the complete current game snapshot while reading monthly input.
    month is the current zero-based month. resources stores available resources. buildings keeps known buildings ordered by id.
    pods stores transport pods ordered by id. Tube ownership and monthly astronaut arrivals are stored inside buildings."""
    month: int = -1
    resources: int = 0
    buildings: dict[int, Building] = field(default_factory=dict)
    pods: list[Pod] = field(default_factory=list)

    def read_month_input(self):
        """Reads one complete monthly input snapshot into self.
        self receives updated resources plus newly constructed buildings."""
        self.month += 1
        self.resources = int(input())
        for _ in range(int(input())):
            input()
        for _ in range(int(input())):
            input()
        self.read_new_buildings()

    def read_new_buildings(self):
        """Reads buildings constructed for the current month.
        New buildings are stored in self.buildings. New landing-pad astronauts are initialized in each building initial terminal."""
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            building = Building(values[1], values[0], np.array((values[2], values[3]), dtype=int))
            self.buildings[building.id] = building
            if building.kind == 0:
                building.initial = Terminal(building)
                for index, astronaut_type in enumerate(values[5:]):
                    building.initial.astronauts.append(Astronaut(building.initial, 1000 * building.id + index, astronaut_type, [], 0))

    def simulate_month(self) -> SimulationSummary:
        """Simulates astronaut movement before the current month ends.
        Returns a SimulationSummary containing score and wait_times split by building id and astronaut type."""
        distance_matrix = self.get_all_distances()
        self.reset(distance_matrix)
        summary = SimulationSummary()
        for day in range(MONTH_DAYS):
            self.process_teleport_phase(self.buildings, distance_matrix)
            summary.score += self.settle_arrivals(day, self.buildings)
            self.process_tube_phase(self.buildings, self.pods, distance_matrix)
            for building in self.buildings.values():
                if building.departing.astronauts:
                    wait_times = summary.wait_times.setdefault(building.id, {})
                    for astronaut in building.departing.astronauts:
                        wait_times[astronaut.kind] = wait_times.get(astronaut.kind, 0) + 1
            summary.score += self.settle_arrivals(day + 1, self.buildings)
        return summary

    def get_all_distances(self) -> NDArray[int]:
        """Calculates shortest travel distances between all known buildings.
        self supplies buildings, tubes, and teleport links. Returns distances where distances[start_id, end_id] is the shortest travel distance."""
        building_count = len(self.buildings)
        unreachable_distance = building_count + 1
        distances = np.full((building_count, building_count), unreachable_distance, dtype=int)
        for building_id in self.buildings:
            distances[building_id, building_id] = 0
            outgoing_id = self.buildings[building_id].teleport[1]
            if outgoing_id != -1:
                distances[building_id, outgoing_id] = 0
        for (start_id, end_id), _ in self.all_tubes():
            distances[start_id, end_id] = 1
            distances[end_id, start_id] = 1
        for middle_id in range(building_count):
            distances = np.minimum(distances, distances[:, [middle_id]] + distances[[middle_id], :])
        return distances

    def reset(self, distance_matrix: NDArray[int]):
        """Resets live building terminals and pods to their monthly initial state.
        distance_matrix provides shortest distances used to retain nearest targets. self provides buildings whose arriving terminals are cleared
        and whose departing terminals are copied from initial terminals, plus pods whose path_index and seats are reset."""
        for building in self.buildings.values():
            building.population = 0
            building.arriving = Terminal(building)
            building.departing = Terminal(building)
            for astronaut in building.initial.astronauts:
                building.departing.astronauts.append(astronaut.clone(building.departing, self.buildings, distance_matrix))
        for pod in self.pods:
            pod.path_index = 0

    def process_teleport_phase(self, buildings: dict[int, Building], distance_matrix: NDArray[int]):
        """Moves eligible departing astronauts through teleporters.
        buildings stores simulation queues keyed by building id. distance_matrix provides shortest distances for target selection."""
        for building in buildings.values():
            outgoing_id = self.buildings[building.id].teleport[1]
            if outgoing_id == -1:
                continue
            remaining_astronauts = []
            for astronaut in building.departing.astronauts:
                new_astronaut = astronaut.clone(buildings[outgoing_id].arriving, buildings, distance_matrix)
                if new_astronaut.target_distance <= astronaut.target_distance:
                    buildings[outgoing_id].arriving.astronauts.append(new_astronaut)
                else:
                    remaining_astronauts.append(astronaut)
            building.departing.astronauts = remaining_astronauts

    def process_tube_phase(self, buildings: dict[int, Building], pods: list[Pod], distance_matrix: NDArray[int]):
        """Moves pods through tubes and boards astronauts that get closer to a target.
        buildings stores simulation queues keyed by building id. pods stores pod movement state ordered by pod id.
        distance_matrix provides shortest distances for target selection."""
        tube_uses = {}
        for pod in pods:
            start_id, destination_id = pod.get_building_ids()
            edge = (start_id, destination_id) if start_id < destination_id else (destination_id, start_id)
            if start_id != destination_id and tube_uses.get(edge, 0) < self.buildings[edge[0]].tubes[edge[1]]:
                pod.seats = POD_CAPACITY
                source_terminal = buildings[start_id].departing
                remaining_astronauts = []
                for astronaut in source_terminal.astronauts:
                    if pod.seats > 0:
                        new_astronaut = astronaut.clone(buildings[destination_id].arriving, buildings, distance_matrix)
                        if new_astronaut.target_distance < astronaut.target_distance:
                            pod.seats -= 1
                            buildings[destination_id].arriving.astronauts.append(new_astronaut)
                            continue
                    remaining_astronauts.append(astronaut)
                source_terminal.astronauts = remaining_astronauts
                pod.path_index = pod.next_path_index()
                tube_uses[edge] = tube_uses.get(edge, 0) + 1

    def all_tubes(self) -> Iterator[tuple[tuple[int, int], int]]:
        """Iterates through tube capacities owned by buildings.
        self provides buildings with adjacent tube data. Returns sorted endpoint id pairs and the corresponding tube capacity once per tube."""
        for building_id, building in self.buildings.items():
            for other_id, capacity in building.tubes.items():
                if building_id < other_id:
                    yield (building_id, other_id), capacity

    @staticmethod
    def settle_arrivals(day: int, buildings: dict[int, Building]) -> int:
        """Scores astronauts in arriving terminals and moves unsettled astronauts to departing terminals.
        day determines the speed score. buildings provides arrival queues and stores population for balancing score.
        Returns the score gained from astronauts settled by this call."""
        score_gain = 0
        for building in buildings.values():
            remaining_astronauts = []
            for astronaut in building.arriving.astronauts:
                if building.id in astronaut.target_building_ids:
                    score_gain += 50 - day + max(0, 50 - building.population)
                    building.population += 1
                else:
                    remaining_astronauts.append(astronaut)
            building.arriving.astronauts = []
            for astronaut in remaining_astronauts:
                astronaut.terminal = building.departing
                building.departing.astronauts.append(astronaut)
            building.departing.astronauts.sort(key=lambda astronaut: astronaut.id)
        return score_gain


@dataclass(slots=True)
class Building:
    """Stores one landing pad or lunar module.
    id is the building identifier. kind is zero for a landing pad, or the positive module type for a lunar module. coords stores
    position. tubes stores adjacent tube capacities keyed by neighbor id. teleport stores incoming and outgoing building ids.
    population stores the number of astronauts settled in this building during the current month.
    initial stores the monthly departing terminal template. arriving stores astronauts that reached the building during the current pod phase.
    departing stores astronauts available to board pods."""
    id: int
    kind: int
    coords: NDArray[int]
    tubes: dict[int, int] = field(default_factory=dict)
    teleport: tuple[int, int] = (-1, -1)
    population: int = 0
    initial: Terminal = field(init=False)
    arriving: Terminal = field(init=False)
    departing: Terminal = field(init=False)


@dataclass(slots=True)
class Pod:
    """Stores one transport pod and its monthly movement state.
    id is the pod identifier. path stores the itinerary. path_index stores the current index in path. seats stores empty seats."""
    id: int
    path: list[int]
    path_index: int = 0
    seats: int = 0

    def get_building_ids(self) -> tuple[int, int]:
        """Gets this pod current and next building ids.
        Returns current and next building ids, or the current id twice when this pod has stopped at the end of a non-looping path."""
        next_index = self.next_path_index()
        return self.path[self.path_index], self.path[next_index]

    def next_path_index(self) -> int:
        """Gets the next path index from path_index.
        Returns path_index when this pod has stopped at the end of a non-looping path."""
        if self.path_index == len(self.path) - 1:
            return 1 if self.path[0] == self.path[-1] else self.path_index
        return self.path_index + 1


@dataclass(slots=True)
class Terminal:
    """Stores one astronaut movement queue owned by a building.
    building is the owning building. astronauts stores astronauts waiting in this queue."""
    building: Building
    astronauts: list[Astronaut] = field(default_factory=list)


@dataclass(slots=True)
class Astronaut:
    """Stores one astronaut during monthly movement simulation.
    terminal is the queue where the astronaut stands. id gives boarding priority. kind stores the destination module kind.
    target_building_ids stores candidate destination modules. target_distance stores the current distance to the nearest candidate target."""
    terminal: Terminal
    id: int
    kind: int
    target_building_ids: list[int]
    target_distance: int

    def clone(self, terminal: Terminal, buildings: dict[int, Building], distance_matrix: NDArray[int]) -> Astronaut:
        """Creates a copy of self at terminal with only nearest targets retained.
        terminal is the destination queue. buildings provides candidate target modules. distance_matrix gives shortest distances by building id.
        Returns the new Astronaut with target_distance set."""
        astronaut = Astronaut(terminal, self.id, self.kind, self.target_building_ids, 0)
        astronaut.update_targets(buildings, distance_matrix)
        return astronaut

    def update_targets(self, buildings: dict[int, Building], distance_matrix: NDArray[int]):
        """Updates target_building_ids and target_distance for this astronaut.
        buildings provides candidate target modules when target_building_ids is empty. distance_matrix gives shortest distances."""
        if not self.target_building_ids:
            self.target_building_ids = [building.id for building in buildings.values() if building.kind == self.kind]
        building_id = self.terminal.building.id
        self.target_distance = min(distance_matrix[building_id, target_id] for target_id in self.target_building_ids)
        self.target_building_ids = [target_id for target_id in self.target_building_ids if distance_matrix[building_id, target_id] == self.target_distance]

@dataclass(slots=True)
class SimulationSummary:
    """Stores the result of one monthly movement simulation.
    score is the total score earned during the month. wait_times maps building id to person-days by astronaut type."""
    score: int = 0
    wait_times: dict[int, dict[int, int]] = field(default_factory=dict)


def play():
    """Runs the interactive game loop."""
    state = GameState()
    while True:
        state.read_month_input()
        print(choose_action(state))


def choose_action(state: GameState) -> str:
    """Chooses the action line for the current month.
    state provides the current parsed game snapshot. Returns a valid action string for CodinGame."""
    return "WAIT"


if __name__ == "__main__":
    play()
