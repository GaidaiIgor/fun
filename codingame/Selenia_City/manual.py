"""Reads Selenia City turns and emits a no-op action skeleton."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

MONTH_DAYS = 20
POD_CAPACITY = 10
PathsByPair = dict[tuple[int, int], list[list[int]]]


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
                    building.initial.astronauts.append(Astronaut(building.initial, 1000 * building.id + index, astronaut_type, np.empty((0, 0), dtype=int)))

    def simulate_month(self) -> SimulationSummary:
        """Simulates astronaut movement before the current month ends.
        Returns a SimulationSummary containing score and wait_times split by building id and astronaut type."""
        distance_matrix, paths_by_pair = self.get_shortest_distances_paths()
        self.reset(distance_matrix, paths_by_pair)
        summary = SimulationSummary()
        for day in range(MONTH_DAYS):
            self.process_teleport_phase(self.buildings)
            summary.score += self.settle_arrivals(day, self.buildings)
            self.process_tube_phase(self.buildings, self.pods)
            for building in self.buildings.values():
                if building.departing.astronauts:
                    wait_times = summary.wait_times.setdefault(building.id, {})
                    for astronaut in building.departing.astronauts:
                        wait_times[astronaut.kind] = wait_times.get(astronaut.kind, 0) + 1
            summary.score += self.settle_arrivals(day + 1, self.buildings)
        return summary

    def get_shortest_distances_paths(self) -> tuple[NDArray[int], PathsByPair]:
        """Calculates shortest travel distances and all paths realizing each distance.
        Returns distances where distances[start_id, end_id] is the shortest travel distance, and paths where paths[start_id, end_id] contains every
        shortest building sequence from start_id to end_id."""
        building_count = len(self.buildings)
        unreachable_distance = building_count + 1
        distances = np.full((building_count, building_count), unreachable_distance, dtype=int)
        paths = {}
        for building_id in self.buildings:
            distances[building_id, building_id] = 0
            paths[building_id, building_id] = [[building_id]]
            outgoing_id = self.buildings[building_id].teleport[1]
            if outgoing_id != -1:
                distances[building_id, outgoing_id] = 0
                paths[building_id, outgoing_id] = [[building_id, outgoing_id]]
        for (start_id, end_id), _ in self.all_tubes():
            if distances[start_id, end_id] > 1:
                distances[start_id, end_id] = 1
                paths[start_id, end_id] = [[start_id, end_id]]
            if distances[end_id, start_id] > 1:
                distances[end_id, start_id] = 1
                paths[end_id, start_id] = [[end_id, start_id]]
        for middle_id in range(building_count):
            for start_id in range(building_count):
                for end_id in range(building_count):
                    if middle_id in (start_id, end_id) or (start_id, middle_id) not in paths or (middle_id, end_id) not in paths:
                        continue
                    new_distance = distances[start_id, middle_id] + distances[middle_id, end_id]
                    if new_distance <= distances[start_id, end_id]:
                        new_paths = self.combine_paths(paths[start_id, middle_id], paths[middle_id, end_id])
                        if new_distance < distances[start_id, end_id]:
                            distances[start_id, end_id] = new_distance
                            paths[start_id, end_id] = new_paths
                        else:
                            paths[start_id, end_id].extend(new_paths)
        return distances, paths

    @staticmethod
    def combine_paths(left_paths: list[list[int]], right_paths: list[list[int]]) -> list[list[int]]:
        """Combines shortest path prefixes and suffixes that meet at the same middle building.
        left_paths contains paths from a start building to the middle building. right_paths contains paths from the middle building to an end building.
        Returns paths from the start building to the end building."""
        paths = []
        for left_path in left_paths:
            for right_path in right_paths:
                paths.append([*left_path, *right_path[1:]])
        return paths

    def reset(self, distance_matrix: NDArray[int], paths_by_pair: PathsByPair):
        """Resets live building terminals and pods to their monthly initial state.
        distance_matrix and paths_by_pair provide shortest paths used to route astronauts. self provides buildings whose arriving terminals are cleared
        and whose departing terminals are copied from initial terminals, plus pods whose path_index is reset."""
        paths_by_start_kind = {}
        for building in self.buildings.values():
            building.population = 0
            building.arriving = Terminal(building)
            building.departing = Terminal(building)
            if building.kind == 0:
                for astronaut in building.initial.astronauts:
                    key = building.id, astronaut.kind
                    if key not in paths_by_start_kind:
                        paths_by_start_kind[key] = astronaut.get_paths(self.buildings, distance_matrix, paths_by_pair)
                    building.departing.astronauts.append(Astronaut(building.departing, astronaut.id, astronaut.kind, paths_by_start_kind[key]))
        for pod in self.pods:
            pod.path_index = 0

    def process_teleport_phase(self, buildings: dict[int, Building]):
        """Moves eligible departing astronauts through teleporters.
        buildings stores simulation queues keyed by building id."""
        for building in buildings.values():
            outgoing_id = self.buildings[building.id].teleport[1]
            if outgoing_id == -1:
                continue
            remaining_astronauts = []
            for astronaut in building.departing.astronauts:
                if astronaut.paths.shape[1] > 1 and np.any(astronaut.paths[:, 1] == outgoing_id):
                    astronaut.move(buildings[outgoing_id].arriving)
                else:
                    remaining_astronauts.append(astronaut)
            building.departing.astronauts = remaining_astronauts

    def process_tube_phase(self, buildings: dict[int, Building], pods: list[Pod]):
        """Moves pods through tubes and boards astronauts that get closer to a target.
        buildings stores simulation queues keyed by building id. pods stores pod movement state ordered by pod id."""
        tube_uses = {}
        for pod in pods:
            start_id, destination_id = pod.get_building_ids()
            edge = (start_id, destination_id) if start_id < destination_id else (destination_id, start_id)
            if start_id != destination_id and tube_uses.get(edge, 0) < self.buildings[edge[0]].tubes[edge[1]]:
                pod.seats = POD_CAPACITY
                remaining_astronauts = []
                for astronaut in buildings[start_id].departing.astronauts:
                    if pod.seats > 0 and astronaut.paths.shape[1] > 1 and np.any(astronaut.paths[:, 1] == destination_id):
                        astronaut.move(buildings[destination_id].arriving)
                        pod.seats -= 1
                        continue
                    remaining_astronauts.append(astronaut)
                buildings[start_id].departing.astronauts = remaining_astronauts
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
                if astronaut.paths.shape[1] == 1:
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
    paths stores possible paths to target modules, one path per row."""
    terminal: Terminal
    id: int
    kind: int
    paths: NDArray[int]

    def get_paths(self, buildings: dict[int, Building], distance_matrix: NDArray[int], paths_by_pair: PathsByPair) -> NDArray[int]:
        """Selects shortest paths from this astronaut building to the nearest matching modules.
        buildings stores candidate modules keyed by building id. distance_matrix gives shortest travel distances.
        paths_by_pair maps building pairs to all shortest paths between them. Returns a padded matrix with one path per row."""
        building_id = self.terminal.building.id
        target_ids = [building.id for building in buildings.values() if building.kind == self.kind and (building_id, building.id) in paths_by_pair]
        if not target_ids:
            return np.empty((0, 0), dtype=int)
        nearest_distance = min(distance_matrix[building_id, target_id] for target_id in target_ids)
        paths = []
        for target_id in target_ids:
            if distance_matrix[building_id, target_id] == nearest_distance:
                paths.extend(paths_by_pair[building_id, target_id])
        path_matrix = np.full((len(paths), max(len(path) for path in paths)), -1, dtype=int)
        for index, path in enumerate(paths):
            path_matrix[index, :len(path)] = path
        return path_matrix

    def move(self, terminal: Terminal):
        """Moves this astronaut to terminal.
        terminal is the destination queue receiving this astronaut."""
        self.terminal = terminal
        self.paths = self.paths[self.paths[:, 1] == terminal.building.id, 1:]
        while self.paths.shape[1] > 1 and np.all(self.paths[:, -1] == -1):
            self.paths = self.paths[:, :-1]
        terminal.astronauts.append(self)


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
