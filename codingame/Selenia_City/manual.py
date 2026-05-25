"""Reads Selenia City turns and emits a no-op action skeleton."""

from __future__ import annotations

from collections.abc import Iterator
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

MONTH_DAYS = 20
POD_CAPACITY = 10


@dataclass(slots=True)
class GameState:
    """Stores the complete current game snapshot while reading monthly input.
    month is the current zero-based month. resources stores available resources. buildings keeps known buildings ordered by id.
    pod_routes stores transport pod paths. Tube ownership is stored inside buildings."""
    month: int = -1
    resources: int = 0
    buildings: dict[int, Building] = field(default_factory=dict)
    pod_routes: dict[int, list[int]] = field(default_factory=dict)

    @staticmethod
    def read_month(previous: GameState) -> GameState:
        """Reads one complete monthly input snapshot.
        previous provides persistent buildings from earlier months. Returns the parsed fresh state for the next month."""
        state = GameState()
        state.month = previous.month + 1
        state.buildings = {building_id: building.copy_building_only() for building_id, building in previous.buildings.items()}
        state.resources = int(input())
        state.read_tubes()
        state.pod_routes = state.read_pod_routes()
        state.read_buildings()
        state.buildings = dict(sorted(state.buildings.items()))
        return state

    def read_tubes(self):
        """Reads current route input into self.
        Tube routes are written to endpoint building tubes, while teleport routes are written to endpoint building teleport fields."""
        for _ in range(int(input())):
            start_id, end_id, capacity = map(int, input().split())
            if capacity == 0:
                self.buildings[start_id].teleport = (-1, end_id)
                self.buildings[end_id].teleport = (start_id, -1)
            else:
                self.buildings[start_id].tubes[end_id] = capacity
                self.buildings[end_id].tubes[start_id] = capacity

    def read_pod_routes(self) -> dict[int, list[int]]:
        """Reads the currently existing transport pod routes.
        Returns pod routes keyed by pod identifier and ordered by pod identifier."""
        pod_routes = {}
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            pod_routes[values[0]] = values[2:]
        return dict(sorted(pod_routes.items()))

    def read_buildings(self):
        """Reads buildings constructed for the current month into self.buildings."""
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            astronaut_types = {}
            if values[0] == 0:
                astronaut_types = dict(Counter(values[5:]))
            building = Building(values[1], values[0], np.array((values[2], values[3]), dtype=int), astronaut_types=astronaut_types)
            self.buildings[building.id] = building

    def simulate_month(self) -> SimulationSummary:
        """Simulates astronaut movement before the current month ends.
        Returns a SimulationSummary containing score and wait_times split by building id and astronaut type."""
        distance_matrix = self.all_distances()
        terminals = self.init_groups(distance_matrix)
        summary = SimulationSummary()
        module_load = {}
        pods = {pod_id: Pod(pod_id, path) for pod_id, path in self.pod_routes.items()}
        for day in range(MONTH_DAYS):
            self.process_teleport_phase(terminals, distance_matrix)
            summary.score += self.settle_arrivals(day, module_load, terminals, queue="departing")
            self.process_tube_phase(terminals, pods, distance_matrix)
            for terminal in terminals.values():
                if terminal.departing.groups:
                    wait_times = summary.wait_times.setdefault(terminal.building_id, {})
                    for group in terminal.departing.groups:
                        wait_times[group.kind] = wait_times.get(group.kind, 0) + group.size
            summary.score += self.settle_arrivals(day + 1, module_load, terminals, queue="arriving")
        return summary

    def init_groups(self, distance_matrix: NDArray[int]) -> dict[int, Terminal]:
        """Initializes monthly astronaut groups at landing pad terminals.
        distance_matrix provides shortest distances used to retain nearest targets. Returns terminals keyed by building id."""
        terminals = {building_id: Terminal(building_id) for building_id in self.buildings}
        for building in self.buildings.values():
            if building.kind == 0:
                for astronaut_type, count in building.astronaut_types.items():
                    target_ids = [target.id for target in self.buildings.values() if target.kind == astronaut_type]
                    sub_terminal = terminals[building.id].departing
                    sub_terminal.add_group(AstronautGroup.make_group(sub_terminal, astronaut_type, target_ids, count, distance_matrix))
        return terminals

    def process_teleport_phase(self, terminals: dict[int, Terminal], distance_matrix: NDArray[int]):
        """Moves eligible departing groups through teleporters.
        terminals stores simulation queues keyed by building id. distance_matrix provides shortest distances for target selection."""
        for terminal in terminals.values():
            outgoing_id = self.buildings[terminal.building_id].teleport[1]
            if outgoing_id == -1:
                continue
            group_index = 0
            while group_index < len(terminal.departing.groups):
                group = terminal.departing.groups[group_index]
                new_group = AstronautGroup.make_group(terminals[outgoing_id].departing, group.kind, group.target_building_ids, group.size, distance_matrix)
                if new_group.target_distance <= group.target_distance:
                    self.transfer_group(terminal.departing, group_index, new_group)
                else:
                    group_index += 1

    def process_tube_phase(self, terminals: dict[int, Terminal], pods: dict[int, Pod], distance_matrix: NDArray[int]):
        """Moves pods through tubes and boards groups that get closer to a target.
        terminals stores simulation queues keyed by building id. pods stores pod movement state keyed by pod id.
        distance_matrix provides shortest distances for target selection."""
        tube_uses = {}
        for pod in pods.values():
            start_id, destination_id = pod.get_building_ids()
            edge = (start_id, destination_id) if start_id < destination_id else (destination_id, start_id)
            if start_id != destination_id and tube_uses.get(edge, 0) < self.buildings[edge[0]].tubes[edge[1]]:
                pod.seats = POD_CAPACITY
                group_index = 0
                source_sub_terminal = terminals[start_id].departing
                departing = source_sub_terminal.groups
                while pod.seats > 0 and group_index < len(departing):
                    group = departing[group_index]
                    boarding_count = min(group.size, pod.seats)
                    new_group = \
                        AstronautGroup.make_group(terminals[destination_id].arriving, group.kind, group.target_building_ids, boarding_count, distance_matrix)
                    if new_group.target_distance < group.target_distance:
                        pod.seats -= boarding_count
                        if boarding_count == group.size:
                            self.transfer_group(source_sub_terminal, group_index, new_group)
                        else:
                            group.size -= boarding_count
                            new_group.sub_terminal.add_group(new_group)
                            group_index += 1
                    else:
                        group_index += 1
                pod.path_index = pod.next_path_index()
                tube_uses[edge] = tube_uses.get(edge, 0) + 1

    @staticmethod
    def transfer_group(source_sub_terminal: SubTerminal, group_index: int, new_group: AstronautGroup) -> AstronautGroup:
        """Transfers a group by replacing the source entry with new_group.
        source_sub_terminal provides the source queue. group_index selects the source group. new_group is added to its destination queue.
        Returns new_group after it is added."""
        source_sub_terminal.groups.pop(group_index)
        new_group.sub_terminal.add_group(new_group)
        return new_group

    def all_distances(self) -> NDArray[int]:
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

    def all_tubes(self) -> Iterator[tuple[tuple[int, int], int]]:
        """Iterates through tube capacities owned by buildings.
        self provides buildings with adjacent tube data. Returns sorted endpoint id pairs and the corresponding tube capacity once per tube."""
        for building_id, building in self.buildings.items():
            for other_id, capacity in building.tubes.items():
                if building_id < other_id:
                    yield (building_id, other_id), capacity

    @staticmethod
    def settle_arrivals(day: int, module_load: dict[int, int], terminals: dict[int, Terminal], queue: str) -> int:
        """Scores groups already standing in matching modules.
        day determines the speed score. module_load stores arrivals by module id. terminals provides queues to inspect.
        queue selects whether departing or arriving queues are checked.
        Returns the score gained from groups settled by this call."""
        score_gain = 0
        for terminal in terminals.values():
            sub_terminal = getattr(terminal, queue)
            building_id = sub_terminal.parent.building_id
            group_index = 0
            while group_index < len(sub_terminal.groups):
                group = sub_terminal.groups[group_index]
                if building_id in group.target_building_ids:
                    current_load = module_load.get(building_id, 0)
                    scoring_count = max(0, min(group.size, 50 - current_load))
                    score_gain += group.size * (50 - day) + scoring_count * (50 - current_load) - scoring_count * (scoring_count - 1) // 2
                    module_load[building_id] = current_load + group.size
                    sub_terminal.groups.pop(group_index)
                else:
                    group_index += 1
            if queue == "arriving":
                for group in sub_terminal.groups:
                    sub_terminal.parent.departing.add_group(group)
                sub_terminal.groups = []
        return score_gain

@dataclass(slots=True)
class Building:
    """Stores one landing pad or lunar module.
    id is the building identifier. kind is zero for a landing pad, or the positive module type for a lunar module. coords stores
    position. tubes stores adjacent tube capacities keyed by neighbor id. teleport stores incoming and outgoing building ids.
    astronaut_types stores demand."""
    id: int
    kind: int
    coords: NDArray[int]
    tubes: dict[int, int] = field(default_factory=dict)
    teleport: tuple[int, int] = (-1, -1)
    astronaut_types: dict[int, int] = field(default_factory=dict)

    def copy_building_only(self) -> Building:
        """Copies this building without current-month route attachments.
        Returns a building sharing persistent coords and astronaut_types with self."""
        return Building(self.id, self.kind, self.coords, astronaut_types=self.astronaut_types)

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
class AstronautGroup:
    """Stores astronauts moving together during monthly movement simulation.
    sub_terminal is the queue where the group stands. kind stores the destination module kind.
    target_building_ids stores candidate destination modules. target_distance stores the current distance to the nearest candidate target.
    size stores the number of astronauts in the group."""
    sub_terminal: SubTerminal
    kind: int
    target_building_ids: list[int]
    target_distance: int
    size: int

    @staticmethod
    def make_group(sub_terminal: SubTerminal, astronaut_type: int, target_building_ids: list[int], size: int, distance_matrix: NDArray[int]) -> AstronautGroup:
        """Creates an astronaut group at sub_terminal with only nearest targets retained.
        sub_terminal is the current queue. astronaut_type is the destination module kind. target_building_ids are candidate module ids.
        size is the number of astronauts. distance_matrix gives shortest distances by building id.
        Returns the new AstronautGroup with target_distance set."""
        building_id = sub_terminal.parent.building_id
        target_distance = min(distance_matrix[building_id, target_id] for target_id in target_building_ids)
        targets = [target_id for target_id in target_building_ids if distance_matrix[building_id, target_id] == target_distance]
        return AstronautGroup(sub_terminal, astronaut_type, targets, target_distance, size)


@dataclass(slots=True)
class Terminal:
    """Stores astronaut groups at one building during movement simulation.
    building_id is the building represented by this terminal. arriving stores groups that reached the building during the current pod phase.
    departing stores groups available to board pods."""
    building_id: int
    arriving: SubTerminal = field(init=False)
    departing: SubTerminal = field(init=False)

    def __post_init__(self):
        """Creates the arriving and departing subterminals owned by this terminal."""
        self.arriving = SubTerminal(self)
        self.departing = SubTerminal(self)


@dataclass(slots=True)
class SubTerminal:
    """Stores one movement queue owned by a terminal.
    parent is the owning terminal. groups stores astronaut groups waiting in this queue."""
    parent: Terminal
    groups: list[AstronautGroup] = field(default_factory=list)

    def add_group(self, group: AstronautGroup):
        """Adds group to this queue and updates its reverse reference.
        group is the astronaut group that should wait in this subterminal."""
        group.sub_terminal = self
        self.groups.append(group)


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
        state = GameState.read_month(state)
        print(choose_action(state))


def choose_action(state: GameState) -> str:
    """Chooses the action line for the current month.
    state provides the current parsed game snapshot. Returns a valid action string for CodinGame."""
    return "WAIT"


if __name__ == "__main__":
    play()
