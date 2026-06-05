"""Reads Selenia City turns and emits planned infrastructure actions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy import linalg
from numpy.typing import NDArray

MONTH_DAYS = 20
POD_CAPACITY = 10
POD_COST = 1000
DistanceMatrix = dict[int, dict[int, int]]
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

    def fix_dynamic_pods(self):
        """Marks all pods in the current snapshot as no longer dynamically routed.
        self provides pods that existed before this month decision phase starts."""
        for pod in self.pods:
            pod.dynamic = False

    def choose_action(self) -> str:
        """Builds missing transport infrastructure for the current month.
        Returns a valid action string containing planned edge and pod commands."""
        actions = self.build_edges()
        actions.extend(self.build_pods())
        return ";".join(actions) if actions else "WAIT"

    def build_edges(self) -> list[str]:
        """Builds tubes until every known astronaut can reach a matching module.
        Returns tube action strings for the edges added to self."""
        actions = []
        while True:
            self.prepare_simulation()
            source_building, target_building = self.find_missing_path_buildings()
            if source_building is None or not self.build_edge(source_building, target_building):
                break
            actions.append(f"TUBE {source_building.id} {target_building.id}")
        return actions

    def find_missing_path_buildings(self) -> tuple[Building, Building]:
        """Finds the next landing pad and target module not connected by any astronaut path.
        Returns source and target buildings, or None when all kinds have paths."""
        for building in self.buildings.values():
            if building.kind == 0:
                initial_kinds = {astronaut.kind for astronaut in building.initial.astronauts}
                reachable_kinds = {astronaut.kind for astronaut in building.departing.astronauts}
                unreachable_kinds = initial_kinds - reachable_kinds
                for astronaut_kind in unreachable_kinds:
                    targets = (target for target in self.buildings.values() if target.kind == astronaut_kind)
                    target = min(targets, key=lambda target: linalg.norm(building.coords - target.coords))
                    return building, target
        return None, None

    def build_edge(self, start: Building, end: Building) -> bool:
        """Adds an affordable planned tube edge to the current game snapshot.
        start and end identify the buildings connected by the tube. Returns whether the tube was added."""
        cost = int(linalg.norm(start.coords - end.coords) * 10)
        if cost > self.resources:
            return False
        start.tubes[end.id] = 1
        end.tubes[start.id] = 1
        self.resources -= cost
        return True

    def build_pods(self) -> list[str]:
        """Builds dynamic pods until every tube edge is served or resources run out.
        Returns pod action strings with full generated routes for the pods added to self."""
        new_pods = []
        serviced_edges = {edge for pod in self.pods for edge in pod.service_edges}
        while self.resources >= POD_COST:
            edge = None
            for current_edge in self.iter_edges():
                if current_edge not in serviced_edges:
                    edge = current_edge
                    break
            if edge is None:
                break
            pod = self.build_pod(edge)
            new_pods.append(pod)
            serviced_edges.add(edge)
        if new_pods:
            self.simulate_month()
        return [" ".join(["POD", str(pod.id), *(str(building_id) for building_id in pod.path)]) for pod in new_pods]

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        """Iterates over tube edges stored in the current game snapshot.
        Returns each tube once as a canonical endpoint pair."""
        for building in self.buildings.values():
            for end_id in building.tubes:
                if building.id < end_id:
                    yield building.id, end_id

    def build_pod(self, edge: tuple[int, int]) -> Pod:
        """Creates a dynamic pod serving one canonical tube edge.
        edge gives the canonical tube endpoints served by the new pod. Returns the pod added to self."""
        pod = Pod(max((pod.id for pod in self.pods), default=0) + 1, [], dynamic=True, service_edges=[edge])
        self.pods.append(pod)
        self.resources -= POD_COST
        return pod

    @staticmethod
    def make_edge(start_id: int, end_id: int) -> tuple[int, int]:
        """Orders two building ids into a canonical tube edge.
        start_id and end_id identify the tube endpoints. Returns the canonical endpoint pair."""
        return (start_id, end_id) if start_id < end_id else (end_id, start_id)

    def simulate_month(self) -> SimulationSummary:
        """Simulates astronaut movement before the current month ends.
        Returns a SimulationSummary containing score and wait_times split by building id and astronaut type."""
        self.prepare_simulation()
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

    def prepare_simulation(self):
        """Initializes path, terminal, pod, and building counters before movement simulation.
        self receives freshly calculated astronaut paths, dynamic pod distances, reset terminals, reset pod state, and building pod counts."""
        self.reset()
        distance_matrix, paths_by_pair = self.get_shortest_distances_paths(set(self.buildings))
        self.update_dynamic_pod_distances()
        self.build_astronaut_paths(distance_matrix, paths_by_pair)
        self.count_pods()

    def reset(self):
        """Resets live building terminals and pods to their monthly initial state.
        self provides buildings whose live terminals are cleared and pods whose path_index is reset."""
        for building in self.buildings.values():
            building.population = 0
            building.arriving = Terminal(building)
            building.departing = Terminal(building)
        for pod in self.pods:
            pod.path_index = 0
            if pod.dynamic:
                pod.path = []

    def get_shortest_distances_paths(self, nodes: set[int], use_teleports: bool = True, return_paths: bool = True) -> tuple[DistanceMatrix, PathsByPair]:
        """Calculates shortest travel distances and all paths realizing each distance.
        nodes limits the start ids, end ids, and intermediate ids considered. use_teleports controls whether teleport links contribute zero-distance edges.
        return_paths controls whether shortest path lists are built. Returns distances where distances[start_id][end_id] is the shortest travel
        distance, and paths where paths[start_id, end_id] contains every shortest building sequence when return_paths is true."""
        unreachable_distance = len(self.buildings) + 1
        distances = {start_id: {end_id: unreachable_distance for end_id in nodes} for start_id in nodes}
        paths = {}
        for building_id in nodes:
            distances[building_id][building_id] = 0
            if return_paths:
                paths[building_id, building_id] = [[building_id]]
            outgoing_id = self.buildings[building_id].teleport[1]
            if use_teleports and outgoing_id in nodes:
                distances[building_id][outgoing_id] = 0
                if return_paths:
                    paths[building_id, outgoing_id] = [[building_id, outgoing_id]]
        for start_id in nodes:
            for end_id in self.buildings[start_id].tubes:
                if end_id not in nodes:
                    continue
                for source_id, destination_id in ((start_id, end_id), (end_id, start_id)):
                    if distances[source_id][destination_id] > 1:
                        distances[source_id][destination_id] = 1
                        if return_paths:
                            paths[source_id, destination_id] = [[source_id, destination_id]]
        for middle_id in nodes:
            for start_id in nodes:
                for end_id in nodes:
                    current_distance = distances[start_id][end_id]
                    new_distance = distances[start_id][middle_id] + distances[middle_id][end_id]
                    if new_distance > current_distance:
                        continue
                    distances[start_id][end_id] = new_distance
                    if not return_paths or middle_id in (start_id, end_id) or (start_id, middle_id) not in paths or (middle_id, end_id) not in paths:
                        continue
                    new_paths = self.combine_paths(paths[start_id, middle_id], paths[middle_id, end_id])
                    if new_distance < current_distance:
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

    def update_dynamic_pod_distances(self):
        """Stores tube-only shortest-distance matrices on dynamic pods.
        Each dynamic pod receives distances among nodes incident to its service_edges."""
        for pod in self.pods:
            if pod.dynamic:
                service_nodes = set()
                for start_id, end_id in pod.service_edges:
                    service_nodes.add(start_id)
                    service_nodes.add(end_id)
                pod.distance_matrix, _ = self.get_shortest_distances_paths(service_nodes, use_teleports=False, return_paths=False)

    def build_astronaut_paths(self, distance_matrix: DistanceMatrix, paths_by_pair: PathsByPair):
        """Builds monthly path matrices for astronauts waiting at landing pads.
        distance_matrix and paths_by_pair provide shortest paths. self receives reachable astronauts in each landing pad departing terminal."""
        paths_by_start_kind = {}
        for building in self.buildings.values():
            if building.kind == 0:
                for astronaut in building.initial.astronauts:
                    key = building.id, astronaut.kind
                    if key not in paths_by_start_kind:
                        paths_by_start_kind[key] = astronaut.get_paths(self.buildings, distance_matrix, paths_by_pair)
                    if paths_by_start_kind[key].size:
                        building.departing.astronauts.append(Astronaut(building.departing, astronaut.id, astronaut.kind, paths_by_start_kind[key]))

    def count_pods(self):
        """Counts how many pods serve each building.
        self receives updated num_pods values on buildings from pod service_edges."""
        for building in self.buildings.values():
            building.num_pods = 0
        for pod in self.pods:
            for building_id in {building_id for edge in pod.service_edges for building_id in edge}:
                self.buildings[building_id].num_pods += 1

    def process_teleport_phase(self, buildings: dict[int, Building]):
        """Moves eligible departing astronauts through teleporters.
        buildings stores simulation queues keyed by building id."""
        for building in buildings.values():
            outgoing_id = self.buildings[building.id].teleport[1]
            if outgoing_id == -1:
                continue
            remaining_astronauts = []
            for astronaut in building.departing.astronauts:
                if np.any(astronaut.paths[:, 1] == outgoing_id):
                    astronaut.move(buildings[outgoing_id].arriving)
                else:
                    remaining_astronauts.append(astronaut)
            building.departing.astronauts = remaining_astronauts

    def process_tube_phase(self, buildings: dict[int, Building], pods: list[Pod]):
        """Moves pods through tubes and boards astronauts that get closer to a target.
        buildings stores simulation queues keyed by building id. pods stores pod movement state ordered by pod id."""
        tube_uses = {}
        for pod in pods:
            if pod.dynamic:
                start_id = pod.path[-1] if pod.path else pod.get_next_stop(buildings)
                destination_id = pod.get_next_stop(buildings)
            else:
                start_id, destination_id = pod.get_building_ids()
            edge = self.make_edge(start_id, destination_id)
            if tube_uses.get(edge, 0) < self.buildings[edge[0]].tubes[edge[1]]:
                pod.seats = POD_CAPACITY
                remaining_astronauts = []
                for astronaut in buildings[start_id].departing.astronauts:
                    if pod.seats > 0 and np.any(astronaut.paths[:, 1] == destination_id):
                        astronaut.move(buildings[destination_id].arriving)
                        pod.seats -= 1
                        continue
                    remaining_astronauts.append(astronaut)
                buildings[start_id].departing.astronauts = remaining_astronauts
                pod.path_index = pod.next_path_index()
                tube_uses[edge] = tube_uses.get(edge, 0) + 1

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
    population stores the number of astronauts settled in this building during the current month. num_pods stores the number of pods stopping here.
    initial stores the monthly departing terminal template. arriving stores astronauts that reached the building during the current pod phase.
    departing stores astronauts available to board pods."""
    id: int
    kind: int
    coords: NDArray[int]
    tubes: dict[int, int] = field(default_factory=dict)
    teleport: tuple[int, int] = (-1, -1)
    population: int = 0
    num_pods: int = 0
    initial: Terminal = field(init=False)
    arriving: Terminal = field(init=False)
    departing: Terminal = field(init=False)


@dataclass(slots=True)
class Pod:
    """Stores one transport pod and its monthly movement state.
    id is the pod identifier. path stores the itinerary. path_index stores the current index in path. seats stores empty seats.
    service_edges stores tube edges served by this pod. dynamic selects whether the path is generated during simulation.
    distance_matrix stores tube-only shortest distances among service edge nodes for dynamic routing."""
    id: int
    path: list[int]
    path_index: int = 0
    seats: int = 0
    dynamic: bool = False
    service_edges: list[tuple[int, int]] = None
    distance_matrix: DistanceMatrix = None

    def get_next_stop(self, buildings: dict[int, Building]) -> int:
        """Chooses and appends this service pod next stop.
        buildings stores current building terminals and tube links. Returns the building id appended to path."""
        keys = []
        for start_id, end_id in self.service_edges:
            for source_id, destination_id in ((start_id, end_id), (end_id, start_id)):
                demand = sum(1 for astronaut in buildings[source_id].departing.astronauts if np.any(astronaut.paths[:, 1] == destination_id))
                demand = min(POD_CAPACITY, demand)
                if self.path:
                    keys.append((-demand, buildings[source_id].num_pods, self.distance_matrix[self.path[-1]][source_id], source_id, destination_id))
                else:
                    keys.append((-demand, buildings[source_id].num_pods, source_id, destination_id))
        source_id, destination_id = min(keys)[-2:]
        if not self.path:
            self.path.append(source_id)
        elif self.path[-1] == source_id:
            self.path.append(destination_id)
        else:
            current_id = self.path[-1]
            source_distance = self.distance_matrix[current_id][source_id]
            next_id = next(next_id for next_id in buildings[current_id].tubes
                if next_id in self.distance_matrix and self.distance_matrix[next_id][source_id] == source_distance - 1)
            self.path.append(next_id)
        return self.path[-1]

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

    def get_paths(self, buildings: dict[int, Building], distance_matrix: DistanceMatrix, paths_by_pair: PathsByPair) -> NDArray[int]:
        """Selects shortest paths from this astronaut building to the nearest matching modules.
        buildings stores candidate modules keyed by building id. distance_matrix gives shortest travel distances.
        paths_by_pair maps building pairs to all shortest paths between them. Returns a padded matrix with one path per row."""
        building_id = self.terminal.building.id
        target_ids = [building.id for building in buildings.values() if building.kind == self.kind and (building_id, building.id) in paths_by_pair]
        if not target_ids:
            return np.empty((0, 0), dtype=int)
        nearest_distance = min(distance_matrix[building_id][target_id] for target_id in target_ids)
        paths = []
        for target_id in target_ids:
            if distance_matrix[building_id][target_id] == nearest_distance:
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
        while np.all(self.paths[:, -1] == -1):
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
        state.fix_dynamic_pods()
        print(state.choose_action())


if __name__ == "__main__":
    play()
