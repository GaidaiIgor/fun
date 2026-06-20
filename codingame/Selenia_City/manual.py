"""Reads Selenia City turns and emits planned infrastructure actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from sys import stderr

import numpy as np
from numpy import linalg
from numpy.typing import NDArray

MONTH_DAYS = 20
POD_CAPACITY = 10
POD_COST = 1000
POD_REROUTE_COST = 250
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
        New buildings are stored in self.buildings. Each building receives an initial terminal, populated only for landing pads."""
        for _ in range(int(input())):
            values = list(map(int, input().split()))
            building = Building(values[1], values[0], np.array((values[2], values[3]), dtype=int))
            building.initial = Terminal(building)
            self.buildings[building.id] = building
            if building.kind == 0:
                for index, astronaut_type in enumerate(values[5:]):
                    building.initial.astronauts.append(Astronaut(building.initial, 1000 * building.id + index, astronaut_type, np.empty((0, 0), dtype=int)))

    def fix_dynamic_pods(self):
        """Marks all pods in the current snapshot as no longer dynamically routed.
        self provides pods that existed before this month decision phase starts."""
        for pod in self.pods:
            pod.dynamic = False

    def print(self):
        """Prints the current month-start game state to stderr.
        self provides resources, buildings, tubes, pods, and landing-pad astronaut arrivals included in the snapshot."""
        print(f"MONTH {self.month}", file=stderr)
        print(f"RESOURCES {self.resources}", file=stderr)
        print(f"BUILDINGS {len(self.buildings)}", file=stderr)
        for building in self.buildings.values():
            building.print()
        print(f"PODS {len(self.pods)}", file=stderr)
        for pod in self.pods:
            pod.print()

    def choose_action(self) -> str:
        """Builds missing transport infrastructure for the current month.
        Returns a valid action string containing planned edge, pod, reroute, and upgrade commands."""
        actions = self.develop_edges()
        actions.extend(self.develop_pods())
        upgrade_actions, simulation_result = self.upgrade_routes()
        actions.extend(upgrade_actions)
        print(f"MONTHLY_SCORE {simulation_result.score}", file=stderr)
        return ";".join(actions) if actions else "WAIT"

    def develop_edges(self) -> list[str]:
        """Makes decisions regarding edge development.
        Returns tube action strings for the edges added to self."""
        actions = []
        while True:
            components = self.get_components()
            if len(components) == 1:
                break
            connection = self.find_best_connection(components)
            if connection is None:
                break
            source_building, target_building, cost = connection
            if cost > self.resources:
                break
            self.build_tube(source_building, target_building)
            actions.append(f"TUBE {source_building.id} {target_building.id}")
        return actions

    def get_components(self) -> list[list[Building]]:
        """Finds connected components in the current tube graph.
        Returns a list of components, where each component contains the buildings connected by existing tubes."""
        components = []
        visited_ids = set()
        for building in self.buildings.values():
            if building.id in visited_ids:
                continue
            component = []
            pending_ids = {building.id}
            visited_ids.add(building.id)
            while pending_ids:
                current = self.buildings[pending_ids.pop()]
                component.append(current)
                for neighbor_id in current.tubes:
                    if neighbor_id not in visited_ids:
                        visited_ids.add(neighbor_id)
                        pending_ids.add(neighbor_id)
            components.append(component)
        return components

    def find_best_connection(self, components: list[list[Building]]) -> tuple[Building, Building, int]:
        """Finds the cheapest buildable tube between two disconnected components.
        components stores disconnected building groups. Returns the selected endpoint buildings and tube cost, or None when no edge can be built."""
        best_connection = None
        best_cost = 0
        for component_index, component in enumerate(components):
            for other_component in components[component_index + 1:]:
                for start in component:
                    for end in other_component:
                        if not self.can_build_tube(start, end):
                            continue
                        cost = int(linalg.norm(start.coords - end.coords) * 10)
                        if best_connection is None or cost < best_cost:
                            best_connection = start, end, cost
                            best_cost = cost
        return best_connection

    def can_build_tube(self, start: Building, end: Building) -> bool:
        """Checks whether a tube can be built without crossing tubes or passing through buildings.
        start and end identify the proposed tube endpoints. Returns whether the proposed segment avoids invalid intersections."""
        for tube in self.get_tubes():
            if self.do_tubes_intersect(start, end, tube.start, tube.end):
                return False
        for building in self.buildings.values():
            if building.id not in (start.id, end.id) and self.is_building_on_segment(building, start, end):
                return False
        return True

    @staticmethod
    def do_tubes_intersect(start: Building, end: Building, other_start: Building, other_end: Building) -> bool:
        """Checks whether two tube segments cross away from their endpoints.
        start and end identify the proposed tube. other_start and other_end identify an existing tube.
        Returns whether the two segments strictly cross, excluding collinear contact and endpoint contact."""
        return GameState.get_orientation(start, end, other_start) * GameState.get_orientation(start, end, other_end) < 0 \
            and GameState.get_orientation(other_start, other_end, start) * GameState.get_orientation(other_start, other_end, end) < 0

    @staticmethod
    def get_orientation(start: Building, end: Building, point: Building) -> int:
        """Calculates the side of a directed segment where a point lies.
        start and end identify the directed segment. point identifies the tested building.
        Returns 1 or -1 for opposite sides of the directed segment, and 0 when the three buildings are collinear."""
        product = (point.coords[1] - start.coords[1]) * (end.coords[0] - start.coords[0]) \
                  - (end.coords[1] - start.coords[1]) * (point.coords[0] - start.coords[0])
        return np.sign(product).item()

    @staticmethod
    def is_building_on_segment(building: Building, start: Building, end: Building) -> bool:
        """Checks whether a building lies on a tube segment.
        building identifies the tested building. start and end identify the segment endpoints.
        Returns whether the sum of endpoint-to-building distances equals the segment length within epsilon."""
        distance_delta = linalg.norm(start.coords - building.coords) + linalg.norm(building.coords - end.coords) - linalg.norm(start.coords - end.coords)
        return abs(distance_delta) < 1e-7

    def build_tube(self, start: Building, end: Building):
        """Adds a planned tube edge to the current game snapshot.
        start and end identify the buildings connected by the tube."""
        tube = Tube(start, end, 1)
        start.tubes[end.id] = tube
        end.tubes[start.id] = tube
        self.resources -= int(linalg.norm(start.coords - end.coords) * 10)

    def develop_pods(self) -> list[str]:
        """Makes decisions regarding pod development.
        Returns pod action strings generated for new and rerouted pods."""
        all_edges = {(tube.start.id, tube.end.id) for tube in self.get_tubes()}
        new_pods = []
        rerouted_pods = set()
        if not self.pods:
            if all_edges and self.resources >= POD_COST:
                new_pods.append(self.build_pod(all_edges))
        if self.pods:
            serviced_edges = {edge for pod in self.pods for edge in pod.service_edges}
            unserviced_edges = {edge for edge in all_edges if edge not in serviced_edges}
            rerouted_pods = self.assign_unserviced_edges(unserviced_edges)
            split_pods = self.reduce_pod_load()
            new_pods.extend(split_pods[0])
            rerouted_pods.update(split_pods[1])
        if not new_pods and not rerouted_pods:
            return []
        self.simulate_month()
        actions = [" ".join(["POD", str(pod.id), *(str(building_id) for building_id in pod.path)]) for pod in new_pods]
        for pod in rerouted_pods:
            actions.append(f"DESTROY {pod.id}")
            actions.append(" ".join(["POD", str(pod.id), *(str(building_id) for building_id in pod.path)]))
        return actions

    def assign_unserviced_edges(self, unserviced_edges: set[tuple[int, int]]) -> set[Pod]:
        """Assigns unserviced edges to adjacent existing pods when rerouting is affordable.
        unserviced_edges stores canonical tube edges still without service and is mutated by removing assigned edges.
        Returns pods that must be rebuilt after receiving new service edges."""
        rerouted_pods = set()
        while unserviced_edges:
            assignment = self.find_unserviced_edge_pod(unserviced_edges)
            if assignment is None:
                break
            if not assignment[1].dynamic:
                rerouted_pods.add(assignment[1])
            self.add_service_edge(assignment[0], assignment[1])
            unserviced_edges.remove(assignment[0])
        return rerouted_pods

    def find_unserviced_edge_pod(self, unserviced_edges: set[tuple[int, int]]) -> tuple[tuple[int, int], Pod]:
        """Finds an unserviced edge and an adjacent pod that can serve it.
        unserviced_edges stores canonical tube edges still without service. Returns the selected edge and pod, or None when no usable adjacent pod exists."""
        for edge in unserviced_edges:
            available_pods = self.buildings[edge[0]].pods | self.buildings[edge[1]].pods
            dynamic_pods = {pod for pod in available_pods if pod.dynamic}
            if self.resources < POD_REROUTE_COST:
                available_pods = dynamic_pods
            if available_pods:
                return edge, min(dynamic_pods or available_pods, key=lambda pod: len(pod.service_edges))
        return None

    def add_service_edge(self, edge: tuple[int, int], pod: Pod):
        """Adds a service edge to a pod and updates all dependent planning state.
        edge is the canonical tube edge being assigned. pod is the pod receiving the edge."""
        if not pod.dynamic:
            self.resources -= POD_REROUTE_COST
            pod.dynamic = True
        pod.service_edges.add(edge)
        pod.path = []
        for building_id in edge:
            self.buildings[building_id].pods.add(pod)

    def reduce_pod_load(self) -> tuple[list[Pod], set[Pod]]:
        """Tries to build new pods or reroute existing pods to alleviate highest workload.
        Returns new pods and original pods that need rebuild actions."""
        new_pods = []
        rerouted_pods = set()
        while True:
            pod = max(self.pods, key=lambda pod: len(pod.service_edges))
            if len(pod.service_edges) == 1:
                break
            overlap_found, overlap_removed, pod_rerouted = self.remove_overlapping_edges(pod)
            if overlap_found:
                if not overlap_removed:
                    break
                if pod_rerouted:
                    rerouted_pods.add(pod)
                continue
            if self.resources < POD_COST:
                break
            new_service_area = self.split_service_area(pod)
            new_pods.append(self.build_pod(new_service_area))
            if not pod.dynamic and self.resources < POD_REROUTE_COST:
                break
            if not pod.dynamic:
                rerouted_pods.add(pod)
            for edge in new_service_area:
                self.remove_service_edge(edge, pod)
        return new_pods, rerouted_pods

    def remove_overlapping_edges(self, pod: Pod) -> tuple[bool, bool, bool]:
        """Removes service edges from a pod when those edges are already served by another pod.
        pod is the pod whose service area is checked.
        Returns whether overlapping edges existed, whether they were removed, and whether pod was rerouted."""
        overlapped_edges = set()
        for other_pod in self.pods:
            if other_pod is not pod:
                overlapped_edges.update(pod.service_edges & other_pod.service_edges)
        if not overlapped_edges:
            return False, False, False
        if not pod.dynamic and self.resources < POD_REROUTE_COST:
            return True, False, False
        pod_rerouted = not pod.dynamic
        for edge in overlapped_edges:
            self.remove_service_edge(edge, pod)
        return True, True, pod_rerouted

    def remove_service_edge(self, edge: tuple[int, int], pod: Pod):
        """Removes a service edge from a pod and updates all dependent planning state.
        edge is the canonical tube edge being removed. pod is the pod losing the edge."""
        if not pod.dynamic:
            self.resources -= POD_REROUTE_COST
            pod.dynamic = True
        pod.service_edges.remove(edge)
        pod.path = []
        for building_id in edge:
            if all(building_id not in service_edge for service_edge in pod.service_edges):
                self.buildings[building_id].pods.remove(pod)

    def split_service_area(self, pod: Pod) -> set[tuple[int, int]]:
        """Splits a connected pod service area into two service areas.
        pod gives the existing pod being split. Returns the connected service edge set assigned to a new pod after assigning the side with
        pod first path edge to pod."""
        target_size = len(pod.service_edges) // 2
        for edge in pod.service_edges:
            remaining_area = pod.service_edges - {edge}
            if not self.is_service_area_connected(remaining_area):
                continue
            moving_area = {edge}
            moving_nodes = set(edge)
            break
        while len(moving_area) < target_size:
            for edge in remaining_area:
                if edge[0] not in moving_nodes and edge[1] not in moving_nodes:
                    continue
                candidate_remaining_area = remaining_area - {edge}
                if not self.is_service_area_connected(candidate_remaining_area):
                    continue
                moving_area.add(edge)
                moving_nodes.update(edge)
                remaining_area = candidate_remaining_area
                break
            else:
                break
        if pod.path:
            first_path_edge = Tube.get_standard_order(pod.path[0], pod.path[1])
            return remaining_area if first_path_edge in moving_area else moving_area
        return moving_area

    @staticmethod
    def is_service_area_connected(edges: set[tuple[int, int]]) -> bool:
        """Checks whether service edges form one connected area.
        edges gives canonical tube edges. Returns whether all edges belong to one connected component."""
        edge = next(iter(edges))
        connected_edges = {edge}
        connected_nodes = set(edge)
        while len(connected_edges) < len(edges):
            for edge in edges:
                if edge not in connected_edges and (edge[0] in connected_nodes or edge[1] in connected_nodes):
                    connected_edges.add(edge)
                    connected_nodes.update(edge)
                    break
            else:
                return False
        return True

    def upgrade_routes(self) -> tuple[list[str], SimulationSummary]:
        """Repeatedly upgrades the highest-demand affordable tube edge and adds a pod to it.
        Returns upgrade and pod creation action strings plus the final monthly simulation result after no more upgrade is affordable."""
        actions = []
        while True:
            simulation_result = self.simulate_month()
            edges = list(simulation_result.tube_demands)
            lengths = np.array([linalg.norm(self.buildings[edge[0]].coords - self.buildings[edge[1]].coords) for edge in edges])
            capacities = np.array([self.buildings[edge[0]].tubes[edge[1]].capacity for edge in edges], dtype=int)
            demands = np.array([simulation_result.tube_demands[edge] for edge in edges], dtype=int)
            upgrade_costs = (lengths * 10).astype(int) * (capacities + 1)
            candidates = (upgrade_costs + POD_COST <= self.resources) & (demands > 0)
            if not np.any(candidates):
                return actions, simulation_result
            best_index = int(np.argmax(np.where(candidates, demands, -1)))
            best_edge = edges[best_index]
            best_upgrade_cost = int(upgrade_costs[best_index])
            self.resources -= best_upgrade_cost
            self.buildings[best_edge[0]].tubes[best_edge[1]].capacity += 1
            pod = self.build_pod({best_edge})
            pod.path = [best_edge[index % 2] for index in range(MONTH_DAYS + 1)]
            actions.append(f"UPGRADE {best_edge[0]} {best_edge[1]}")
            actions.append(" ".join(["POD", str(pod.id), *(str(building_id) for building_id in pod.path)]))

    def build_pod(self, service_edges: set[tuple[int, int]]) -> Pod:
        """Creates a dynamic pod serving a continuous area.
        service_edges gives canonical tube edges served by the new pod. Returns the pod added to self."""
        pod = Pod(max((pod.id for pod in self.pods), default=0) + 1, [], dynamic=True, service_edges=service_edges)
        self.pods.append(pod)
        for building_id in {building_id for edge in service_edges for building_id in edge}:
            self.buildings[building_id].pods.add(pod)
        self.resources -= POD_COST
        return pod

    def simulate_month(self) -> SimulationSummary:
        """Simulates astronaut movement before the current month ends.
        Returns a SimulationSummary containing score and cumulative tube demand."""
        self.update_paths()
        self.reset_all()
        tubes = self.get_tubes()
        simulation_summary = SimulationSummary()
        for day in range(MONTH_DAYS):
            self.process_teleport_phase(self.buildings)
            simulation_summary.score += self.settle_arrivals(day, self.buildings)
            self.process_tube_phase(self.buildings, self.pods, tubes, simulation_summary)
            simulation_summary.score += self.settle_arrivals(day + 1, self.buildings)
        return simulation_summary

    def update_paths(self):
        """Prepares path and dynamic pod routing data before planning or movement simulation.
        self receives freshly calculated astronaut paths and dynamic pod distances."""
        distance_matrix, paths_by_pair = self.get_shortest_distances_paths(set(self.buildings))
        self.build_astronaut_paths(distance_matrix, paths_by_pair)
        self.prepare_pods()

    def build_astronaut_paths(self, distance_matrix: DistanceMatrix, paths_by_pair: PathsByPair):
        """Builds monthly path matrices for astronauts waiting at landing pads.
        distance_matrix and paths_by_pair provide shortest paths. self receives path matrices stored on initial landing-pad astronauts."""
        paths_by_start_kind = {}
        for building in self.buildings.values():
            if building.kind == 0:
                for astronaut in building.initial.astronauts:
                    key = building.id, astronaut.kind
                    if key not in paths_by_start_kind:
                        paths_by_start_kind[key] = astronaut.get_paths(self.buildings, distance_matrix, paths_by_pair)
                    astronaut.paths = paths_by_start_kind[key]

    def prepare_pods(self):
        """Prepares dynamic pod routing data before movement simulation.
        self provides dynamic pods. Each dynamic pod receives tube-only shortest distances among its service edge nodes."""
        for pod in self.pods:
            if pod.dynamic:
                service_nodes = {building_id for edge in pod.service_edges for building_id in edge}
                pod.distance_matrix, _ = self.get_shortest_distances_paths(service_nodes, use_teleports=False, return_paths=False)

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

    def reset_all(self):
        """Restores state to the beginning of the current month after simulation has temporarily advanced it.
        self receives fresh terminals, reset pod positions, and restored live astronauts while keeping generated pod paths."""
        self.reset_buildings()
        self.reset_pods()
        self.populate_departing_terminals()

    def reset_buildings(self):
        """Resets live building state to the beginning of the current month.
        self provides buildings whose population and terminals are refreshed."""
        for building in self.buildings.values():
            building.population = 0
            building.arriving = Terminal(building)
            building.departing = Terminal(building)

    def reset_pods(self):
        """Resets pod movement state to the beginning of the current month.
        self provides pods whose path index is reset."""
        for pod in self.pods:
            pod.path_index = 0

    def populate_departing_terminals(self):
        """Repopulates live departing terminals from prepared landing-pad astronauts.
        self provides initial landing-pad astronauts with path matrices. Buildings receive live astronauts that have at least one path."""
        for building in self.buildings.values():
            if building.kind == 0:
                for astronaut in building.initial.astronauts:
                    if astronaut.paths.size:
                        building.departing.astronauts.append(Astronaut(building.departing, astronaut.id, astronaut.kind, astronaut.paths))

    def get_tubes(self) -> list[Tube]:
        """Collects tubes stored in the current game snapshot.
        Returns each tube once."""
        tubes = []
        for building in self.buildings.values():
            for end_id, tube in building.tubes.items():
                if building.id < end_id:
                    tubes.append(tube)
        return tubes

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

    def process_tube_phase(self, buildings: dict[int, Building], pods: list[Pod], tubes: list[Tube], simulation_summary: SimulationSummary):
        """Moves pods through tubes and boards astronauts that get closer to a target.
        buildings stores simulation queues keyed by building id. pods stores pod movement state ordered by pod id. tubes stores tube objects.
        simulation_summary receives cumulative demand updates."""
        self.update_tube_demands(buildings, tubes, simulation_summary)
        tube_uses = {}
        for pod in pods:
            if pod.dynamic:
                start_id = pod.path[-1] if pod.path else pod.get_next_stop(buildings)
                destination_id = pod.get_next_stop(buildings)
            else:
                start_id, destination_id = pod.get_building_ids()
            edge = Tube.get_standard_order(start_id, destination_id)
            if tube_uses.get(edge, 0) < self.buildings[edge[0]].tubes[edge[1]].capacity:
                pod.seats = POD_CAPACITY
                remaining_astronauts = []
                for astronaut in buildings[start_id].departing.astronauts:
                    if pod.seats > 0 and np.any(astronaut.paths[:, 1] == destination_id):
                        astronaut.move(buildings[destination_id].arriving)
                        pod.seats -= 1
                        continue
                    remaining_astronauts.append(astronaut)
                buildings[start_id].departing.astronauts = remaining_astronauts
                pod.path_index = pod.get_next_path_index()
                tube_uses[edge] = tube_uses.get(edge, 0) + 1

    @staticmethod
    def update_tube_demands(buildings: dict[int, Building], tubes: list[Tube], simulation_summary: SimulationSummary):
        """Updates tube passenger demand for the current simulation day.
        buildings stores departure queues keyed by building id. tubes stores current tube objects. simulation_summary receives cumulative demand updates."""
        for tube in tubes:
            tube.demand = {tube.start.id: 0, tube.end.id: 0}
        for building in buildings.values():
            for astronaut in building.departing.astronauts:
                for destination_id in set(astronaut.paths[:, 1].tolist()):
                    building.tubes[destination_id].demand[building.id] += 1
        for tube in tubes:
            edge = tube.start.id, tube.end.id
            simulation_summary.tube_demands[edge] = simulation_summary.tube_demands.get(edge, 0) + tube.demand[tube.start.id] + tube.demand[tube.end.id]

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
    position. tubes stores adjacent tubes keyed by neighbor id. teleport stores incoming and outgoing building ids.
    population stores the number of astronauts settled in this building during the current month. pods stores pods stopping here.
    initial stores the monthly departing terminal template. arriving stores astronauts that reached the building during the current pod phase.
    departing stores astronauts available to board pods."""
    id: int
    kind: int
    coords: NDArray[int]
    tubes: dict[int, Tube] = field(default_factory=dict)
    teleport: tuple[int, int] = (-1, -1)
    population: int = 0
    pods: set[Pod] = field(default_factory=set)
    initial: Terminal = field(init=False)
    arriving: Terminal = field(init=False)
    departing: Terminal = field(init=False)

    def print(self):
        """Prints this building month-start debug state to stderr.
        self provides id, kind, coordinates, teleport links, tube links, pod membership, and landing-pad astronauts."""
        tube_text = " ".join(f"{end_id}:{tube.capacity}" for end_id, tube in self.tubes.items()) or "-"
        teleport_text = f"{self.teleport[0]}:{self.teleport[1]}"
        pod_text = " ".join(str(pod.id) for pod in self.pods) or "-"
        astronaut_text = " ".join(str(astronaut.kind) for astronaut in self.initial.astronauts) or "-"
        print(f"BUILDING {self.id}, kind={self.kind}, coords={self.coords.tolist()}, tubes={tube_text}, teleport={teleport_text}, pods={pod_text}, "
              f"astronauts={astronaut_text}", file=stderr)


@dataclass(slots=True)
class Tube:
    """Stores one magnetic tube.
    start and end store the canonical endpoint buildings. capacity stores how many pods can use the tube per day.
    demand stores current directed passenger demand keyed by source building id."""
    start: Building
    end: Building
    capacity: int
    demand: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        """Stores endpoint buildings in canonical order after construction."""
        if self.end.id < self.start.id:
            self.start, self.end = self.end, self.start

    @staticmethod
    def get_standard_order(start_id: int, end_id: int) -> tuple[int, int]:
        """Orders two building ids into a canonical tube edge.
        start_id and end_id identify the tube endpoints. Returns the canonical endpoint pair."""
        return (start_id, end_id) if start_id < end_id else (end_id, start_id)


@dataclass(slots=True, eq=False)
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
    service_edges: set[tuple[int, int]] = None
    distance_matrix: DistanceMatrix = None

    def print(self):
        """Prints this pod month-start debug state to stderr.
        self provides id, dynamic flag, path state, and service edges."""
        path_text = " ".join(str(building_id) for building_id in self.path)
        edge_text = " ".join(f"{edge[0]}-{edge[1]}" for edge in self.service_edges)
        print(f"POD {self.id}, path={path_text}, service_edges={edge_text}", file=stderr)

    def get_next_stop(self, buildings: dict[int, Building]) -> int:
        """Chooses and appends this service pod next stop.
        buildings stores current building terminals and tube links. Returns the building id appended to path."""
        keys = []
        for start_id, end_id in self.service_edges:
            tube = buildings[start_id].tubes[end_id]
            for source_id, destination_id in ((start_id, end_id), (end_id, start_id)):
                demand = min(POD_CAPACITY, tube.demand[source_id])
                if self.path:
                    keys.append((-demand, len(buildings[source_id].pods), self.distance_matrix[self.path[-1]][source_id], source_id, destination_id))
                else:
                    keys.append((-demand, len(buildings[source_id].pods), source_id, destination_id))
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
        next_index = self.get_next_path_index()
        return self.path[self.path_index], self.path[next_index]

    def get_next_path_index(self) -> int:
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
    score is the total score earned during the month. tube_demands maps canonical tube edges to cumulative demand."""
    score: int = 0
    tube_demands: dict[tuple[int, int], int] = field(default_factory=dict)


def play():
    """Runs the interactive game loop."""
    state = GameState()
    while True:
        state.read_month_input()
        state.fix_dynamic_pods()
        state.print()
        print(state.choose_action())
        print(f"REMAINING_RESOURCES {state.resources}", file=stderr)


if __name__ == "__main__":
    play()
