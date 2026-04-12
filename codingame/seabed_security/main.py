"""Controls a Seabed Security bot with radar fish hunting and visible-monster avoidance."""

from dataclasses import dataclass, field
import sys

import numpy as np
import numpy.typing as npt


IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

FIELD_SIZE = 10000
DRONE_SPEED = 600
SCAN_RADIUS = 800
MONSTER_COLLISION_RADIUS = 500


@dataclass(slots=True)
class CreatureInfo:
    """Stores creature metadata from the initialization block.
    :var color: Color index reported by the server.
    :var kind: Type index reported by the server, or -1 for monsters.
    """
    color: int
    kind: int


@dataclass(slots=True)
class VisibleCreature:
    """Stores one visible creature snapshot.
    :var coords: Current creature coordinates.
    :var velocity: Current creature velocity vector.
    """
    coords: IntArray
    velocity: IntArray


@dataclass(slots=True)
class Drone:
    """Stores the subset of drone state used by the planner.
    :var drone_id: Unique drone identifier.
    :var coords: Current drone coordinates.
    :var emergency: Indicates whether the drone is already forced to surface.
    :var battery: Current battery charge.
    :var last_enabled: Most recent turn index when this drone used the strong light.
    :var scans: Fish scans currently carried by the drone.
    :var radar: Last radar blip direction for each relevant creature.
    """
    drone_id: int
    coords: IntArray
    emergency: bool
    battery: int
    last_enabled: int = -4
    scans: set[int] = field(default_factory=set)
    radar: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class GameState:
    """Stores one full turn snapshot.
    :var drones: Our drones keyed by id in the exact server order.
    :var visible_creatures: Current visible creature snapshots keyed by creature id.
    :var known_scans: Fish ids already scanned by us, whether saved or currently carried.
    """
    drones: dict[int, Drone]
    visible_creatures: dict[int, VisibleCreature]
    known_scans: set[int]


def main_loop():
    """Runs the game loop and prints one command per drone each turn."""
    creature_infos = read_initial_data()
    previous_drones = {}
    monster_ids = {creature_id for creature_id, info in creature_infos.items() if info.kind == -1}
    turn_number = 0
    while True:
        game_state = read_game_state()

        for drone_id, drone in game_state.drones.items():
            print(f"drone {drone.drone_id}: pos=({drone.coords[0]}, {drone.coords[1]})", file=sys.stderr)
        for creature_id, creature in game_state.visible_creatures.items():
            label = "monster" if creature_id in monster_ids else "fish"
            print(f"{label} {creature_id}: pos=({creature.coords[0]}, {creature.coords[1]}) angle={np.rad2deg(np.arctan2(-creature.velocity[1], creature.velocity[0]))}",
                  file=sys.stderr)

        for drone in game_state.drones.values():
            if turn_number > 0:
                drone.last_enabled = previous_drones[drone.drone_id].last_enabled
            x, y, light = choose_action(drone, monster_ids, game_state.visible_creatures, game_state.known_scans, turn_number)
            if light == 1:
                drone.last_enabled = turn_number
            print(f"MOVE {x} {y} {light}")
        previous_drones = game_state.drones
        turn_number += 1


def read_initial_data() -> dict[int, CreatureInfo]:
    """Reads creature metadata from the initialization block.
    :return: Creature metadata keyed by creature id.
    """
    creature_count = int(input())
    creatures = {}
    for _ in range(creature_count):
        creature_id, color, kind = map(int, input().split())
        creatures[creature_id] = CreatureInfo(color, kind)
    return creatures


def read_game_state() -> GameState:
    """Reads one full game turn.
    :return: Parsed state for the current turn.
    """
    _ = int(input())
    _ = int(input())
    my_scan_count = int(input())
    known_scans = set()
    for _ in range(my_scan_count):
        known_scans.add(int(input()))
    foe_scan_count = int(input())
    for _ in range(foe_scan_count):
        _ = int(input())

    my_drone_count = int(input())
    drones = {}
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = map(int, input().split())
        drones[drone_id] = Drone(drone_id, np.array((drone_x, drone_y)), bool(emergency), battery)

    foe_drone_count = int(input())
    for _ in range(foe_drone_count):
        input()

    drone_scan_count = int(input())
    for _ in range(drone_scan_count):
        drone_id, creature_id = map(int, input().split())
        drone = drones.get(drone_id)
        if drone is not None:
            drone.scans.add(creature_id)
            known_scans.add(creature_id)

    visible_creatures = {}
    visible_creature_count = int(input())
    for _ in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = map(int, input().split())
        visible_creatures[creature_id] = VisibleCreature(np.array((creature_x, creature_y)), np.array((creature_vx, creature_vy)))

    radar_blip_count = int(input())
    for _ in range(radar_blip_count):
        drone_id_str, creature_id_str, radar = input().split()
        drone_id, creature_id = int(drone_id_str), int(creature_id_str)
        drone = drones.get(drone_id)
        if creature_id not in known_scans and drone is not None:
            drone.radar[creature_id] = radar

    return GameState(drones, visible_creatures, known_scans)


def choose_action(drone: Drone, monster_ids: set[int], visible_creatures: dict[int, VisibleCreature], known_scans: set[int],
                  turn_number: int) -> tuple[int, int, int]:
    """Chooses one move target and light setting for a drone.
    :param drone: Drone state to plan for.
    :param monster_ids: Creature ids belonging to monsters.
    :param visible_creatures: Visible creature snapshots for the turn.
    :param known_scans: Fish ids already scanned by us, whether saved or currently carried.
    :param turn_number: Zero-based turn index.
    :return: Move x, move y, and light state.
    """
    monsters = [creature for creature_id, creature in visible_creatures.items() if creature_id in monster_ids]
    target = choose_base_target(drone, monster_ids, visible_creatures, known_scans)
    safe_target = choose_safe_target(drone, target, monsters)
    light = choose_light(drone, turn_number)
    return safe_target[0], safe_target[1], light


def choose_base_target(drone: Drone, monster_ids: set[int], visible_creatures: dict[int, VisibleCreature], known_scans: set[int]) -> IntArray:
    """Chooses the fish pursuit target before monster safety is considered.
    :param drone: Drone state to plan for.
    :param monster_ids: Creature ids belonging to monsters.
    :param visible_creatures: Visible creature snapshots for the turn.
    :param known_scans: Fish ids already scanned by us, whether saved or currently carried.
    :return: Desired target point when only fish collection is considered.
    """
    fish_targets = {}
    for creature_id, creature in visible_creatures.items():
        if creature_id not in monster_ids and creature_id not in known_scans:
            fish_targets[creature_id] = creature.coords
    for creature_id, radar in drone.radar.items():
        if creature_id not in monster_ids:
            fish_targets.setdefault(creature_id, guess_creature_coords(drone.coords, radar))
    if not fish_targets:
        return np.array((drone.coords[0], 0))
    return min(fish_targets.values(), key=lambda coords: np.linalg.norm(drone.coords - coords))


def guess_creature_coords(drone_coords: IntArray, radar: str) -> IntArray:
    """Guesses one creature position from a single radar quadrant.
    :param drone_coords: Current drone coordinates.
    :param radar: Radar quadrant reported by the server.
    :return: Midpoint guess inside the indicated map quadrant.
    """
    match radar:
        case "TL":
            return np.array((drone_coords[0] // 2, drone_coords[1] // 2))
        case "TR":
            return np.array(((drone_coords[0] + FIELD_SIZE) // 2, drone_coords[1] // 2))
        case "BL":
            return np.array((drone_coords[0] // 2, (drone_coords[1] + FIELD_SIZE) // 2))
        case "BR":
            return np.array(((drone_coords[0] + FIELD_SIZE) // 2, (drone_coords[1] + FIELD_SIZE) // 2))
        case _:
            raise AssertionError(f"Unexpected radar value: {radar}")


def choose_safe_target(drone: Drone, target: IntArray, monsters: list[VisibleCreature]) -> IntArray:
    """Chooses the nearest-angle collision-free target for the turn.
    :param drone: Drone state to plan for.
    :param target: Current target that the safety layer should try to preserve.
    :param monsters: Visible monster snapshots for the turn.
    :return: Safe target point.
    """
    direction = target - drone.coords
    for angle_degrees in range(181):
        for sign in (-1, 1):
            if angle_degrees == 0 and sign == -1:
                continue
            angle_radians = np.deg2rad(sign * angle_degrees)
            rotation_matrix = np.array(((np.cos(angle_radians), -np.sin(angle_radians)), (np.sin(angle_radians), np.cos(angle_radians))))
            rotated_target = drone.coords + rotation_matrix @ direction
            drone_end = get_end_point(drone.coords, rotated_target, DRONE_SPEED)
            if not any(minimum_distance_between_paths(drone.coords, drone_end - drone.coords, monster.coords, monster.velocity) < MONSTER_COLLISION_RADIUS
                       for monster in monsters):

                base_angle = np.rad2deg(np.arctan2(-direction[1], direction[0]))
                corrected_direction = drone_end - drone.coords
                final_angle = np.rad2deg(np.arctan2(-corrected_direction[1], corrected_direction[0]))
                print(f"drone {drone.drone_id}: angle1={base_angle} angle2={final_angle}", file=sys.stderr)

                return np.rint(rotated_target).astype(np.int64)
    return np.array((drone.coords[0], 0))


def minimum_distance_between_paths(start_a: IntArray, velocity_a: IntArray, start_b: IntArray, velocity_b: IntArray) -> float:
    """Computes the minimum distance between two linear movements during one turn.
    :param start_a: First path starting point.
    :param velocity_a: First path velocity vector for the turn.
    :param start_b: Second path starting point.
    :param velocity_b: Second path velocity vector for the turn.
    :return: Minimum distance reached between the two moving points during the turn.
    """
    relative = start_a - start_b
    relative_velocity = velocity_a - velocity_b
    relative_speed_sq = relative_velocity @ relative_velocity
    if relative_speed_sq == 0:
        return np.linalg.norm(relative)
    time = max(0, min(1, -(relative @ relative_velocity) / relative_speed_sq))
    return np.linalg.norm(relative + relative_velocity * time)


def choose_light(drone: Drone, turn_number: int) -> int:
    """Chooses the light setting from the turn timer.
    :param drone: Drone state to plan for.
    :param turn_number: Zero-based turn index.
    :return: Light setting for the move.
    """
    return int(turn_number - drone.last_enabled >= 4)


def get_end_point(start_point: IntArray, target_point: FloatArray | IntArray, speed: int) -> IntArray:
    """Gets the end point after moving from one point toward another with the game rounding rules.
    :param start_point: Starting coordinates.
    :param target_point: Desired target coordinates.
    :param speed: Maximum movement distance.
    :return: End coordinates after moving toward the target point.
    """
    direction = target_point - start_point
    distance_to_target = np.linalg.norm(direction)
    if distance_to_target <= speed:
        return np.rint(target_point).astype(np.int64)
    return start_point + np.rint(direction * (speed / distance_to_target)).astype(np.int64)


main_loop()
