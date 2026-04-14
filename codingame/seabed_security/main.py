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
MAX_FISH_SPEED = 400
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
    :var preferred_side: Preferred horizontal direction, either "left" or "right".
    :var scans: Fish scans currently carried by the drone.
    :var radar: Last radar blip direction for each relevant creature.
    """
    drone_id: int
    coords: IntArray
    emergency: bool
    battery: int
    last_enabled: int = -4
    preferred_side: str = "right"
    scans: set[int] = field(default_factory=set)
    radar: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class GameState:
    """Stores one full turn snapshot.
    :var turn_number: Zero-based turn index.
    :var my_score: Our current score.
    :var foe_score: Enemy current score.
    :var my_saved_scans: Fish ids already saved by us.
    :var my_known_scans: Fish ids already scanned by us, whether saved or currently carried.
    :var foe_saved_scans: Fish ids already saved by the enemy.
    :var foe_known_scans: Fish ids already scanned by the enemy, whether saved or currently carried.
    :var creature_ids: Creature ids still present in the game zone this turn.
    :var drones: Our drones keyed by id in the exact server order.
    :var foe_drones: Enemy drones keyed by id in the exact server order.
    :var visible_creatures: Current visible creature snapshots keyed by creature id.
    :var fish_regions: Estimated fish rectangles as min_x, max_x, min_y, max_y arrays keyed by creature id.
    """
    turn_number: int
    my_score: int
    foe_score: int
    my_saved_scans: set[int]
    my_known_scans: set[int]
    foe_saved_scans: set[int]
    foe_known_scans: set[int]
    creature_ids: set[int]
    drones: dict[int, Drone]
    foe_drones: dict[int, Drone]
    visible_creatures: dict[int, VisibleCreature]
    fish_regions: dict[int, IntArray] = field(default_factory=dict)


def main_loop():
    """Runs the game loop and prints one command per drone each turn."""
    creature_infos = read_initial_data()
    previous_game_state = None
    while True:
        game_state = update_game_state(read_game_state(), creature_infos, previous_game_state)
        my_score_after_cashout = \
            game_state.my_score + calculate_score_gain(creature_infos, game_state.my_known_scans, game_state.my_saved_scans, game_state.foe_saved_scans)
        foe_best_case_known = game_state.creature_ids | game_state.foe_known_scans
        foe_max_score = game_state.foe_score + calculate_score_gain(creature_infos, foe_best_case_known, game_state.foe_saved_scans, game_state.my_known_scans)
        withdraw_now = my_score_after_cashout > foe_max_score

        for drone in game_state.drones.values():
            print(f"drone {drone.drone_id}: pos=({drone.coords[0]}, {drone.coords[1]})", file=sys.stderr)
        for foe_drone in game_state.foe_drones.values():
            print(f"foe drone {foe_drone.drone_id}: pos=({foe_drone.coords[0]}, {foe_drone.coords[1]})", file=sys.stderr)
        for creature_id, creature in game_state.visible_creatures.items():
            label = "monster" if creature_infos[creature_id].kind == -1 else "fish"
            print(f"{label} {creature_id}: pos=({creature.coords[0]}, {creature.coords[1]}), "
                f"angle={np.rad2deg(np.arctan2(-creature.velocity[1], creature.velocity[0])):.0f}", file=sys.stderr)
        print(f"Cashout: my={my_score_after_cashout}, foe_max={foe_max_score}", file=sys.stderr)

        for drone in game_state.drones.values():
            x, y, light = choose_action(drone, creature_infos, game_state, withdraw_now)
            print(f"MOVE {x} {y} {light}")
        previous_game_state = game_state


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
    my_score = int(input())
    foe_score = int(input())
    my_scan_count = int(input())
    my_saved_scans = set()
    for _ in range(my_scan_count):
        my_saved_scans.add(int(input()))
    foe_scan_count = int(input())
    foe_saved_scans = set()
    for _ in range(foe_scan_count):
        foe_saved_scans.add(int(input()))

    my_drone_count = int(input())
    my_drones = {}
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = map(int, input().split())
        my_drones[drone_id] = Drone(drone_id, np.array((drone_x, drone_y)), bool(emergency), battery)

    foe_drone_count = int(input())
    foe_drones = {}
    for _ in range(foe_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = map(int, input().split())
        foe_drones[drone_id] = Drone(drone_id, np.array((drone_x, drone_y)), bool(emergency), battery)

    drone_scan_count = int(input())
    my_known_scans = set(my_saved_scans)
    foe_known_scans = set(foe_saved_scans)
    for _ in range(drone_scan_count):
        drone_id, creature_id = map(int, input().split())
        if drone_id in my_drones:
            my_drones[drone_id].scans.add(creature_id)
            my_known_scans.add(creature_id)
        else:
            foe_drones[drone_id].scans.add(creature_id)
            foe_known_scans.add(creature_id)

    visible_creatures = {}
    visible_creature_count = int(input())
    for _ in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = map(int, input().split())
        visible_creatures[creature_id] = VisibleCreature(np.array((creature_x, creature_y)), np.array((creature_vx, creature_vy)))

    existing_creature_ids = set()
    radar_blip_count = int(input())
    for _ in range(radar_blip_count):
        drone_id_str, creature_id_str, radar_location = input().split()
        drone_id, creature_id = int(drone_id_str), int(creature_id_str)
        existing_creature_ids.add(creature_id)
        my_drones[drone_id].radar[creature_id] = radar_location

    return GameState(0, my_score, foe_score, my_saved_scans, my_known_scans, foe_saved_scans, foe_known_scans, existing_creature_ids, my_drones,
                     foe_drones, visible_creatures)


def update_game_state(game_state: GameState, creature_infos: dict[int, CreatureInfo], previous_game_state: GameState | None) -> GameState:
    """Updates one parsed game state with data carried over from the previous turn.
    :param game_state: Parsed state for the current turn.
    :param creature_infos: Creature metadata keyed by creature id.
    :param previous_game_state: Previous turn state, if any.
    :return: Updated game state.
    """
    if previous_game_state is None:
        game_state.turn_number = 0
        left_drone, right_drone = sorted(game_state.drones.values(), key=lambda drone: drone.coords[0])
        left_drone.preferred_side = "left"
        right_drone.preferred_side = "right"
    else:
        game_state.turn_number = previous_game_state.turn_number + 1
        for drone in game_state.drones.values():
            drone.last_enabled = previous_game_state.drones[drone.drone_id].last_enabled
            drone.preferred_side = previous_game_state.drones[drone.drone_id].preferred_side
    game_state.fish_regions = update_fish_regions(creature_infos, game_state, previous_game_state)
    return game_state


def update_fish_regions(creature_infos: dict[int, CreatureInfo], game_state: GameState, previous_game_state: GameState | None) -> dict[int, IntArray]:
    """Updates persistent fish rectangles from current radar, visibility, and the previous turn estimate.
    :param creature_infos: Creature metadata keyed by creature id.
    :param game_state: Parsed state for the current turn.
    :param previous_game_state: Previous turn state, if any.
    :return: Updated estimated fish rectangles keyed by creature id.
    """
    fish_regions = {}
    for creature_id in game_state.creature_ids:
        if creature_infos[creature_id].kind == -1 or creature_id in game_state.my_known_scans:
            continue
        if creature_id in game_state.visible_creatures:
            coords = game_state.visible_creatures[creature_id].coords
            fish_regions[creature_id] = np.array((coords[0], coords[0], coords[1], coords[1]))
            continue
        current_region = get_radar_region(creature_id, creature_infos, game_state.drones)
        if previous_game_state is not None:
            previous_region = previous_game_state.fish_regions[creature_id]
            current_region[0] = max(current_region[0], previous_region[0] - MAX_FISH_SPEED)
            current_region[1] = min(current_region[1], previous_region[1] + MAX_FISH_SPEED)
            current_region[2] = max(current_region[2], previous_region[2] - MAX_FISH_SPEED)
            current_region[3] = min(current_region[3], previous_region[3] + MAX_FISH_SPEED)
        fish_regions[creature_id] = current_region
    return fish_regions


def get_radar_region(creature_id: int, creature_infos: dict[int, CreatureInfo], drones: dict[int, Drone]) -> IntArray:
    """Builds the current-turn feasible radar rectangle for one fish.
    :param creature_id: Creature id whose feasible rectangle should be built.
    :param creature_infos: Creature metadata keyed by creature id.
    :param drones: Our drones keyed by id in the exact server order.
    :return: Current-turn feasible fish rectangle as min_x, max_x, min_y, max_y.
    """
    fish_kind = creature_infos[creature_id].kind
    min_x, max_x = 0, FIELD_SIZE - 1
    min_y, max_y = 2500 * (fish_kind + 1), 2500 * (fish_kind + 2)
    for drone in drones.values():
        radar = drone.radar[creature_id]
        if radar[1] == "L":
            max_x = min(max_x, drone.coords[0])
        else:
            min_x = max(min_x, drone.coords[0] + 1)
        if radar[0] == "T":
            max_y = min(max_y, drone.coords[1])
        else:
            min_y = max(min_y, drone.coords[1] + 1)
    return np.array((min_x, max_x, min_y, max_y))


def calculate_score_gain(creature_infos: dict[int, CreatureInfo], known_scans: set[int], saved_scans: set[int], foe_saved_scans: set[int]) -> int:
    """Calculates the score gained by turning all known scans into saved scans against the foe's saved scans.
    :param creature_infos: Creature metadata keyed by creature id.
    :param known_scans: Fish ids considered owned after cashout.
    :param saved_scans: Fish ids already saved before cashout.
    :param foe_saved_scans: Fish ids already saved by the opponent.
    :return: Additional score gained beyond the current saved scans.
    """
    score_gain = 0
    for creature_id in known_scans - saved_scans:
        points = creature_infos[creature_id].kind + 1
        score_gain += points * (1 if creature_id in foe_saved_scans else 2)
    for kind in range(3):
        fish_of_kind = {creature_id for creature_id, info in creature_infos.items() if info.kind == kind}
        if fish_of_kind <= known_scans and not fish_of_kind <= saved_scans:
            score_gain += 4 if fish_of_kind <= foe_saved_scans else 8
    for color in range(4):
        fish_of_color = {creature_id for creature_id, info in creature_infos.items() if info.color == color}
        if fish_of_color <= known_scans and not fish_of_color <= saved_scans:
            score_gain += 3 if fish_of_color <= foe_saved_scans else 6
    return score_gain


def choose_action(drone: Drone, creature_infos: dict[int, CreatureInfo], game_state: GameState, withdraw_now: bool) -> tuple[int, int, int]:
    """Chooses one move target and light setting for a drone.
    :param drone: Drone state to plan for.
    :param creature_infos: Creature metadata keyed by creature id.
    :param game_state: Parsed state for the current turn.
    :param withdraw_now: Indicates whether cashing out already guarantees victory.
    :return: Move x, move y, and light state.
    """
    if withdraw_now:
        base_target = np.array((drone.coords[0], 0))
    else:
        base_target = choose_base_target(drone, game_state)
    visible_monsters = [creature for creature_id, creature in game_state.visible_creatures.items() if creature_infos[creature_id].kind == -1]
    safe_target = choose_safe_target(drone, base_target, visible_monsters)
    light = choose_light(drone, game_state)
    if light == 1:
        drone.last_enabled = game_state.turn_number
    return safe_target[0], safe_target[1], light


def choose_base_target(drone: Drone, game_state: GameState) -> IntArray:
    """Chooses the fish pursuit target before monster safety is considered.
    :param drone: Drone state to plan for.
    :param game_state: Parsed state for the current turn.
    :return: Desired target point when only fish collection is considered.
    """
    fish_targets = [np.array(((region[0] + region[1]) // 2, (region[2] + region[3]) // 2)) for region in game_state.fish_regions.values()]
    if not fish_targets:
        return np.array((drone.coords[0], 0))
    if drone.preferred_side == "left":
        preferred_targets = [coords for coords in fish_targets if coords[0] < drone.coords[0]]
    else:
        preferred_targets = [coords for coords in fish_targets if coords[0] > drone.coords[0]]
    return min(preferred_targets or fish_targets, key=lambda coords: np.linalg.norm(drone.coords - coords))


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
                print(f"Safety: drone {drone.drone_id}: angle1={base_angle:.0f} angle2={final_angle:.0f}", file=sys.stderr)
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


def choose_light(drone: Drone, game_state: GameState) -> int:
    """Chooses the light setting from the turn timer and drone depth.
    :param drone: Drone state to plan for.
    :param game_state: Parsed state for the current turn.
    :return: Light setting for the move.
    """
    return int(drone.coords[1] >= 2500 and game_state.turn_number - drone.last_enabled >= 3)


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
