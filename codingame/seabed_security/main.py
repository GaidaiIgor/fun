"""Controls a Seabed Security bot with radar fish hunting and visible-monster avoidance."""

import math
from dataclasses import dataclass, field


type Point = tuple[int, int]
type Vector = tuple[int, int]

FIELD_SIZE = 10000
MAX_COORD = FIELD_SIZE - 1
SURFACE_Y = 0
DRONE_SPEED = 600
SCAN_RADIUS = 800
BIG_SCAN_RADIUS = 2000
MONSTER_COLLISION_RADIUS = 500
MONSTER_WAKE_MARGIN = 150
MONSTER_WARNING_DISTANCE = 900
SURFACE_IF_CARRYING_DISTANCE = 2000


@dataclass(slots=True)
class CreatureInfo:
    """Stores creature metadata from the initialization block.

    :var color: Color index reported by the referee.
    :var kind: Type index reported by the referee, or -1 for monsters.
    """

    color: int
    kind: int


@dataclass(slots=True)
class VisibleCreature:
    """Stores one visible creature snapshot.

    :var coords: Current creature coordinates.
    :var speed: Current creature velocity vector.
    """

    coords: Point
    speed: Vector


@dataclass(slots=True)
class Drone:
    """Stores the subset of drone state used by the planner.

    :var drone_id: Unique drone identifier.
    :var coords: Current drone coordinates.
    :var emergency: Indicates whether the drone is already forced to surface.
    :var battery: Current battery charge.
    :var scans: Fish scans currently carried by the drone.
    :var radar: Last radar blip direction for each relevant creature.
    """

    drone_id: int
    coords: Point
    emergency: bool
    battery: int
    scans: set[int] = field(default_factory=set)
    radar: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class TurnState:
    """Stores one full turn snapshot.

    :var drones: Our drones in the exact referee order.
    :var visible_creatures: Current visible creature snapshots keyed by creature id.
    """

    drones: list[Drone]
    visible_creatures: dict[int, VisibleCreature]


def main_loop() -> None:
    """Runs the game loop and prints one command per drone each turn."""
    creature_infos = read_initial_data()
    monster_ids = {creature_id for creature_id, info in creature_infos.items() if info.kind == -1}
    while True:
        turn = read_turn()
        for drone in turn.drones:
            print(choose_action(drone, monster_ids, turn.visible_creatures))


def read_initial_data() -> dict[int, CreatureInfo]:
    """Reads creature metadata from the initialization block.

    :return: Creature metadata keyed by creature id.
    """
    creature_count = int(input())
    creatures: dict[int, CreatureInfo] = {}
    for _ in range(creature_count):
        creature_id, color, kind = map(int, input().split())
        creatures[creature_id] = CreatureInfo(color, kind)
    return creatures


def read_turn() -> TurnState:
    """Reads one full game turn.

    :return: Parsed state for the current turn.
    """
    _ = int(input())
    _ = int(input())
    my_scan_count = int(input())
    for _ in range(my_scan_count):
        _ = int(input())
    foe_scan_count = int(input())
    for _ in range(foe_scan_count):
        _ = int(input())

    my_drone_count = int(input())
    drones: list[Drone] = []
    drones_by_id: dict[int, Drone] = {}
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = map(int, input().split())
        drone = Drone(drone_id, (drone_x, drone_y), bool(emergency), battery)
        drones.append(drone)
        drones_by_id[drone_id] = drone

    foe_drone_count = int(input())
    for _ in range(foe_drone_count):
        input()

    carried_scans: set[int] = set()
    drone_scan_count = int(input())
    for _ in range(drone_scan_count):
        drone_id, creature_id = map(int, input().split())
        if drone_id in drones_by_id:
            drones_by_id[drone_id].scans.add(creature_id)
            carried_scans.add(creature_id)

    visible_creatures: dict[int, VisibleCreature] = {}
    visible_creature_count = int(input())
    for _ in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = map(int, input().split())
        visible_creatures[creature_id] = VisibleCreature((creature_x, creature_y), (creature_vx, creature_vy))

    radar_blip_count = int(input())
    for _ in range(radar_blip_count):
        drone_id_str, creature_id_str, radar = input().split()
        drone_id, creature_id = int(drone_id_str), int(creature_id_str)
        if creature_id not in carried_scans and drone_id in drones_by_id:
            drones_by_id[drone_id].radar[creature_id] = radar

    return TurnState(drones, visible_creatures)


def choose_action(drone: Drone, monster_ids: set[int], visible_creatures: dict[int, VisibleCreature]) -> str:
    """Chooses one command for a drone.

    :param drone: Drone state to plan for.
    :param monster_ids: Creature ids belonging to monsters.
    :param visible_creatures: Visible creature snapshots for the turn.
    :return: Referee command for the drone.
    """
    if drone.emergency:
        return f"MOVE {drone.coords[0]} {SURFACE_Y} 0"
    visible_monsters = [creature for creature_id, creature in visible_creatures.items() if creature_id in monster_ids]
    base_target = choose_base_target(drone, monster_ids)
    preferred_light = int(drone.battery >= 5 and (drone.battery == 30 or drone.battery % 4 == 0))
    strategic_target = choose_strategic_target(drone, base_target, preferred_light, visible_monsters)
    safe_target, light = choose_safe_target(drone, strategic_target, preferred_light, visible_monsters)
    return f"MOVE {safe_target[0]} {safe_target[1]} {light}"


def choose_base_target(drone: Drone, monster_ids: set[int]) -> Point:
    """Chooses the fish pursuit target before monster safety is considered.

    :param drone: Drone state to plan for.
    :param monster_ids: Creature ids belonging to monsters.
    :return: Desired target point when only fish collection is considered.
    """
    fish_targets = \
        [guess_creature_coords(drone.coords, radar) for creature_id, radar in drone.radar.items() if creature_id not in monster_ids]
    if not fish_targets:
        return drone.coords[0], SURFACE_Y
    return min(fish_targets, key=lambda coords: distance_sq(drone.coords, coords))


def guess_creature_coords(drone_coords: Point, radar: str) -> Point:
    """Guesses one creature position from a single radar quadrant.

    :param drone_coords: Current drone coordinates.
    :param radar: Radar quadrant reported by the referee.
    :return: Midpoint guess inside the indicated map quadrant.
    """
    match radar:
        case "TL":
            return drone_coords[0] // 2, drone_coords[1] // 2
        case "TR":
            return (drone_coords[0] + FIELD_SIZE) // 2, drone_coords[1] // 2
        case "BL":
            return drone_coords[0] // 2, (drone_coords[1] + FIELD_SIZE) // 2
        case "BR":
            return (drone_coords[0] + FIELD_SIZE) // 2, (drone_coords[1] + FIELD_SIZE) // 2
        case _:
            raise AssertionError(f"Unexpected radar value: {radar}")


def choose_strategic_target(drone: Drone, base_target: Point, preferred_light: int, visible_monsters: list[VisibleCreature]) -> Point:
    """Chooses whether the drone should keep hunting or break toward the surface.

    :param drone: Drone state to plan for.
    :param base_target: Fish pursuit target chosen before monster safety is applied.
    :param preferred_light: Preferred light setting before safety overrides.
    :param visible_monsters: Visible monster snapshots for the turn.
    :return: Strategic target that the safety layer should try to preserve.
    """
    if not drone.scans or not visible_monsters:
        return base_target
    base_score = evaluate_candidate(drone, base_target, preferred_light, base_target, preferred_light, visible_monsters)
    nearest_monster_distance = min(distance(drone.coords, monster.coords) for monster in visible_monsters)
    if base_score[0] or base_score[1] or base_score[2] or nearest_monster_distance <= SURFACE_IF_CARRYING_DISTANCE:
        return drone.coords[0], SURFACE_Y
    return base_target


def choose_safe_target(drone: Drone, strategic_target: Point, preferred_light: int, visible_monsters: list[VisibleCreature]) -> tuple[Point, int]:
    """Chooses the safest available target and light combination for the turn.

    :param drone: Drone state to plan for.
    :param strategic_target: High-level target chosen before evasive detours are considered.
    :param preferred_light: Preferred light setting before safety overrides.
    :param visible_monsters: Visible monster snapshots for the turn.
    :return: Safe target point and light setting.
    """
    if not visible_monsters:
        return strategic_target, preferred_light
    light_options = [preferred_light] if preferred_light == 0 else [1, 0]
    best_choice = strategic_target, light_options[0]
    best_score = evaluate_candidate(drone, strategic_target, light_options[0], strategic_target, preferred_light, visible_monsters)
    for target in build_candidate_targets(drone, strategic_target, visible_monsters):
        for light in light_options:
            score = evaluate_candidate(drone, target, light, strategic_target, preferred_light, visible_monsters)
            if score < best_score:
                best_score = score
                best_choice = target, light
    return best_choice


def build_candidate_targets(drone: Drone, strategic_target: Point, visible_monsters: list[VisibleCreature]) -> list[Point]:
    """Builds a small set of evasive candidate targets for one drone.

    :param drone: Drone state to plan for.
    :param strategic_target: High-level target chosen before evasive detours are considered.
    :param visible_monsters: Visible monster snapshots for the turn.
    :return: Candidate target points ordered from direct progress to stronger evasive maneuvers.
    """
    nearest_monster = min(visible_monsters, key=lambda monster: distance_sq(drone.coords, monster.coords))
    nearest_offset_x = drone.coords[0] - nearest_monster.coords[0]
    nearest_offset_y = drone.coords[1] - nearest_monster.coords[1]
    escape_x = 0
    escape_y = -2 if drone.scans else -1
    for monster in visible_monsters:
        offset_x = drone.coords[0] - monster.coords[0]
        offset_y = drone.coords[1] - monster.coords[1]
        weight = 1 / max(distance_sq(drone.coords, monster.coords), 1)
        escape_x += offset_x * weight
        escape_y += offset_y * weight
    candidates = [
        strategic_target,
        (drone.coords[0], SURFACE_Y),
        point_in_direction(drone.coords, (escape_x, escape_y)),
        point_in_direction(drone.coords, (escape_x - 1, escape_y - 1)),
        point_in_direction(drone.coords, (escape_x + 1, escape_y - 1)),
        point_in_direction(drone.coords, (-nearest_offset_y, nearest_offset_x - 1)),
        point_in_direction(drone.coords, (nearest_offset_y, -nearest_offset_x - 1)),
        (0, SURFACE_Y),
        (MAX_COORD, SURFACE_Y),
    ]
    return list(dict.fromkeys(clamp_point(candidate) for candidate in candidates))


def evaluate_candidate(
    drone: Drone,
    target: Point,
    light: int,
    strategic_target: Point,
    preferred_light: int,
    visible_monsters: list[VisibleCreature],
) -> tuple[int, int, int, int, int, int]:
    """Scores one target and light combination for lexicographic safety-first selection.

    :param drone: Drone state to plan for.
    :param target: Candidate move target.
    :param light: Candidate light setting.
    :param strategic_target: High-level target chosen before evasive detours are considered.
    :param preferred_light: Preferred light setting before safety overrides.
    :param visible_monsters: Visible monster snapshots for the turn.
    :return: Lexicographic score where smaller is better.
    """
    drone_end = move_towards(drone.coords, target, DRONE_SPEED)
    drone_velocity = drone_end[0] - drone.coords[0], drone_end[1] - drone.coords[1]
    light_radius = BIG_SCAN_RADIUS if light else SCAN_RADIUS
    collision_count = 0
    aggro_count = 0
    close_count = 0
    min_distance = FIELD_SIZE * 2
    for monster in visible_monsters:
        monster_end = clamp_point((monster.coords[0] + monster.speed[0], monster.coords[1] + monster.speed[1]))
        approach_distance = minimum_distance_between_paths(drone.coords, drone_velocity, monster.coords, monster.speed)
        end_distance = distance(drone_end, monster_end)
        min_distance = min(min_distance, round_to_int(min(approach_distance, end_distance)))
        collision_count += approach_distance < MONSTER_COLLISION_RADIUS
        aggro_count += end_distance <= light_radius + MONSTER_WAKE_MARGIN
        close_count += min(approach_distance, end_distance) < MONSTER_WARNING_DISTANCE
    return \
        collision_count, \
        aggro_count, \
        close_count, \
        -min_distance, \
        round_to_int(distance(drone_end, strategic_target)), \
        int(light != preferred_light)


def move_towards(coords: Point, target_coords: Point, speed: int) -> Point:
    """Moves one point toward another with the game rounding rules.

    :param coords: Starting coordinates.
    :param target_coords: Desired destination point.
    :param speed: Maximum movement distance for the turn.
    :return: End coordinates after one turn of movement.
    """
    shift_x = target_coords[0] - coords[0]
    shift_y = target_coords[1] - coords[1]
    distance_to_target = math.hypot(shift_x, shift_y)
    if distance_to_target <= speed:
        return clamp_point(target_coords)
    scale = speed / distance_to_target
    return clamp_point((coords[0] + round_to_int(shift_x * scale), coords[1] + round_to_int(shift_y * scale)))


def minimum_distance_between_paths(
    start_a: Point,
    velocity_a: Vector,
    start_b: Point,
    velocity_b: Vector,
) -> float:
    """Computes the minimum distance between two linear movements during one turn.

    :param start_a: First path starting point.
    :param velocity_a: First path velocity vector for the turn.
    :param start_b: Second path starting point.
    :param velocity_b: Second path velocity vector for the turn.
    :return: Minimum distance reached between the two moving points during the turn.
    """
    relative_x = start_a[0] - start_b[0]
    relative_y = start_a[1] - start_b[1]
    relative_vx = velocity_a[0] - velocity_b[0]
    relative_vy = velocity_a[1] - velocity_b[1]
    relative_speed_sq = relative_vx * relative_vx + relative_vy * relative_vy
    if relative_speed_sq == 0:
        return math.hypot(relative_x, relative_y)
    time = max(0, min(1, -(relative_x * relative_vx + relative_y * relative_vy) / relative_speed_sq))
    return math.hypot(relative_x + relative_vx * time, relative_y + relative_vy * time)


def point_in_direction(origin: Point, direction: tuple[float, float]) -> Point:
    """Projects a point far away from an origin along a direction vector.

    :param origin: Starting coordinates.
    :param direction: Direction vector that defines the projection heading.
    :return: Clamped map coordinate lying far away in the chosen direction.
    """
    distance_to_target = math.hypot(direction[0], direction[1])
    if distance_to_target == 0:
        return origin[0], SURFACE_Y
    scale = FIELD_SIZE / distance_to_target
    return clamp_point((origin[0] + round_to_int(direction[0] * scale), origin[1] + round_to_int(direction[1] * scale)))


def distance(point_a: Point, point_b: Point) -> float:
    """Computes Euclidean distance between two points.

    :param point_a: First point.
    :param point_b: Second point.
    :return: Euclidean distance between the points.
    """
    return math.dist(point_a, point_b)


def distance_sq(point_a: Point, point_b: Point) -> int:
    """Computes squared Euclidean distance between two points.

    :param point_a: First point.
    :param point_b: Second point.
    :return: Squared Euclidean distance between the points.
    """
    delta_x = point_a[0] - point_b[0]
    delta_y = point_a[1] - point_b[1]
    return delta_x * delta_x + delta_y * delta_y


def clamp_point(point: Point) -> Point:
    """Clamps a point to the game map boundaries.

    :param point: Point to clamp.
    :return: Point limited to the playable map area.
    """
    return max(0, min(MAX_COORD, point[0])), max(0, min(MAX_COORD, point[1]))


def round_to_int(value: float) -> int:
    """Rounds one float to the nearest integer with symmetric half-away-from-zero behavior.

    :param value: Float value to round.
    :return: Rounded integer.
    """
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


main_loop()
