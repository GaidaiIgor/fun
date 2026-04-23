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
BIG_SCAN_RADIUS = 2000
MAX_FISH_SPEED = 400
MONSTER_COLLISION_RADIUS = 500
MONSTER_MONSTER_DISTANCE = 600
MONSTER_SWIM_SPEED = 270
MONSTER_DASH_SPEED = 540
MONSTER_HABITAT_TOP = 2500
SCAN_PROBABILITY = 0.5


@dataclass(slots=True)
class Creature:
    """Stores one creature snapshot and metadata.
    :var id: Unique creature identifier.
    :var color: Color index reported by the server.
    :var kind: Type index reported by the server, or -1 for monsters.
    :var coords: Current creature coordinates, or None if not currently known exactly.
    :var velocity: Current creature velocity vector, or None if not currently known exactly.
    :var region: Estimated fish rectangle as min_x, max_x, min_y, max_y, or None for monsters.
    """
    id: int
    color: int
    kind: int
    coords: IntArray | None = None
    velocity: IntArray | None = None
    region: IntArray | None = None


@dataclass(slots=True)
class Drone:
    """Stores the subset of drone state used by the planner.
    :var id: Unique drone identifier.
    :var coords: Current drone coordinates.
    :var emergency: Indicates whether the drone is already forced to surface.
    :var battery: Current battery charge.
    :var scans: Fish scans currently carried by the drone.
    :var radar: Last radar blip direction for each relevant creature.
    """
    id: int
    coords: IntArray
    emergency: bool
    battery: int
    scans: set[int] = field(default_factory=set)
    radar: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class PlayerState:
    """Stores one player's turn snapshot.
    :var score: Current player score.
    :var known_scans: Fish ids already scanned by the player, whether saved or currently carried.
    :var saved_scans: Fish ids already saved by the player.
    :var drones: Player drones keyed by id in the exact server order.
    """
    score: int
    known_scans: set[int]
    saved_scans: set[int]
    drones: dict[int, Drone]


@dataclass(slots=True)
class GameState:
    """Stores one full turn snapshot.
    :var turn_number: Zero-based turn index.
    :var my_state: Our current player state.
    :var foe_state: Enemy current player state.
    :var fishes: Current fish states keyed by creature id.
    :var monsters: Current monster states keyed by creature id.
    """
    turn_number: int
    my_state: PlayerState
    foe_state: PlayerState
    fishes: dict[int, Creature] = field(default_factory=dict)
    monsters: dict[int, Creature] = field(default_factory=dict)


def main_loop():
    """Runs the game loop and prints one command per drone each turn."""
    previous_game_state = read_initial_data()
    while True:
        game_state = update_game_state(previous_game_state)

        for drone in game_state.my_state.drones.values():
            print(f"drone {drone.id}: pos=({drone.coords[0]}, {drone.coords[1]})", file=sys.stderr)
        for foe_drone in game_state.foe_state.drones.values():
            print(f"foe drone {foe_drone.id}: pos=({foe_drone.coords[0]}, {foe_drone.coords[1]})", file=sys.stderr)
        for fish in game_state.fishes.values():
            if fish.coords is None:
                continue
            print(f"fish {fish.id}: pos=({fish.coords[0]}, {fish.coords[1]}), "
                  f"angle={np.rad2deg(np.arctan2(-fish.velocity[1], fish.velocity[0])):.0f}", file=sys.stderr)
        for monster in game_state.monsters.values():
            if monster.coords is None:
                continue
            print(f"monster {monster.id}: pos=({monster.coords[0]}, {monster.coords[1]}), "
                  f"angle={np.rad2deg(np.arctan2(-monster.velocity[1], monster.velocity[0])):.0f}", file=sys.stderr)
        for fish in sorted(game_state.fishes.values(), key=lambda fish: fish.id):
            if fish.region is not None:
                coords = get_midpoint(fish.region)
                print(f"estimate fish {fish.id}: pos=({coords[0]}, {coords[1]})", file=sys.stderr)

        choose_action(game_state)
        previous_game_state = game_state


def read_initial_data() -> GameState:
    """Reads creature metadata from the initialization block.
    :return: Metadata-only game state containing fishes and monsters.
    """
    creature_count = int(input())
    fishes = {}
    monsters = {}
    for _ in range(creature_count):
        creature_id, color, kind = map(int, input().split())
        creature = Creature(creature_id, color, kind)
        if kind == -1:
            monsters[creature_id] = creature
        else:
            fishes[creature_id] = creature
    return GameState(0, PlayerState(0, set(), set(), {}), PlayerState(0, set(), set(), {}), fishes, monsters)


def update_game_state(previous_game_state: GameState) -> GameState:
    """Updates one turn state from the previous snapshot and fresh server data.
    :param previous_game_state: Previous turn state, or the initial metadata-only state before turn 0.
    :return: Updated game state.
    """
    game_state = read_game_state(previous_game_state)
    update_fishes(game_state, previous_game_state)
    update_monsters(game_state, previous_game_state)
    return game_state


def read_game_state(previous_game_state: GameState) -> GameState:
    """Reads one full game turn into a freshly built game state.
    :param previous_game_state: Previous turn state used as metadata and carry-over reference.
    :return: Fresh current-turn game state.
    """
    scores = [int(input()) for _ in range(2)]
    saved_scans = []
    for _ in range(2):
        player_saved_scans = set()
        for _ in range(int(input())):
            player_saved_scans.add(int(input()))
        saved_scans.append(player_saved_scans)

    drones = []
    for _ in range(2):
        player_drones = {}
        for _ in range(int(input())):
            drone_id, drone_x, drone_y, emergency, battery = map(int, input().split())
            player_drones[drone_id] = Drone(drone_id, np.array((drone_x, drone_y)), bool(emergency), battery)
        drones.append(player_drones)

    drone_scan_count = int(input())
    known_scans = [set(saved_scans[0]), set(saved_scans[1])]
    for _ in range(drone_scan_count):
        drone_id, creature_id = map(int, input().split())
        player_ind = 0 if drone_id in drones[0] else 1
        drones[player_ind][drone_id].scans.add(creature_id)
        known_scans[player_ind].add(creature_id)

    fishes = {creature_id: Creature(creature_id, fish.color, fish.kind) for creature_id, fish in previous_game_state.fishes.items()}
    monsters = {creature_id: Creature(creature_id, monster.color, monster.kind) for creature_id, monster in previous_game_state.monsters.items()}

    visible_creature_count = int(input())
    for _ in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = map(int, input().split())
        if creature_id in fishes:
            fishes[creature_id].coords = np.array((creature_x, creature_y))
            fishes[creature_id].velocity = np.array((creature_vx, creature_vy))
        else:
            monsters[creature_id].coords = np.array((creature_x, creature_y))
            monsters[creature_id].velocity = np.array((creature_vx, creature_vy))

    radar_blip_count = int(input())
    for _ in range(radar_blip_count):
        drone_id_str, creature_id_str, radar_location = input().split()
        drone_id, creature_id = int(drone_id_str), int(creature_id_str)
        drones[0][drone_id].radar[creature_id] = radar_location
    states = [PlayerState(scores[player_ind], known_scans[player_ind], saved_scans[player_ind], drones[player_ind]) for player_ind in range(2)]
    return GameState(previous_game_state.turn_number + 1, states[0], states[1], fishes, monsters)


def update_fishes(game_state: GameState, previous_game_state: GameState):
    """Updates persistent fish rectangles from current radar, visibility, and the previous turn estimate.
    :param game_state: Parsed state for the current turn.
    :param previous_game_state: Previous turn state, or the initial metadata-only state before turn 0.
    """
    for fish in game_state.fishes.values():
        if fish.coords is not None:
            fish.region = np.array((fish.coords[0], fish.coords[0], fish.coords[1], fish.coords[1]))
            continue
        if all(fish.id not in drone.radar for drone in game_state.my_state.drones.values()):
            fish.region = None
            continue
        current_region = get_radar_region(fish, game_state.my_state.drones)
        previous_region = previous_game_state.fishes[fish.id].region
        if previous_region is not None:
            current_region[0] = max(current_region[0], previous_region[0] - MAX_FISH_SPEED)
            current_region[1] = min(current_region[1], previous_region[1] + MAX_FISH_SPEED)
            current_region[2] = max(current_region[2], previous_region[2] - MAX_FISH_SPEED)
            current_region[3] = min(current_region[3], previous_region[3] + MAX_FISH_SPEED)
        fish.region = current_region


def get_radar_region(fish: Creature, drones: dict[int, Drone]) -> IntArray:
    """Builds the current-turn feasible radar rectangle for one fish.
    :param fish: Fish whose feasible rectangle should be built.
    :param drones: Our drones keyed by id in the exact server order.
    :return: Current-turn feasible fish rectangle as min_x, max_x, min_y, max_y.
    """
    min_x, max_x = 0, FIELD_SIZE - 1
    min_y, max_y = 2500 * (fish.kind + 1), 2500 * (fish.kind + 2)
    for drone in drones.values():
        radar = drone.radar[fish.id]
        if radar[1] == "L":
            max_x = min(max_x, drone.coords[0])
        else:
            min_x = max(min_x, drone.coords[0] + 1)
        if radar[0] == "T":
            max_y = min(max_y, drone.coords[1])
        else:
            min_y = max(min_y, drone.coords[1] + 1)
    return np.array((min_x, max_x, min_y, max_y))


def update_monsters(game_state: GameState, previous_game_state: GameState):
    """Updates visible and hidden monster states from last exact sightings and server-observable inputs.
    :param game_state: Parsed state for the current turn.
    :param previous_game_state: Previous turn state, or the initial metadata-only state before turn 0.
    """
    if previous_game_state.turn_number == 0:
        return

    for monster in game_state.monsters.values():
        if monster.coords is None and previous_game_state.monsters[monster.id].coords is not None:
            monster.coords = previous_game_state.monsters[monster.id].coords + previous_game_state.monsters[monster.id].velocity

    drones = game_state.my_state.drones | game_state.foe_state.drones
    previous_drones = previous_game_state.my_state.drones | previous_game_state.foe_state.drones
    drone_light_circles = [(drone.coords, BIG_SCAN_RADIUS if drone.battery < previous_drones[drone_id].battery else SCAN_RADIUS)
                           for drone_id, drone in drones.items()]
    for monster in game_state.monsters.values():
        if monster.coords is None or monster.velocity is not None:
            continue
        monster.velocity = get_monster_velocity(monster, previous_game_state.monsters[monster.id].velocity, game_state.monsters, drone_light_circles)


def get_monster_velocity(monster: Creature, previous_velocity: IntArray, monsters: dict[int, Creature], drone_light_circles: list[tuple[IntArray, int]]) \
    -> IntArray:
    """Gets the current-turn monster velocity from its current coordinates and previous-turn observable state.
    :param monster: Monster whose velocity should be updated.
    :param previous_velocity: Monster velocity during the previous turn.
    :param monsters: Current tracked monster states keyed by creature id.
    :param drone_light_circles: Current drone coordinates paired with the light radius they used last turn.
    :return: Current monster velocity.
    """
    lit_drone_coords = [drone_coords for drone_coords, light_radius in drone_light_circles if np.linalg.norm(drone_coords - monster.coords) <= light_radius]
    if lit_drone_coords:
        nearest_drone_coords = min(lit_drone_coords, key=lambda drone_coords: np.linalg.norm(drone_coords - monster.coords))
        direction = nearest_drone_coords - monster.coords
        speed = MONSTER_DASH_SPEED
    else:
        direction = get_non_aggressive_monster_direction(monster, previous_velocity, monsters)
        speed = MONSTER_SWIM_SPEED
    if not direction.any():
        return direction
    return np.rint(direction * (speed / np.linalg.norm(direction))).astype(np.int64)


def get_non_aggressive_monster_direction(monster: Creature, previous_velocity: IntArray, monsters: dict[int, Creature]) -> IntArray:
    """Gets the current-turn non-aggressive monster direction.
    :param monster: Monster whose velocity should be updated.
    :param previous_velocity: Monster velocity during the previous turn.
    :param monsters: Current tracked monster states keyed by creature id.
    :return: Current non-aggressive monster direction.
    """
    if not previous_velocity.any():
        return previous_velocity
    nearby_monsters = [other_monster for other_monster in monsters.values() if other_monster.id != monster.id
                       and other_monster.coords is not None and np.linalg.norm(other_monster.coords - monster.coords) <= MONSTER_MONSTER_DISTANCE]
    if nearby_monsters:
        nearest_monster = min(nearby_monsters, key=lambda other_monster: np.linalg.norm(other_monster.coords - monster.coords))
        direction = monster.coords - nearest_monster.coords
    else:
        direction = previous_velocity
        if monster.coords[0] <= 0 or monster.coords[0] >= FIELD_SIZE - 1:
            direction = np.array((-direction[0], direction[1]))
        if monster.coords[1] <= MONSTER_HABITAT_TOP:
            direction = np.array((direction[0], abs(direction[1])))
    return direction


def choose_action(game_state: GameState):
    """Chooses move targets and light settings for both drones.
    :param game_state: Parsed state for the current turn.
    """
    player_states = [game_state.my_state, game_state.foe_state]
    base_scores, bonus_scores = get_projected_scores(game_state, player_states)
    defense_base_scores, defense_bonus_scores = get_projected_scores(game_state, player_states[::-1], False)
    totals = [base_scores[player_ind] + bonus_scores[player_ind] for player_ind in range(2)]
    defense_totals = [defense_base_scores[player_ind] + defense_bonus_scores[player_ind] for player_ind in range(2)]
    my_advantage = totals[0] > totals[1]
    defense_mode = defense_totals[0] > defense_totals[1]
    print(f"Projected: my_base={base_scores[0]}, my_bonus={bonus_scores[0]}, foe_base={base_scores[1]}, foe_bonus={bonus_scores[1]}", file=sys.stderr)
    print(f"Projected: my_base={defense_base_scores[1]}, my_bonus={defense_bonus_scores[1]}, "
          f"foe_base={defense_base_scores[0]}, foe_bonus={defense_bonus_scores[0]}", file=sys.stderr)
    drone_paths = get_drone_paths(game_state)
    monsters = [monster for monster in game_state.monsters.values() if monster.coords is not None]
    for drone in game_state.my_state.drones.values():
        max_depth = get_max_depth(drone, game_state)
        blocked_by_depth = defense_mode and max_depth is not None and drone.coords[1] >= max_depth
        drone_path = drone_paths[drone.id]
        if my_advantage and drone.scans or blocked_by_depth or not drone_path:
            base_target = np.array((drone.coords[0], 0))
        else:
            base_target = drone_path[0]
            if defense_mode and max_depth is not None:
                base_target = get_depth_limited_target(drone_path, max_depth)
                if base_target is None:
                    nearest_blocked_target = min(drone_path, key=lambda target: np.linalg.norm(target - drone.coords))
                    base_target = np.array((drone.coords[0], 0)) if drone.coords[1] - 500 < np.linalg.norm(nearest_blocked_target - drone.coords) \
                        else np.array((nearest_blocked_target[0], max_depth))

        safe_target = choose_safe_target(drone, base_target, monsters)
        light = choose_light(drone, safe_target, game_state, blocked_by_depth)
        print(f"MOVE {safe_target[0]} {safe_target[1]} {light}")


def get_projected_scores(game_state: GameState, states: list[PlayerState], use_claim_y: bool = True) -> tuple[list[int], list[int]]:
    """Calculates projected final base and bonus scores for two players in the given order.
    :param game_state: Parsed state for the current turn.
    :param states: Two player states whose projections should be returned in the same order.
    :param use_claim_y: Whether enemy blockers must also be above our claim depth.
    :return: Projected base scores and projected bonus scores in the same player order.
    """
    existing_fish_ids = {fish.id for fish in game_state.fishes.values() if fish.region is not None}
    possible_scans = [existing_fish_ids | state.known_scans for state in states]
    base_scores = [get_base_score(game_state.fishes, scans) for scans in possible_scans]
    saved_bonus_scores = [states[player_ind].score - get_base_score(game_state.fishes, states[player_ind].saved_scans) for player_ind in range(2)]
    future_bonus_pool = 0
    claimable_bonus = 0
    bonus_items = [(fish.kind + 1, {fish.id}) for fish in game_state.fishes.values()]
    bonus_items += [(4, {fish.id for fish in game_state.fishes.values() if fish.kind == kind}) for kind in range(3)]
    bonus_items += [(3, {fish.id for fish in game_state.fishes.values() if fish.color == color}) for color in range(4)]
    for points, required_scans in bonus_items:
        if required_scans <= states[0].saved_scans or required_scans <= states[1].saved_scans or \
            not required_scans <= possible_scans[0] and not required_scans <= possible_scans[1]:
            continue
        future_bonus_pool += points
        if not required_scans <= states[0].known_scans:
            continue
        if not required_scans <= states[1].known_scans or not use_claim_y:
            claimable_bonus += points
            continue
        claim_ys = [max(drone.coords[1] for drone in states[player_ind].drones.values() if drone.scans & (required_scans - states[player_ind].saved_scans))
                    for player_ind in range(2)]
        if not (claim_ys[1] < FIELD_SIZE // 2 and claim_ys[1] < claim_ys[0]):
            claimable_bonus += points
    bonus_scores = [saved_bonus_scores[0] + claimable_bonus, saved_bonus_scores[1] + future_bonus_pool - claimable_bonus]
    return base_scores, bonus_scores


def get_base_score(fishes: dict[int, Creature], obtainable_scans: set[int]) -> int:
    """Calculates the non-bonus score from fish and completed sets for one scan set.
    :param fishes: Fish states keyed by creature id.
    :param obtainable_scans: Fish ids assumed obtainable by one side.
    :return: Base score from those fish ids without first-completion bonuses.
    """
    fish_points = sum(fishes[fish_id].kind + 1 for fish_id in obtainable_scans)
    color_points = sum(3 for color in range(4) if {fish.id for fish in fishes.values() if fish.color == color} <= obtainable_scans)
    kind_points = sum(4 for kind in range(3) if {fish.id for fish in fishes.values() if fish.kind == kind} <= obtainable_scans)
    return fish_points + color_points + kind_points


def get_max_depth(drone: Drone, game_state: GameState) -> int | None:
    """Gets one drone's defense-mode depth cap from foe drones carrying competing unsaved scans.
    :param drone: Our drone whose depth cap should be calculated.
    :param game_state: Parsed state for the current turn.
    :return: Deepest allowed defense-mode y-coordinate for this drone, or None if no foe drone competes with it.
    """
    competing_foe_depths = [foe_drone.coords[1] for foe_drone in game_state.foe_state.drones.values()
                            if (drone.scans & foe_drone.scans) - game_state.my_state.saved_scans - game_state.foe_state.saved_scans]
    return None if not competing_foe_depths else max(0, min(competing_foe_depths) - 800)


def get_depth_limited_target(drone_path: list[IntArray], max_depth: int) -> IntArray | None:
    """Gets the first path target that can still be scanned while staying inside allowed depth.
    :param drone_path: Planned target coordinates for one drone.
    :param max_depth: Deepest y-coordinate the drone should still occupy in defense mode.
    :return: First target allowed by defense mode with capped depth, or None if none are allowed.
    """
    for target in drone_path:
        if target[1] <= max_depth + BIG_SCAN_RADIUS:
            return np.array((target[0], min(target[1], max_depth)))
    return None


def get_drone_paths(game_state: GameState) -> dict[int, list[IntArray]]:
    """Builds greedy fish-center paths for both drones.
    :param game_state: Parsed state for the current turn.
    :return: Full target coordinate path for each drone.
    """
    drone_ids = tuple(game_state.my_state.drones)
    fish_ids = \
        [fish_id for fish_id in sorted(game_state.fishes) if fish_id not in game_state.my_state.known_scans and game_state.fishes[fish_id].region is not None]
    fish_coords = {fish_id: get_midpoint(game_state.fishes[fish_id].region) for fish_id in fish_ids}
    drone_paths = {drone_id: [] for drone_id in drone_ids}
    drone_path_fish_ids = {drone_id: [] for drone_id in drone_ids}
    path_lengths = {drone_id: 0 for drone_id in drone_ids}
    while fish_ids:
        best_score = None
        best_drone_id = drone_ids[0]
        best_fish_id = fish_ids[0]
        for fish_id in fish_ids:
            for drone_id in drone_ids:
                previous_coords = game_state.my_state.drones[drone_id].coords if not drone_paths[drone_id] else drone_paths[drone_id][-1]
                candidate_length = path_lengths[drone_id] + np.linalg.norm(fish_coords[fish_id] - previous_coords)
                other_drone_id = drone_ids[1] if drone_id == drone_ids[0] else drone_ids[0]
                candidate_score = max(candidate_length, path_lengths[other_drone_id]), min(fish_coords[fish_id][0], FIELD_SIZE - fish_coords[fish_id][0])
                if best_score is None or candidate_score < best_score:
                    best_score = candidate_score
                    best_drone_id = drone_id
                    best_fish_id = fish_id
        previous_coords = game_state.my_state.drones[best_drone_id].coords if not drone_paths[best_drone_id] else drone_paths[best_drone_id][-1]
        drone_paths[best_drone_id].append(fish_coords[best_fish_id])
        drone_path_fish_ids[best_drone_id].append(best_fish_id)
        path_lengths[best_drone_id] += np.linalg.norm(fish_coords[best_fish_id] - previous_coords)
        fish_ids.remove(best_fish_id)

    for drone_id in drone_ids:
        previous_coords = game_state.my_state.drones[drone_id].coords
        path = []
        for path_ind in range(len(drone_paths[drone_id])):
            coords = drone_paths[drone_id][path_ind]
            fish_id = drone_path_fish_ids[drone_id][path_ind]
            path.append((coords, fish_id, f"{np.linalg.norm(coords - previous_coords):.0f}"))
            previous_coords = coords
        print(f"Path: drone {drone_id}: {path}", file=sys.stderr)

    return drone_paths


def choose_safe_target(drone: Drone, target: IntArray, monsters: list[Creature]) -> IntArray:
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
                print(f"Safety: drone {drone.id}: angle1={base_angle:.0f} angle2={final_angle:.0f}", file=sys.stderr)
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


def choose_light(drone: Drone, target: IntArray, game_state: GameState, blocked_by_depth: bool) -> int:
    """Chooses the light setting from scan probabilities at the planned end position.
    :param drone: Drone state to plan for.
    :param target: Planned move target for the turn.
    :param game_state: Parsed state for the current turn.
    :param blocked_by_depth: Whether this turn's intended hunt was blocked by defense mode.
    :return: Light setting for the move.
    """
    drone_end = get_end_point(drone.coords, target, DRONE_SPEED)
    scan_probability_threshold = SCAN_PROBABILITY / 2 if blocked_by_depth else SCAN_PROBABILITY
    fish_likely_nearby = any(get_scan_probability(fish.region, drone_end, BIG_SCAN_RADIUS) > scan_probability_threshold
                             for fish_id, fish in game_state.fishes.items() if fish_id not in game_state.my_state.known_scans and fish.region is not None)
    return int(drone.battery >= 5 and fish_likely_nearby)


def get_scan_probability(region: IntArray, scan_center: IntArray, scan_radius: int) -> float:
    """Gets the probability of scanning a fish uniformly distributed over one estimated region.
    :param region: Fish rectangle as min_x, max_x, min_y, max_y.
    :param scan_center: Center of the drone scan at end of turn.
    :param scan_radius: Radius of the considered scan.
    :return: Fraction of region positions covered by the scan.
    """
    region_width = region[1] - region[0] + 1
    region_height = region[3] - region[2] + 1
    x_coords = np.arange(region[0], region[1] + 1)
    max_dy_sq = scan_radius ** 2 - (x_coords - scan_center[0]) ** 2
    covered_xs = max_dy_sq >= 0
    if not covered_xs.any():
        return 0
    max_dy = np.sqrt(max_dy_sq[covered_xs]).astype(np.int64)
    min_y = np.maximum(region[2], scan_center[1] - max_dy)
    max_y = np.minimum(region[3], scan_center[1] + max_dy)
    return np.maximum(0, max_y - min_y + 1).sum() / (region_width * region_height)


def get_midpoint(region: IntArray) -> IntArray:
    """Gets the midpoint of one rectangular fish region.
    :param region: Fish rectangle as min_x, max_x, min_y, max_y.
    :return: Midpoint coordinates of the region.
    """
    return np.array(((region[0] + region[1]) // 2, (region[2] + region[3]) // 2))


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
