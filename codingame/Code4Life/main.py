"""Implements an expertise-aware Silver bot for Code4Life."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations, permutations

MAX_SAMPLES = 3
STORAGE_LIMIT = 10
BATCH_RETURN_TURNS = 1
DIAGNOSIS_SLICE = 8
DROP_COOLDOWN = 8
TOTAL_TURNS = 200
SELF = 0
MOLECULE_TYPES = "ABCDE"
PROJECTS: tuple[tuple[int, int, int, int, int], ...] = ()
ACTIVE_PROJECTS: tuple[tuple[int, int, int, int, int], ...] = ()
RECENT_DROPS: dict[int, int] = {}
DISTANCE = {
    "START_POS": {"SAMPLES": 2, "DIAGNOSIS": 2, "MOLECULES": 2, "LABORATORY": 2},
    "SAMPLES": {"SAMPLES": 0, "DIAGNOSIS": 3, "MOLECULES": 3, "LABORATORY": 3},
    "DIAGNOSIS": {"SAMPLES": 3, "DIAGNOSIS": 0, "MOLECULES": 3, "LABORATORY": 4},
    "MOLECULES": {"SAMPLES": 3, "DIAGNOSIS": 3, "MOLECULES": 0, "LABORATORY": 3},
    "LABORATORY": {"SAMPLES": 3, "DIAGNOSIS": 4, "MOLECULES": 3, "LABORATORY": 0},
}


@dataclass(slots=True, frozen=True)
class Player:
    """:var target: Module currently occupied or targeted by the robot.
    :var eta: Turns remaining before the robot can act at a module.
    :var score: Health points already scored by the robot.
    :var storage: Molecules currently carried by the robot in A-E order.
    :var expertise: Expertise already gained by the robot in A-E order.
    """

    target: str
    eta: int
    score: int
    storage: tuple[int, int, int, int, int]
    expertise: tuple[int, int, int, int, int]


@dataclass(slots=True, frozen=True)
class Sample:
    """:var sample_id: Unique identifier of the sample.
    :var carried_by: Owner flag for the sample, or -1 while it is in the cloud.
    :var rank: Rank of the sample.
    :var gain: Molecule expertise granted when the sample is completed.
    :var health: Health points granted when the sample is produced.
    :var cost: Molecule requirements for the sample in A-E order.
    """

    sample_id: int
    carried_by: int
    rank: int
    gain: str
    health: int
    cost: tuple[int, int, int, int, int]

    def is_diagnosed(self) -> bool:
        """:return: Whether the sample is already diagnosed."""
        return self.health >= 0


def main():
    """Runs the game loop and prints one action per turn."""
    global ACTIVE_PROJECTS, PROJECTS
    PROJECTS = tuple(tuple(int(value) for value in input().split()) for _ in range(int(input())))
    ACTIVE_PROJECTS = PROJECTS
    debug(f"projects={PROJECTS}")
    turn = 0
    while True:
        me, opponent, available, samples = read_turn()
        update_projects(me.expertise, opponent.expertise)
        mine = [sample for sample in samples if sample.carried_by == SELF]
        theirs = [sample for sample in samples if sample.carried_by == 1]
        cloud = [sample for sample in samples if sample.carried_by == -1]
        remaining_turns = TOTAL_TURNS - turn
        action = choose_action(me, opponent, available, mine, theirs, cloud, remaining_turns)
        debug(
            f"t={turn} at={me.target}/{me.eta} hp={me.score}-{opponent.score} av={available} hold={me.storage} exp={me.expertise} "
            f"opp={opponent.target}/{opponent.eta} {opponent.storage}/{opponent.expertise} mine={sample_trace(mine)} cloud={len(cloud)} "
            f"theirs={sample_trace(theirs)} proj={ACTIVE_PROJECTS} -> {action}"
        )
        print(action)
        turn += 1


def read_turn() -> tuple[Player, Player, tuple[int, int, int, int, int], list[Sample]]:
    """:return: Both players, the available molecules, and every sample visible on the current turn."""
    me = read_player()
    opponent = read_player()
    available = tuple(int(value) for value in input().split())
    return me, opponent, available, read_samples(int(input()))


def read_player() -> Player:
    """:return: Parsed state for one player."""
    inputs = input().split()
    return Player(
        inputs[0],
        int(inputs[1]),
        int(inputs[2]),
        tuple(int(value) for value in inputs[3:8]),
        tuple(int(value) for value in inputs[8:13]),
    )


def read_samples(count: int) -> list[Sample]:
    """:param count: Number of sample lines to read.
    :return: Parsed samples for the current turn.
    """
    return [read_sample() for _ in range(count)]


def read_sample() -> Sample:
    """:return: Parsed state for one sample."""
    inputs = input().split()
    return Sample(
        int(inputs[0]),
        int(inputs[1]),
        int(inputs[2]),
        inputs[3],
        int(inputs[4]),
        tuple(int(value) for value in inputs[5:10]),
    )


def choose_action(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    mine: list[Sample],
    theirs: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param mine: Samples currently carried by our robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print for the current turn.
    """
    if me.eta:
        return "WAIT"
    planned_available = pressured_available(available, opponent, theirs)
    match me.target:
        case "SAMPLES":
            return choose_at_samples(me, opponent, available, planned_available, mine, theirs, cloud, remaining_turns)
        case "DIAGNOSIS":
            return choose_at_diagnosis(me, opponent, available, planned_available, mine, theirs, cloud, remaining_turns)
        case "MOLECULES":
            return choose_at_molecules(me, opponent, available, planned_available, mine, theirs, cloud, remaining_turns)
        case "LABORATORY":
            return choose_at_laboratory(me, opponent, available, planned_available, mine, theirs, cloud, remaining_turns)
        case _:
            return "GOTO SAMPLES" if not mine else "GOTO DIAGNOSIS"


def choose_at_samples(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    mine: list[Sample],
    theirs: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param mine: Samples currently carried by our robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at SAMPLES.
    """
    desired = desired_sample_count(me.expertise, remaining_turns)
    cloud_batch = best_cloud_batch(me, opponent, available, planned_available, theirs, cloud, remaining_turns, "SAMPLES") if diagnosed_samples(cloud) else []
    if should_deny_endgame(me, opponent, theirs, available, remaining_turns):
        return "GOTO MOLECULES"
    if remaining_turns <= 14 and not mine and not cloud_batch:
        return "WAIT"
    if cloud_batch and (
        len(mine) >= desired
        or remaining_turns <= 42
        or batch_health(cloud_batch) >= 20
        or batch_completed_project_indexes(cloud_batch, me.expertise)
    ):
        return "GOTO DIAGNOSIS"
    if len(mine) < desired:
        return f"CONNECT {sample_rank(me.expertise, remaining_turns)}"
    return "GOTO DIAGNOSIS" if mine or cloud_batch else "WAIT"


def desired_sample_count(expertise: tuple[int, int, int, int, int], remaining_turns: int) -> int:
    """:param expertise: Expertise already gained by our robot.
    :param remaining_turns: Number of turns left including the current one.
    :return: Preferred number of carried samples.
    """
    total = sum(expertise)
    return 1 if remaining_turns <= 16 else 2 if remaining_turns <= 30 else 3


def sample_rank(expertise: tuple[int, int, int, int, int], remaining_turns: int) -> int:
    """:param expertise: Expertise already gained by our robot.
    :param remaining_turns: Number of turns left including the current one.
    :return: Rank to request from the samples machine.
    """
    total = sum(expertise)
    covered = sum(value > 0 for value in expertise)
    gap = best_project_gap(expertise)
    return 1 if remaining_turns <= 14 or total < 2 or gap <= 3 and remaining_turns <= 36 else \
        2 if remaining_turns <= 48 or gap <= 6 and remaining_turns <= 58 or total < 10 else \
        3 if max(expertise) >= 3 and covered >= 4 else 2


def best_project_gap(expertise: tuple[int, int, int, int, int]) -> int:
    """:param expertise: Expertise already gained by our robot.
    :return: Smallest remaining expertise distance to any active science project.
    """
    return min((sum(max(need - have, 0) for need, have in zip(project, expertise)) for project in ACTIVE_PROJECTS), default=99)


def sample_is_junk(sample: Sample, expertise: tuple[int, int, int, int, int], remaining_turns: int) -> bool:
    """:param sample: Diagnosed sample to evaluate.
    :param expertise: Expertise already gained by our robot.
    :param remaining_turns: Number of turns left including the current one.
    :return: Whether the sample is too weak to spend a production cycle on.
    """
    return sample.is_diagnosed() and not sample_gain_helps_project(sample, expertise) and (
        sample.health <= 1 or remaining_turns <= 30 and sample.health < 20
    )


def sample_gain_helps_project(sample: Sample, expertise: tuple[int, int, int, int, int]) -> bool:
    """:param sample: Diagnosed sample to evaluate.
    :param expertise: Expertise already gained by our robot.
    :return: Whether the sample gain advances any active science project.
    """
    index = gain_index(sample.gain)
    return index >= 0 and any(project[index] > expertise[index] for project in ACTIVE_PROJECTS)


def choose_at_diagnosis(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    mine: list[Sample],
    theirs: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param mine: Samples currently carried by our robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at DIAGNOSIS.
    """
    undiagnosed = undiagnosed_samples(mine)
    if undiagnosed:
        return f"CONNECT {undiagnosed[0].sample_id}"
    diagnosed = carried_diagnosed_samples(mine)
    project_deadlines = opponent_project_finish_times(opponent, theirs, available)
    reserve = released_available(available, opponent)
    owned = best_owned_batch(diagnosed, me.storage, me.expertise, reserve, remaining_turns, "DIAGNOSIS", project_deadlines)
    chosen = best_batch(
        diagnosed,
        diagnosed_samples(cloud),
        me.storage,
        me.expertise,
        reserve,
        planned_available,
        remaining_turns,
        project_deadlines,
    )
    if owned and any(sample.carried_by == -1 for sample in chosen):
        owned_projects = batch_completed_project_indexes(owned, me.expertise)
        chosen_projects = batch_completed_project_indexes(chosen, me.expertise)
        extra_projects = chosen_projects - owned_projects
        if not extra_projects:
            chosen = owned
        else:
            chosen_finish_time = finish_time_from(
                "DIAGNOSIS",
                chosen,
                sum(sample.carried_by == -1 for sample in chosen),
                me.storage,
                me.expertise,
            )
            if all(project_deadlines.get(index, TOTAL_TURNS) <= chosen_finish_time for index in extra_projects):
                chosen = owned
    for sample in chosen:
        if sample.carried_by == -1:
            RECENT_DROPS.pop(sample.sample_id, None)
            return f"CONNECT {sample.sample_id}"
    for sample in diagnosed:
        if sample not in chosen and sample_is_junk(sample, me.expertise, remaining_turns):
            RECENT_DROPS[sample.sample_id] = TOTAL_TURNS - remaining_turns
            return f"CONNECT {sample.sample_id}"
    if chosen and batch_complete(chosen, me.storage, me.expertise):
        return "GOTO LABORATORY"
    if chosen:
        return "GOTO MOLECULES"
    rejected = carried_diagnosed_samples(mine)
    if rejected:
        future = best_owned_batch(rejected, me.storage, me.expertise, reserve, remaining_turns - 1, "DIAGNOSIS")
        if future:
            return "WAIT"
        eventual = best_owned_batch(
            rejected, me.storage, me.expertise, eventual_available(available, opponent), remaining_turns - 1, "DIAGNOSIS"
        )
        if eventual:
            return "GOTO MOLECULES"
        sample = worst_sample(rejected, me.expertise)
        if len(mine) < desired_sample_count(me.expertise, remaining_turns):
            index = gain_index(sample.gain)
            if index >= 0 and any(0 < project[index] - me.expertise[index] <= 2 for project in ACTIVE_PROJECTS):
                return "GOTO SAMPLES"
        RECENT_DROPS[sample.sample_id] = TOTAL_TURNS - remaining_turns
        return f"CONNECT {sample.sample_id}"
    if should_deny_endgame(me, opponent, theirs, available, remaining_turns):
        return "GOTO MOLECULES"
    return "WAIT" if remaining_turns <= 14 else "GOTO SAMPLES"


def batch_completed_project_indexes(samples: list[Sample], expertise: tuple[int, int, int, int, int]) -> set[int]:
    """:param samples: Candidate batch of diagnosed samples.
    :param expertise: Expertise already gained by our robot.
    :return: Indexes of active science projects completed after the batch gains resolve.
    """
    progress = list(expertise)
    for sample in samples:
        index = gain_index(sample.gain)
        if index >= 0:
            progress[index] += 1
    return {index for index, project in enumerate(ACTIVE_PROJECTS) if project_complete(project, progress)}


def player_finish_time_from(player: Player, samples: list[Sample]) -> int:
    """:param player: Current state of the player being evaluated.
    :param samples: Diagnosed samples the player may try to finish.
    :return: Turns needed for the player to finish the batch from the current state.
    """
    return player.eta + finish_time_from(player.target, ordered_samples(samples, player.expertise), 0, player.storage, player.expertise)


def opponent_project_finish_times(
    opponent: Player,
    theirs: list[Sample],
    available: tuple[int, int, int, int, int],
) -> dict[int, int]:
    """:param opponent: Current state of the opposing robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param available: Molecules still available in the pool.
    :return: Earliest opponent finish time for every currently claimable active project.
    """
    finish_times: dict[int, int] = {}
    for batch in sample_subsets(diagnosed_samples(theirs)):
        ordered = ordered_samples(batch, opponent.expertise)
        if not ordered or not batch_fits(ordered, opponent.storage, opponent.expertise, available):
            continue
        finish_time = player_finish_time_from(opponent, ordered)
        for index in batch_completed_project_indexes(ordered, opponent.expertise):
            finish_times[index] = min(finish_times.get(index, TOTAL_TURNS), finish_time)
    return finish_times


def should_deny_endgame(
    me: Player,
    opponent: Player,
    theirs: list[Sample],
    available: tuple[int, int, int, int, int],
    remaining_turns: int,
) -> bool:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param available: Molecules still available in the pool.
    :param remaining_turns: Number of turns left including the current one.
    :return: Whether denying opponent molecules is more urgent than starting more samples.
    """
    batch, value = opponent_threat_batch(opponent, theirs, available, remaining_turns)
    missing = batch_missing_vector(batch, opponent.storage, opponent.expertise) if batch else (0, 0, 0, 0, 0)
    return remaining_turns <= 42 and value and opponent.score + value >= me.score - 10 and any(missing)


def denial_molecule(
    me: Player,
    opponent: Player,
    theirs: list[Sample],
    available: tuple[int, int, int, int, int],
    remaining_turns: int,
) -> str | None:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param available: Molecules still available in the pool.
    :param remaining_turns: Number of turns left including the current one.
    :return: Molecule type to steal from the opponent's late threat, if useful.
    """
    if sum(me.storage) >= STORAGE_LIMIT:
        return None
    batch, value = opponent_threat_batch(opponent, theirs, available, remaining_turns)
    if not value or opponent.score + value < me.score - 10:
        return None
    missing = batch_missing_vector(batch, opponent.storage, opponent.expertise)
    collectable = [index for index, need in enumerate(missing) if need and available[index]]
    if not collectable:
        return None
    return MOLECULE_TYPES[max(collectable, key=lambda index: (missing[index], 5 - available[index], project_pressure(index, opponent.expertise)))]


def opponent_threat_batch(
    opponent: Player,
    theirs: list[Sample],
    available: tuple[int, int, int, int, int],
    remaining_turns: int,
) -> tuple[list[Sample], int]:
    """:param opponent: Current state of the opposing robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param available: Molecules still available in the pool.
    :param remaining_turns: Number of turns left including the current one.
    :return: Best opposing finishable batch and its score swing.
    """
    best: list[Sample] = []
    best_value = 0
    pool = eventual_available(available, opponent)
    for batch in sample_subsets(diagnosed_samples(theirs)):
        ordered = ordered_samples(batch, opponent.expertise)
        if not ordered or not batch_fits(ordered, opponent.storage, opponent.expertise, pool):
            continue
        finish_time = player_finish_time_from(opponent, ordered)
        if finish_time > remaining_turns:
            continue
        value = batch_health(ordered) + batch_project_value(ordered, opponent.expertise, finish_time, None)
        if value > best_value:
            best = ordered
            best_value = value
    return best, best_value


def project_pressure(index: int, expertise: tuple[int, int, int, int, int]) -> int:
    """:param index: Molecule index to evaluate.
    :param expertise: Expertise already gained by the opponent.
    :return: How much the molecule type matters for active science projects.
    """
    return max((project[index] - expertise[index] for project in ACTIVE_PROJECTS), default=0)


def best_batch(
    mine: list[Sample],
    cloud: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    remaining_turns: int,
    project_deadlines: dict[int, int] | None = None,
) -> list[Sample]:
    """:param mine: Diagnosed samples currently carried by our robot.
    :param cloud: Diagnosed samples currently available in the cloud.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param remaining_turns: Number of turns left including the current one.
    :param project_deadlines: Earliest opposing finish time for each active project, if race-aware scoring is enabled.
    :return: Highest-value finishable batch reachable from DIAGNOSIS.
    """
    best: list[Sample] = []
    best_value = batch_value([], 0, expertise, available, project_deadlines)
    room = MAX_SAMPLES - len(mine)
    mine = [sample for sample in mine if not sample_is_junk(sample, expertise, remaining_turns)]
    candidates = diagnosis_candidates(cloud, expertise, remaining_turns)
    for owned in sample_subsets(mine):
        for size in range(min(room, len(candidates)) + 1):
            for extra in combinations(candidates, size):
                batch = ordered_samples([*owned, *extra], expertise)
                if not batch or not batch_fits(batch, storage, expertise, available) or size and \
                    not batch_fits(batch, storage, expertise, planned_available):
                    continue
                finish_time = finish_time_from("DIAGNOSIS", batch, size, storage, expertise)
                if finish_time > remaining_turns:
                    continue
                value = batch_value(batch, finish_time, expertise, planned_available, project_deadlines)
                if value > best_value:
                    best = batch
                    best_value = value
    return best


def diagnosis_candidates(cloud: list[Sample], expertise: tuple[int, int, int, int, int], remaining_turns: int) -> list[Sample]:
    """:param cloud: Diagnosed samples currently available in the cloud.
    :param expertise: Expertise already gained by our robot.
    :param remaining_turns: Number of turns left including the current one.
    :return: Short candidate list worth combining at DIAGNOSIS.
    """
    current_turn = TOTAL_TURNS - remaining_turns
    cloud = [
        sample for sample in cloud
        if remaining_turns <= 18 or current_turn - RECENT_DROPS.get(sample.sample_id, -DROP_COOLDOWN - 1) > DROP_COOLDOWN
    ]
    pool: dict[int, Sample] = {}
    for sample in sorted(cloud, key=lambda item: sample_priority(item, expertise), reverse=True)[:DIAGNOSIS_SLICE]:
        pool[sample.sample_id] = sample
    for sample in sorted(cloud, key=sample_health_priority, reverse=True)[:DIAGNOSIS_SLICE]:
        pool[sample.sample_id] = sample
    for sample in sorted(cloud, key=lambda item: sample_cost_priority(item, expertise))[:DIAGNOSIS_SLICE]:
        pool[sample.sample_id] = sample
    return list(pool.values())


def best_cloud_batch(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    theirs: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
    module: str,
) -> list[Sample]:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param theirs: Samples currently carried by the opposing robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :param module: Module from which the robot would travel to DIAGNOSIS.
    :return: Best worth-chasing cloud batch reachable after visiting DIAGNOSIS.
    """
    return best_batch(
        [],
        diagnosed_samples(cloud),
        me.storage,
        me.expertise,
        eventual_available(available, opponent),
        planned_available,
        remaining_turns - DISTANCE[module]["DIAGNOSIS"],
        opponent_project_finish_times(opponent, theirs, available),
    )


def choose_at_molecules(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    mine: list[Sample],
    theirs: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param mine: Samples currently carried by our robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at MOLECULES.
    """
    chosen = best_owned_batch(diagnosed_samples(mine), me.storage, me.expertise, available, remaining_turns, "MOLECULES")
    if not chosen:
        future = best_owned_batch(diagnosed_samples(mine), me.storage, me.expertise, released_available(available, opponent), remaining_turns - 1, "MOLECULES")
        if not future:
            future = best_owned_batch(
                diagnosed_samples(mine), me.storage, me.expertise, eventual_available(available, opponent), remaining_turns - 1, "MOLECULES"
            )
            if future and (
                (opponent.target == "MOLECULES" and opponent.eta <= 1)
                or (opponent.target == "LABORATORY" and opponent.eta <= 2)
            ):
                molecule = next_collectable_molecule(future, me.storage, me.expertise, available, planned_available, opponent.expertise)
                return "WAIT" if molecule is None else f"CONNECT {molecule}"
            molecule = denial_molecule(me, opponent, theirs, available, remaining_turns)
            if molecule is not None:
                return f"CONNECT {molecule}"
            cloud_batch = best_cloud_batch(me, opponent, available, planned_available, theirs, cloud, remaining_turns, "MOLECULES")
            if remaining_turns <= 14 and not mine and not cloud_batch:
                return "WAIT"
            return "GOTO DIAGNOSIS" if mine or cloud_batch else "GOTO SAMPLES"
        molecule = next_collectable_molecule(future, me.storage, me.expertise, available, planned_available, opponent.expertise)
        return "WAIT" if molecule is None else f"CONNECT {molecule}"
    if batch_complete(chosen, me.storage, me.expertise):
        return "GOTO LABORATORY"
    return f"CONNECT {next_needed_molecule(chosen, me.storage, me.expertise, available, planned_available, opponent.expertise)}"


def best_owned_batch(
    mine: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
    remaining_turns: int,
    module: str,
    project_deadlines: dict[int, int] | None = None,
) -> list[Sample]:
    """:param mine: Diagnosed samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param remaining_turns: Number of turns left including the current one.
    :param module: Module from which the robot is planning the finish.
    :param project_deadlines: Earliest opposing finish time for each active project, if race-aware scoring is enabled.
    :return: Highest-value finishable subset of carried samples.
    """
    best: list[Sample] = []
    best_value = batch_value([], 0, expertise, available, project_deadlines)
    mine = [sample for sample in mine if not sample_is_junk(sample, expertise, remaining_turns)]
    for batch in sample_subsets(mine):
        ordered = ordered_samples(batch, expertise)
        if not ordered or not batch_fits(ordered, storage, expertise, available):
            continue
        finish_time = finish_time_from(module, ordered, 0, storage, expertise)
        if finish_time > remaining_turns:
            continue
        value = batch_value(ordered, finish_time, expertise, available, project_deadlines)
        if value > best_value:
            best = ordered
            best_value = value
    return best


def finish_time_from(
    module: str,
    samples: list[Sample],
    extra_downloads: int,
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
) -> int:
    """:param module: Module from which the robot is planning the finish.
    :param samples: Samples chosen for completion.
    :param extra_downloads: Number of extra cloud pickups still needed.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :return: Turns needed to finish the batch from the given module.
    """
    missing = batch_missing_cost(samples, storage, expertise)
    match module:
        case "DIAGNOSIS":
            return (
                extra_downloads + DISTANCE["DIAGNOSIS"]["LABORATORY"] + len(samples)
                if not missing
                else extra_downloads + DISTANCE["DIAGNOSIS"]["MOLECULES"] + missing + DISTANCE["MOLECULES"]["LABORATORY"] + len(samples)
            )
        case "MOLECULES":
            return DISTANCE["MOLECULES"]["LABORATORY"] + len(samples) if not missing else missing + DISTANCE["MOLECULES"]["LABORATORY"] + len(samples)
        case "LABORATORY":
            return len(samples) if not missing else DISTANCE["LABORATORY"]["MOLECULES"] + missing + DISTANCE["MOLECULES"]["LABORATORY"] + len(samples)
        case _:
            return DISTANCE[module]["DIAGNOSIS"] + finish_time_from("DIAGNOSIS", samples, extra_downloads, storage, expertise)


def batch_value(
    samples: list[Sample],
    finish_time: int,
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
    project_deadlines: dict[int, int] | None = None,
) -> tuple[float, int, int, int, int, int, tuple[int, ...]]:
    """:param samples: Candidate batch of diagnosed samples.
    :param finish_time: Turns needed to complete the batch from the current module.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param project_deadlines: Earliest opposing finish time for each active project, if race-aware scoring is enabled.
    :return: Comparable tuple describing how attractive the batch is.
    """
    reward = batch_health(samples) + batch_expertise_value(samples, expertise) + batch_project_value(samples, expertise, finish_time, project_deadlines)
    scarcity = batch_scarcity(samples, expertise, available)
    return (
        reward / (finish_time + BATCH_RETURN_TURNS + scarcity / 2) if samples else 0,
        -scarcity,
        reward,
        batch_health(samples),
        -finish_time,
        -len(samples),
        tuple(-sample.sample_id for sample in samples),
    )


def batch_expertise_value(samples: list[Sample], expertise: tuple[int, int, int, int, int]) -> int:
    """:param samples: Samples whose expertise gains should be valued.
    :param expertise: Expertise already gained by our robot.
    :return: Bonus value granted to the batch for useful expertise gains.
    """
    progress = list(expertise)
    value = 0
    for sample in samples:
        index = gain_index(sample.gain)
        if index < 0:
            continue
        value += expertise_need_value(progress, index)
        progress[index] += 1
    return value


def batch_project_value(
    samples: list[Sample],
    expertise: tuple[int, int, int, int, int],
    finish_time: int,
    project_deadlines: dict[int, int] | None,
) -> int:
    """:param samples: Samples whose project completions should be valued.
    :param expertise: Expertise already gained by our robot.
    :param finish_time: Turns needed to complete the batch from the current module.
    :param project_deadlines: Earliest opposing finish time for each active project.
    :return: Bonus value granted for science projects we can still realistically claim first.
    """
    if project_deadlines is None:
        return 50 * len(batch_completed_project_indexes(samples, expertise))
    return 50 * sum(project_deadlines.get(index, TOTAL_TURNS) > finish_time for index in batch_completed_project_indexes(samples, expertise))


def batch_scarcity(
    samples: list[Sample],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
) -> int:
    """:param samples: Samples whose missing molecules should be evaluated.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :return: Penalty representing how contested the batch is under current supply.
    """
    required = batch_required_vector(samples, expertise)
    return sum(need * max(5 - pool, 0) for need, pool in zip(required, available))


def expertise_need_value(expertise: list[int], index: int) -> int:
    """:param expertise: Expertise already gained by our robot as a mutable list.
    :param index: Molecule type index whose expertise gain is being valued.
    :return: Small heuristic bonus for improving that expertise type.
    """
    value = 0
    for project in ACTIVE_PROJECTS:
        if project[index] <= expertise[index]:
            continue
        needed = project[index] - expertise[index]
        remaining = sum(max(need - have - (slot == index), 0) for slot, (need, have) in enumerate(zip(project, expertise)))
        value += 50 if not remaining else max(44 - 6 * remaining, 0) + 3 * needed
    return value


def choose_at_laboratory(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    mine: list[Sample],
    theirs: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param mine: Samples currently carried by our robot.
    :param theirs: Samples currently carried by the opposing robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at LABORATORY.
    """
    chosen = best_owned_batch(diagnosed_samples(mine), me.storage, me.expertise, available, remaining_turns, "LABORATORY")
    if chosen and sample_fits(chosen[0], me.storage, me.expertise):
        return f"CONNECT {chosen[0].sample_id}"
    producible = ready_samples(chosen, me.storage, me.expertise)
    if producible:
        return f"CONNECT {ordered_samples(producible, me.expertise)[0].sample_id}"
    if chosen:
        return "GOTO MOLECULES"
    future = best_owned_batch(diagnosed_samples(mine), me.storage, me.expertise, released_available(available, opponent), remaining_turns - 1, "LABORATORY")
    if future:
        molecule = next_collectable_molecule(future, me.storage, me.expertise, available, planned_available, opponent.expertise)
        return "GOTO MOLECULES" if molecule is not None else "WAIT"
    eventual = best_owned_batch(
        diagnosed_samples(mine), me.storage, me.expertise, eventual_available(available, opponent), remaining_turns - 1, "LABORATORY"
    )
    if eventual:
        molecule = next_collectable_molecule(eventual, me.storage, me.expertise, available, planned_available, opponent.expertise)
        return "GOTO MOLECULES" if molecule is not None else "WAIT"
    if should_deny_endgame(me, opponent, theirs, available, remaining_turns):
        return "GOTO MOLECULES"
    cloud_batch = best_cloud_batch(me, opponent, available, planned_available, theirs, cloud, remaining_turns, "LABORATORY")
    if remaining_turns <= 14 and not mine and not cloud_batch:
        return "WAIT"
    return "GOTO DIAGNOSIS" if mine or cloud_batch else "GOTO SAMPLES"


def sample_subsets(samples: list[Sample]) -> list[list[Sample]]:
    """:param samples: Samples whose subsets should be enumerated.
    :return: Every subset of the provided samples.
    """
    return [[samples[index] for index in range(len(samples)) if mask & 1 << index] for mask in range(1 << len(samples))]


def diagnosed_samples(samples: list[Sample]) -> list[Sample]:
    """:param samples: Samples to filter.
    :return: Diagnosed samples from the provided list.
    """
    return [sample for sample in samples if sample.is_diagnosed()]


def carried_diagnosed_samples(samples: list[Sample]) -> list[Sample]:
    """:param samples: Carried samples to filter.
    :return: Diagnosed carried samples from the provided list.
    """
    return [sample for sample in samples if sample.carried_by == SELF and sample.is_diagnosed()]


def undiagnosed_samples(samples: list[Sample]) -> list[Sample]:
    """:param samples: Samples to filter.
    :return: Undiagnosed samples from the provided list.
    """
    return [sample for sample in samples if not sample.is_diagnosed()]


def batch_fits(
    samples: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
) -> bool:
    """:param samples: Candidate batch of diagnosed samples.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :return: Whether the batch can still fit under storage and current supply limits.
    """
    missing = batch_missing_vector(samples, storage, expertise)
    return sum(storage) + sum(missing) <= STORAGE_LIMIT and all(need <= pool for need, pool in zip(missing, available))


def batch_complete(samples: list[Sample], storage: tuple[int, int, int, int, int], expertise: tuple[int, int, int, int, int]) -> bool:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :return: Whether storage already covers every molecule needed by the batch.
    """
    return all(have >= need for have, need in zip(storage, batch_required_vector(samples, expertise)))


def batch_missing_vector(
    samples: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    """:param samples: Samples chosen for completion.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :return: Missing molecules per type for the batch.
    """
    return tuple(max(need - have, 0) for need, have in zip(batch_required_vector(samples, expertise), storage))


def batch_missing_cost(
    samples: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
) -> int:
    """:param samples: Samples chosen for completion.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :return: Total number of additional molecules still needed by the batch.
    """
    return sum(batch_missing_vector(samples, storage, expertise))


def batch_required_vector(samples: list[Sample], expertise: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    """:param samples: Samples whose molecule costs should be summed.
    :param expertise: Expertise already gained by our robot.
    :return: Combined A-E molecule requirements after sequential expertise reductions.
    """
    required = [0, 0, 0, 0, 0]
    progress = list(expertise)
    for sample in samples:
        for index in range(5):
            required[index] += max(sample.cost[index] - progress[index], 0)
        index = gain_index(sample.gain)
        if index >= 0:
            progress[index] += 1
    return tuple(required)


def next_needed_molecule(
    samples: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    opponent_expertise: tuple[int, int, int, int, int],
) -> str:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param opponent_expertise: Expertise already gained by the opponent.
    :return: Next molecule type to collect.
    """
    missing = batch_missing_vector(samples, storage, expertise)
    best_index = min(
        (index for index, need in enumerate(missing) if need),
        key=lambda index: (planned_available[index], available[index], -missing[index], opponent_expertise[index], index),
    )
    return MOLECULE_TYPES[best_index]


def next_collectable_molecule(
    samples: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    opponent_expertise: tuple[int, int, int, int, int],
) -> str | None:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param opponent_expertise: Expertise already gained by the opponent.
    :return: Next molecule type that can be collected immediately, if any.
    """
    missing = batch_missing_vector(samples, storage, expertise)
    collectable = [index for index, need in enumerate(missing) if need and available[index]]
    return None if not collectable else MOLECULE_TYPES[min(
        collectable,
        key=lambda index: (planned_available[index], available[index], -missing[index], opponent_expertise[index], index),
    )]


def ready_samples(
    samples: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
) -> list[Sample]:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :return: Samples that can be produced immediately.
    """
    return [sample for sample in samples if sample_fits(sample, storage, expertise)]


def sample_fits(sample: Sample, storage: tuple[int, int, int, int, int], expertise: tuple[int, int, int, int, int]) -> bool:
    """:param sample: Sample to test.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :return: Whether the sample can be produced with the given storage.
    """
    return all(have >= need for have, need in zip(storage, sample_required_vector(sample, expertise)))


def ordered_samples(samples: list[Sample], expertise: tuple[int, int, int, int, int]) -> list[Sample]:
    """:param samples: Samples to rank.
    :param expertise: Expertise already gained by our robot.
    :return: Samples sorted from best immediate value to worst.
    """
    if len(samples) < 2:
        return list(samples)
    return list(max(permutations(samples), key=lambda order: order_priority(order, expertise)))


def order_priority(samples: tuple[Sample, ...], expertise: tuple[int, int, int, int, int]) -> tuple[int, int, tuple[tuple[float, int, int, int], ...]]:
    """:param samples: Ordered candidate sequence to score.
    :param expertise: Expertise already gained by our robot.
    :return: Comparable tuple preferring low molecule demand and high-value first completions.
    """
    required = batch_required_vector(list(samples), expertise)
    return -sum(required), -max(required), tuple(sample_priority(sample, expertise) for sample in samples)


def sample_priority(sample: Sample, expertise: tuple[int, int, int, int, int]) -> tuple[float, int, int, int]:
    """:param sample: Sample to score.
    :param expertise: Expertise already gained by our robot.
    :return: Comparable tuple describing the sample priority.
    """
    required = sample_required_cost(sample, expertise)
    reward = sample.health
    index = gain_index(sample.gain)
    if index >= 0:
        progress = list(expertise)
        reward += expertise_need_value(progress, index)
        progress[index] += 1
        reward += 50 * sum(project_complete(project, progress) for project in ACTIVE_PROJECTS)
    return reward / (2 + required), reward, -required, -sample.sample_id


def sample_health_priority(sample: Sample) -> tuple[int, int, int]:
    """:param sample: Sample to score.
    :return: Comparable tuple prioritizing raw health.
    """
    return sample.health, -sum(sample.cost), -sample.sample_id


def sample_cost_priority(sample: Sample, expertise: tuple[int, int, int, int, int]) -> tuple[int, int, int]:
    """:param sample: Sample to score.
    :param expertise: Expertise already gained by our robot.
    :return: Comparable tuple prioritizing cheap samples.
    """
    return sample_required_cost(sample, expertise), -sample.health, sample.sample_id


def worst_sample(samples: list[Sample], expertise: tuple[int, int, int, int, int]) -> Sample:
    """:param samples: Samples to rank.
    :param expertise: Expertise already gained by our robot.
    :return: Lowest-priority sample from the provided list.
    """
    return min(samples, key=lambda sample: sample_priority(sample, expertise))


def sample_required_cost(sample: Sample, expertise: tuple[int, int, int, int, int]) -> int:
    """:param sample: Sample whose effective cost should be measured.
    :param expertise: Expertise already gained by our robot.
    :return: Total amount of molecules required after expertise reductions.
    """
    return sum(sample_required_vector(sample, expertise))


def sample_required_vector(sample: Sample, expertise: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    """:param sample: Sample whose effective costs should be measured.
    :param expertise: Expertise already gained by our robot.
    :return: Required molecules per type after expertise reductions.
    """
    return tuple(max(sample.cost[index] - expertise[index], 0) for index in range(5))


def batch_health(samples: list[Sample]) -> int:
    """:param samples: Samples whose health should be summed.
    :return: Total health granted by the batch.
    """
    return sum(sample.health for sample in samples)


def pressured_available(
    available: tuple[int, int, int, int, int],
    opponent: Player,
    opponent_samples: list[Sample],
) -> tuple[int, int, int, int, int]:
    """:param available: Molecules still available in the pool.
    :param opponent: Current state of the opposing robot.
    :param opponent_samples: Samples currently carried by the opponent.
    :return: Molecule availability adjusted for immediate opponent pressure.
    """
    if opponent.target != "MOLECULES" or opponent.eta > 2:
        return available
    pressure_batch = best_owned_batch(
        diagnosed_samples(opponent_samples),
        opponent.storage,
        opponent.expertise,
        available,
        TOTAL_TURNS,
        "MOLECULES",
    )
    pressure = batch_missing_vector(pressure_batch, opponent.storage, opponent.expertise)
    return tuple(max(pool - need, 0) for pool, need in zip(available, pressure))


def released_available(available: tuple[int, int, int, int, int], opponent: Player) -> tuple[int, int, int, int, int]:
    """:param available: Molecules still available in the pool.
    :param opponent: Current state of the opposing robot.
    :return: Molecule availability after an imminent opposing laboratory delivery.
    """
    return available if opponent.target != "LABORATORY" or opponent.eta > 1 else \
        tuple(min(pool + held, 5) for pool, held in zip(available, opponent.storage))


def eventual_available(available: tuple[int, int, int, int, int], opponent: Player) -> tuple[int, int, int, int, int]:
    """:param available: Molecules still available in the pool.
    :param opponent: Current state of the opposing robot.
    :return: Approximate future supply after near-term opposing deliveries release held molecules.
    """
    return tuple(min(pool + held, 5) for pool, held in zip(available, opponent.storage)) if opponent.target in {"MOLECULES", "LABORATORY"} and \
        opponent.eta <= 3 else released_available(available, opponent)


def update_projects(me_expertise: tuple[int, int, int, int, int], opponent_expertise: tuple[int, int, int, int, int]):
    """:param me_expertise: Expertise already gained by our robot.
    :param opponent_expertise: Expertise already gained by the opponent.
    """
    global ACTIVE_PROJECTS
    ACTIVE_PROJECTS = tuple(
        project for project in ACTIVE_PROJECTS
        if not project_complete(project, me_expertise) and not project_complete(project, opponent_expertise)
    )


def project_complete(project: tuple[int, int, int, int, int], expertise: tuple[int, int, int, int, int] | list[int]) -> bool:
    """:param project: Science project requirements in A-E order.
    :param expertise: Expertise totals to test against the project.
    :return: Whether the expertise satisfies the full project.
    """
    return all(have >= need for have, need in zip(expertise, project))


def gain_index(gain: str) -> int:
    """:param gain: Expertise gain token associated with a sample.
    :return: Molecule type index for the gain, or -1 when there is none.
    """
    return MOLECULE_TYPES.find(gain)


def debug(message: str):
    """:param message: Human-readable trace emitted to stderr."""
    print(message, file=sys.stderr, flush=True)


def sample_trace(samples: list[Sample]) -> str:
    """:param samples: Samples to summarize for stderr.
    :return: Compact sample state trace.
    """
    return "[" + " ".join(f"{sample.sample_id}:{sample.rank}/{sample.health}/{sample.gain}/{sample.cost}" for sample in samples) + "]"


try:
    main()
except EOFError:
    pass
