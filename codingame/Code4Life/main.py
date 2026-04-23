"""Implements an expertise-aware Silver bot for Code4Life."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations

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
            f"t={turn} at={me.target} eta={me.eta} hp={me.score} hold={me.storage} "
            f"exp={me.expertise} ids={[sample.sample_id for sample in mine]} -> {action}"
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
            return choose_at_samples(me, mine, cloud, remaining_turns)
        case "DIAGNOSIS":
            return choose_at_diagnosis(me, opponent, available, planned_available, mine, cloud, remaining_turns)
        case "MOLECULES":
            return choose_at_molecules(me, opponent, available, planned_available, mine, cloud, remaining_turns)
        case "LABORATORY":
            return choose_at_laboratory(me, available, mine, cloud, remaining_turns)
        case _:
            return "GOTO SAMPLES" if not mine else "GOTO DIAGNOSIS"


def choose_at_samples(me: Player, mine: list[Sample], cloud: list[Sample], remaining_turns: int) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at SAMPLES.
    """
    if len(mine) < desired_sample_count(me.expertise, remaining_turns):
        return f"CONNECT {sample_rank(me.expertise, remaining_turns)}"
    return "GOTO DIAGNOSIS" if mine or diagnosed_samples(cloud) else "WAIT"


def desired_sample_count(expertise: tuple[int, int, int, int, int], remaining_turns: int) -> int:
    """:param expertise: Expertise already gained by our robot.
    :param remaining_turns: Number of turns left including the current one.
    :return: Preferred number of carried samples.
    """
    total = sum(expertise)
    return 1 if remaining_turns <= 24 else 2 if remaining_turns <= 36 or total < 1 else 3


def sample_rank(expertise: tuple[int, int, int, int, int], remaining_turns: int) -> int:
    """:param expertise: Expertise already gained by our robot.
    :param remaining_turns: Number of turns left including the current one.
    :return: Rank to request from the samples machine.
    """
    total = sum(expertise)
    return 1 if remaining_turns <= 12 else 2 if remaining_turns <= 22 or total < 8 or remaining_turns <= 34 else 3


def choose_at_diagnosis(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    mine: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at DIAGNOSIS.
    """
    undiagnosed = undiagnosed_samples(mine)
    if undiagnosed:
        return f"CONNECT {undiagnosed[0].sample_id}"
    diagnosed = carried_diagnosed_samples(mine)
    chosen = best_batch(diagnosed, diagnosed_samples(cloud), me.storage, me.expertise, available, planned_available, remaining_turns)
    for sample in chosen:
        if sample.carried_by == -1:
            RECENT_DROPS.pop(sample.sample_id, None)
            return f"CONNECT {sample.sample_id}"
    if chosen and batch_complete(chosen, me.storage, me.expertise):
        return "GOTO LABORATORY"
    if chosen:
        return "GOTO MOLECULES"
    rejected = carried_diagnosed_samples(mine)
    if rejected:
        if len(mine) < desired_sample_count(me.expertise, remaining_turns):
            sample = ordered_samples(rejected, me.expertise)[0]
            index = gain_index(sample.gain)
            if index >= 0 and any(0 < project[index] - me.expertise[index] <= 2 for project in ACTIVE_PROJECTS):
                return "GOTO SAMPLES"
        future = best_owned_batch(rejected, me.storage, me.expertise, released_available(available, opponent), remaining_turns - 1, "DIAGNOSIS")
        if future:
            return "WAIT"
        sample = worst_sample(rejected, me.expertise)
        RECENT_DROPS[sample.sample_id] = TOTAL_TURNS - remaining_turns
        return f"CONNECT {sample.sample_id}"
    return "GOTO SAMPLES"


def best_batch(
    mine: list[Sample],
    cloud: list[Sample],
    storage: tuple[int, int, int, int, int],
    expertise: tuple[int, int, int, int, int],
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    remaining_turns: int,
) -> list[Sample]:
    """:param mine: Diagnosed samples currently carried by our robot.
    :param cloud: Diagnosed samples currently available in the cloud.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param remaining_turns: Number of turns left including the current one.
    :return: Highest-value finishable batch reachable from DIAGNOSIS.
    """
    best: list[Sample] = []
    best_value = batch_value([], 0, expertise, available)
    room = MAX_SAMPLES - len(mine)
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
                value = batch_value(batch, finish_time, expertise, planned_available)
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


def choose_at_molecules(
    me: Player,
    opponent: Player,
    available: tuple[int, int, int, int, int],
    planned_available: tuple[int, int, int, int, int],
    mine: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param opponent: Current state of the opposing robot.
    :param available: Molecules still available in the pool.
    :param planned_available: Molecules forecast to remain after opponent pressure.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at MOLECULES.
    """
    chosen = best_owned_batch(diagnosed_samples(mine), me.storage, me.expertise, available, remaining_turns, "MOLECULES")
    if not chosen:
        return "GOTO DIAGNOSIS" if mine or diagnosed_samples(cloud) else "GOTO SAMPLES"
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
) -> list[Sample]:
    """:param mine: Diagnosed samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :param remaining_turns: Number of turns left including the current one.
    :param module: Module from which the robot is planning the finish.
    :return: Highest-value finishable subset of carried samples.
    """
    best: list[Sample] = []
    best_value = batch_value([], 0, expertise, available)
    for batch in sample_subsets(mine):
        ordered = ordered_samples(batch, expertise)
        if not ordered or not batch_fits(ordered, storage, expertise, available):
            continue
        finish_time = finish_time_from(module, ordered, 0, storage, expertise)
        if finish_time > remaining_turns:
            continue
        value = batch_value(ordered, finish_time, expertise, available)
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
) -> tuple[float, int, int, int, int, int, tuple[int, ...]]:
    """:param samples: Candidate batch of diagnosed samples.
    :param finish_time: Turns needed to complete the batch from the current module.
    :param expertise: Expertise already gained by our robot.
    :param available: Molecules still available in the pool.
    :return: Comparable tuple describing how attractive the batch is.
    """
    reward = batch_health(samples) + batch_expertise_value(samples, expertise)
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
    active_projects = list(ACTIVE_PROJECTS)
    value = 0
    for sample in samples:
        index = gain_index(sample.gain)
        if index < 0:
            continue
        value += expertise_need_value(progress, index)
        progress[index] += 1
        completed = [project for project in active_projects if project_complete(project, progress)]
        value += 50 * len(completed)
        active_projects = [project for project in active_projects if not project_complete(project, progress)]
    return value


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
    count = sum(project[index] > expertise[index] for project in ACTIVE_PROJECTS)
    close = any(project[index] == expertise[index] + 1 for project in ACTIVE_PROJECTS)
    return 6 * count + 8 * close


def choose_at_laboratory(
    me: Player,
    available: tuple[int, int, int, int, int],
    mine: list[Sample],
    cloud: list[Sample],
    remaining_turns: int,
) -> str:
    """:param me: Current state of our robot.
    :param available: Molecules still available in the pool.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of turns left including the current one.
    :return: Command to print while standing at LABORATORY.
    """
    chosen = best_owned_batch(diagnosed_samples(mine), me.storage, me.expertise, available, remaining_turns, "LABORATORY")
    producible = ready_samples(chosen, me.storage, me.expertise)
    if producible:
        return f"CONNECT {ordered_samples(producible, me.expertise)[0].sample_id}"
    if chosen:
        return "GOTO MOLECULES"
    return "GOTO DIAGNOSIS" if mine or diagnosed_samples(cloud) else "GOTO SAMPLES"


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
    :return: Combined A-E molecule requirements after expertise reductions.
    """
    return tuple(sum(max(sample.cost[index] - expertise[index], 0) for sample in samples) for index in range(5))


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
    return sorted(samples, key=lambda sample: sample_priority(sample, expertise), reverse=True)


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


try:
    main()
except EOFError:
    pass
