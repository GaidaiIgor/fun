"""Implements a batch-oriented bot for the early Code4Life league."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations

MAX_SAMPLES = 3
STORAGE_LIMIT = 10
BATCH_RETURN_TURNS = 1
TOTAL_TURNS = 200
SELF = 0
MOLECULE_TYPES = "ABCDE"


@dataclass(slots=True, frozen=True)
class Player:
    """:var target: Module currently occupied or targeted by the robot.
    :var eta: Turns remaining before the robot can act at a module.
    :var score: Health points already scored by the robot.
    :var storage: Molecules currently carried by the robot in A-E order.
    """

    target: str
    eta: int
    score: int
    storage: tuple[int, int, int, int, int]


@dataclass(slots=True, frozen=True)
class Sample:
    """:var sample_id: Unique identifier of the sample.
    :var carried_by: Owner flag for the sample, or -1 while it is in the cloud.
    :var health: Health points granted when the sample is produced.
    :var cost: Molecule requirements for the sample in A-E order.
    """

    sample_id: int
    carried_by: int
    health: int
    cost: tuple[int, int, int, int, int]

    def total_cost(self) -> int:
        """:return: Total amount of molecules required by the sample."""
        return sum(self.cost)

    def fits(self, storage: tuple[int, int, int, int, int]) -> bool:
        """:param storage: Molecules currently available for the sample.
        :return: Whether the sample can be produced with the given storage.
        """
        return all(have >= need for have, need in zip(storage, self.cost))


def main():
    """Runs the game loop and prints one action per turn."""
    project_count = int(input())
    for _ in range(project_count):
        input()
    turn = 0
    while True:
        me, _, samples = read_turn()
        mine = [sample for sample in samples if sample.carried_by == SELF]
        cloud = [sample for sample in samples if sample.carried_by == -1]
        remaining_turns = TOTAL_TURNS - turn
        action = choose_action(me, mine, cloud, remaining_turns)
        debug(f"t={turn} at={me.target} eta={me.eta} hp={me.score} hold={me.storage} ids={[sample.sample_id for sample in mine]} -> {action}")
        print(action)
        turn += 1


def read_turn() -> tuple[Player, Player, list[Sample]]:
    """:return: Both players and every sample visible on the current turn."""
    me = read_player()
    opponent = read_player()
    input()
    return me, opponent, read_samples(int(input()))


def read_player() -> Player:
    """:return: Parsed state for one player."""
    inputs = input().split()
    return Player(inputs[0], int(inputs[1]), int(inputs[2]), tuple(int(value) for value in inputs[3:8]))


def read_samples(count: int) -> list[Sample]:
    """:param count: Number of sample lines to read.
    :return: Parsed samples for the current turn.
    """
    return [read_sample() for _ in range(count)]


def read_sample() -> Sample:
    """:return: Parsed state for one sample."""
    inputs = input().split()
    return Sample(int(inputs[0]), int(inputs[1]), int(inputs[4]), tuple(int(value) for value in inputs[5:10]))


def choose_action(me: Player, mine: list[Sample], cloud: list[Sample], remaining_turns: int) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of actions left including the current one.
    :return: Command to print for the current turn.
    """
    if me.eta:
        return f"GOTO {me.target}"
    match me.target:
        case "DIAGNOSIS":
            return choose_at_diagnosis(me, mine, cloud, remaining_turns)
        case "MOLECULES":
            return choose_at_molecules(me, mine, remaining_turns)
        case "LABORATORY":
            return choose_at_laboratory(me, mine, remaining_turns)
        case _:
            return "GOTO DIAGNOSIS" if not mine else "GOTO MOLECULES"


def choose_at_diagnosis(me: Player, mine: list[Sample], cloud: list[Sample], remaining_turns: int) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param remaining_turns: Number of actions left including the current one.
    :return: Command to print while standing at DIAGNOSIS.
    """
    planned_ids = {sample.sample_id for sample in mine}
    chosen = best_batch(mine, cloud, me.storage, remaining_turns)
    for sample in chosen:
        if sample.sample_id not in planned_ids:
            return f"CONNECT {sample.sample_id}"
    if chosen and batch_complete(chosen, me.storage):
        return "GOTO LABORATORY"
    return "GOTO MOLECULES" if chosen else "GOTO DIAGNOSIS"


def best_batch(
    mine: list[Sample],
    cloud: list[Sample],
    storage: tuple[int, int, int, int, int],
    remaining_turns: int,
) -> list[Sample]:
    """:param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :param storage: Molecules currently carried by our robot.
    :param remaining_turns: Number of actions left including the current one.
    :return: Highest-value finishable batch reachable from DIAGNOSIS.
    """
    best: list[Sample] = []
    best_value = batch_value([], storage, 0)
    room = MAX_SAMPLES - len(mine)
    for owned in sample_subsets(mine):
        for size in range(room + 1):
            for extra in combinations(cloud, size):
                batch = ordered_samples([*owned, *extra])
                if not batch or not batch_fits(batch, storage):
                    continue
                if diagnosis_finish_time(batch, size, storage) > remaining_turns:
                    continue
                value = batch_value(batch, storage, size)
                if value > best_value:
                    best = batch
                    best_value = value
    return best


def sample_subsets(samples: list[Sample]) -> list[list[Sample]]:
    """:param samples: Samples whose subsets should be enumerated.
    :return: Every subset of the provided samples.
    """
    return [[samples[index] for index in range(len(samples)) if mask & 1 << index] for mask in range(1 << len(samples))]


def batch_fits(samples: list[Sample], storage: tuple[int, int, int, int, int]) -> bool:
    """:param samples: Candidate batch of samples.
    :param storage: Molecules currently carried by our robot.
    :return: Whether the batch can still fit under the storage cap.
    """
    return sum(storage) + batch_missing_cost(samples, storage) <= STORAGE_LIMIT


def diagnosis_finish_time(samples: list[Sample], extra_downloads: int, storage: tuple[int, int, int, int, int]) -> int:
    """:param samples: Samples chosen for the current batch.
    :param extra_downloads: Number of additional samples still to download.
    :param storage: Molecules currently carried by our robot.
    :return: Actions needed to finish the batch from DIAGNOSIS.
    """
    missing = batch_missing_cost(samples, storage)
    return extra_downloads + len(samples) + 1 if not missing else extra_downloads + missing + len(samples) + 2


def batch_value(samples: list[Sample], storage: tuple[int, int, int, int, int], extra_downloads: int) -> tuple[float, int, int, int, int, tuple[int, ...]]:
    """:param samples: Candidate batch of samples.
    :param storage: Molecules currently carried by our robot.
    :param extra_downloads: Number of additional samples still to download.
    :return: Comparable tuple describing how attractive the batch is.
    """
    finish_time = diagnosis_finish_time(samples, extra_downloads, storage)
    return (
        batch_health(samples) / (finish_time + BATCH_RETURN_TURNS) if samples else 0,
        batch_health(samples),
        -finish_time,
        -batch_missing_cost(samples, storage),
        -len(samples),
        tuple(-sample.sample_id for sample in samples),
    )


def choose_at_molecules(me: Player, mine: list[Sample], remaining_turns: int) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param remaining_turns: Number of actions left including the current one.
    :return: Command to print while standing at MOLECULES.
    """
    chosen = best_owned_batch(mine, me.storage, remaining_turns, "MOLECULES")
    if not chosen:
        return "GOTO DIAGNOSIS"
    if batch_complete(chosen, me.storage):
        return "GOTO LABORATORY"
    molecule = next_needed_molecule(chosen, me.storage)
    assert molecule is not None
    return f"CONNECT {molecule}"


def best_owned_batch(
    mine: list[Sample],
    storage: tuple[int, int, int, int, int],
    remaining_turns: int,
    module: str,
) -> list[Sample]:
    """:param mine: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :param remaining_turns: Number of actions left including the current one.
    :param module: Module from which the robot is planning the finish.
    :return: Highest-value finishable subset of carried samples.
    """
    best: list[Sample] = []
    best_value = owned_batch_value([], storage, 0)
    for batch in sample_subsets(mine):
        if not batch or not batch_fits(batch, storage):
            continue
        finish_time = owned_finish_time(batch, storage, module)
        if finish_time > remaining_turns:
            continue
        value = owned_batch_value(batch, storage, finish_time)
        if value > best_value:
            best = ordered_samples(batch)
            best_value = value
    return best


def owned_finish_time(samples: list[Sample], storage: tuple[int, int, int, int, int], module: str) -> int:
    """:param samples: Samples chosen for completion.
    :param storage: Molecules currently carried by our robot.
    :param module: Module from which the robot is planning the finish.
    :return: Actions needed to finish the batch from the given module.
    """
    missing = batch_missing_cost(samples, storage)
    match module:
        case "MOLECULES":
            return missing + len(samples) + 1 if missing else len(samples) + 1
        case "LABORATORY":
            return missing + len(samples) + 2 if missing else len(samples)
        case _:
            return diagnosis_finish_time(samples, 0, storage)


def owned_batch_value(samples: list[Sample], storage: tuple[int, int, int, int, int], finish_time: int) -> tuple[float, int, int, int, int, tuple[int, ...]]:
    """:param samples: Candidate batch of carried samples.
    :param storage: Molecules currently carried by our robot.
    :param finish_time: Actions needed to finish the batch from the current module.
    :return: Comparable tuple describing how attractive the batch is.
    """
    return (
        batch_health(samples) / finish_time if samples else 0,
        batch_health(samples),
        -finish_time,
        -batch_missing_cost(samples, storage),
        -len(samples),
        tuple(-sample.sample_id for sample in samples),
    )


def batch_missing_cost(samples: list[Sample], storage: tuple[int, int, int, int, int]) -> int:
    """:param samples: Samples chosen for completion.
    :param storage: Molecules currently carried by our robot.
    :return: Total number of additional molecules still needed by the batch.
    """
    return sum(max(need - have, 0) for need, have in zip(batch_cost_vector(samples), storage))


def choose_at_laboratory(me: Player, mine: list[Sample], remaining_turns: int) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param remaining_turns: Number of actions left including the current one.
    :return: Command to print while standing at LABORATORY.
    """
    chosen = best_owned_batch(mine, me.storage, remaining_turns, "LABORATORY")
    producible = ready_samples(chosen, me.storage)
    if producible:
        return f"CONNECT {ordered_samples(producible)[0].sample_id}"
    return "GOTO MOLECULES" if chosen else "GOTO DIAGNOSIS"


def next_needed_molecule(samples: list[Sample], storage: tuple[int, int, int, int, int]) -> str | None:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :return: Next molecule type to collect, or nothing if the batch is complete.
    """
    remaining = list(storage)
    for sample in ordered_samples(samples):
        for index, need in enumerate(sample.cost):
            if remaining[index] < need:
                return MOLECULE_TYPES[index]
        for index, need in enumerate(sample.cost):
            remaining[index] -= need
    return None


def ready_samples(samples: list[Sample], storage: tuple[int, int, int, int, int]) -> list[Sample]:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :return: Samples that can be produced immediately.
    """
    return [sample for sample in samples if sample.fits(storage)]


def ordered_samples(samples: list[Sample]) -> list[Sample]:
    """:param samples: Samples to rank.
    :return: Samples sorted from best immediate value to worst.
    """
    return sorted(samples, key=sample_priority, reverse=True)


def sample_priority(sample: Sample) -> tuple[float, int, int, int]:
    """:param sample: Sample to score.
    :return: Comparable tuple describing the sample priority.
    """
    return sample.health / (2 + sample.total_cost()), sample.health, -sample.total_cost(), -sample.sample_id


def batch_complete(samples: list[Sample], storage: tuple[int, int, int, int, int]) -> bool:
    """:param samples: Samples currently carried by our robot.
    :param storage: Molecules currently carried by our robot.
    :return: Whether storage covers every molecule needed by the whole batch.
    """
    return all(have >= need for have, need in zip(storage, batch_cost_vector(samples)))


def batch_cost(samples: list[Sample]) -> int:
    """:param samples: Samples whose molecule costs should be summed.
    :return: Total number of molecules required by the batch.
    """
    return sum(sample.total_cost() for sample in samples)


def batch_health(samples: list[Sample]) -> int:
    """:param samples: Samples whose health should be summed.
    :return: Total health granted by the batch.
    """
    return sum(sample.health for sample in samples)


def batch_cost_vector(samples: list[Sample]) -> tuple[int, int, int, int, int]:
    """:param samples: Samples whose molecule costs should be summed per type.
    :return: Combined A-E molecule requirements for the batch.
    """
    return tuple(sum(sample.cost[index] for sample in samples) for index in range(5))


def debug(message: str):
    """:param message: Human-readable trace emitted to stderr."""
    print(message, file=sys.stderr, flush=True)


try:
    main()
except EOFError:
    pass
