"""Implements a batch-oriented bot for the early Code4Life league."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations

MAX_SAMPLES = 3
STORAGE_LIMIT = 10
BATCH_TRAVEL_TURNS = 10
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
        action = choose_action(me, mine, cloud)
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


def choose_action(me: Player, mine: list[Sample], cloud: list[Sample]) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :return: Command to print for the current turn.
    """
    if me.eta:
        return f"GOTO {me.target}"
    match me.target:
        case "DIAGNOSIS":
            return choose_at_diagnosis(me, mine, cloud)
        case "MOLECULES":
            return choose_at_molecules(me, mine)
        case "LABORATORY":
            return choose_at_laboratory(me, mine)
        case _:
            return "GOTO DIAGNOSIS" if not mine else "GOTO MOLECULES"


def choose_at_diagnosis(me: Player, mine: list[Sample], cloud: list[Sample]) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :return: Command to print while standing at DIAGNOSIS.
    """
    planned_ids = {sample.sample_id for sample in mine}
    for sample in best_batch(mine, cloud):
        if sample.sample_id not in planned_ids:
            return f"CONNECT {sample.sample_id}"
    if mine and batch_complete(mine, me.storage):
        return "GOTO LABORATORY"
    return "GOTO MOLECULES" if mine else "GOTO DIAGNOSIS"


def best_batch(mine: list[Sample], cloud: list[Sample]) -> list[Sample]:
    """:param mine: Samples currently carried by our robot.
    :param cloud: Samples currently available in the cloud.
    :return: Highest-value carried batch that still fits inside storage.
    """
    best = ordered_samples(mine)
    best_value = batch_value(best)
    for size in range(1, MAX_SAMPLES - len(mine) + 1):
        for extra in combinations(cloud, size):
            batch = ordered_samples([*mine, *extra])
            if batch_cost(batch) > STORAGE_LIMIT:
                continue
            value = batch_value(batch)
            if value > best_value:
                best = batch
                best_value = value
    return best


def batch_value(samples: list[Sample]) -> tuple[float, int, int, int, tuple[int, ...]]:
    """:param samples: Candidate batch of samples.
    :return: Comparable tuple describing how attractive the batch is.
    """
    return (
        batch_health(samples) / (BATCH_TRAVEL_TURNS + 2 * len(samples) + batch_cost(samples)) if samples else 0,
        batch_health(samples),
        -batch_cost(samples),
        len(samples),
        tuple(-sample.sample_id for sample in samples),
    )


def choose_at_molecules(me: Player, mine: list[Sample]) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :return: Command to print while standing at MOLECULES.
    """
    if not mine:
        return "GOTO DIAGNOSIS"
    if batch_complete(mine, me.storage):
        return "GOTO LABORATORY"
    molecule = next_needed_molecule(mine, me.storage)
    assert molecule is not None
    return f"CONNECT {molecule}"


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


def choose_at_laboratory(me: Player, mine: list[Sample]) -> str:
    """:param me: Current state of our robot.
    :param mine: Samples currently carried by our robot.
    :return: Command to print while standing at LABORATORY.
    """
    producible = ready_samples(mine, me.storage)
    if producible:
        return f"CONNECT {ordered_samples(producible)[0].sample_id}"
    return "GOTO MOLECULES" if mine else "GOTO DIAGNOSIS"


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
