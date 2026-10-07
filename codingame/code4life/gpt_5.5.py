"""Plays the Code4Life arena game with a compact heuristic bot."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations


TYPES = "ABCDE"
SAMPLES = "SAMPLES"
DIAGNOSIS = "DIAGNOSIS"
MOLECULES = "MOLECULES"
LABORATORY = "LABORATORY"


@dataclass(frozen=True)
class Robot:
    """Stores one robot turn snapshot.
    :var target: Module where the robot is located or moving.
    :var eta: Turns left before arriving.
    :var score: Current health score.
    :var storage: Molecules currently carried by type A through E.
    :var expertise: Expertise currently owned by type A through E."""

    target: str
    eta: int
    score: int
    storage: tuple[int, ...]
    expertise: tuple[int, ...]

    @property
    def storage_total(self) -> int:
        """Returns the total number of carried molecules."""
        return sum(self.storage)

    @property
    def expertise_total(self) -> int:
        """Returns the total molecule expertise."""
        return sum(self.expertise)


@dataclass(frozen=True)
class Sample:
    """Stores one sample data file visible in the game.
    :var id: Unique sample identifier.
    :var carried_by: Owner marker, where 0 is us, 1 is the opponent, and -1 is the cloud.
    :var rank: Sample rank from 1 through 3.
    :var gain: Expertise molecule awarded after research.
    :var health: Health points awarded after research, or -1 while undiagnosed.
    :var cost: Molecule cost by type A through E, or -1 values while undiagnosed."""

    id: int
    carried_by: int
    rank: int
    gain: str
    health: int
    cost: tuple[int, ...]

    @property
    def diagnosed(self) -> bool:
        """Returns whether the sample has known health and molecule costs."""
        return self.health >= 0

    def need(self, expertise: tuple[int, ...]) -> tuple[int, ...]:
        """Calculates molecules needed after expertise discounts.
        :param expertise: Current expertise by molecule type.
        :return: Required molecule counts by type."""
        return tuple(max(cost - expertise[index], 0) for index, cost in enumerate(self.cost))


@dataclass(frozen=True)
class State:
    """Stores the complete information needed to choose one command.
    :var projects: Science project expertise requirements.
    :var turn: Zero-based turn index.
    :var me: Our robot snapshot.
    :var opponent: Opponent robot snapshot.
    :var available: Molecules currently available at the terminal.
    :var samples: All visible samples."""

    projects: list[tuple[int, ...]]
    turn: int
    me: Robot
    opponent: Robot
    available: tuple[int, ...]
    samples: list[Sample]

    @property
    def mine(self) -> list[Sample]:
        """Returns samples currently carried by us."""
        return [sample for sample in self.samples if sample.carried_by == 0]

    @property
    def cloud(self) -> list[Sample]:
        """Returns diagnosed samples currently stored in the cloud."""
        return [sample for sample in self.samples if sample.carried_by == -1 and sample.diagnosed]


def main():
    """Runs the arena input loop and prints one command each turn."""
    first = sys.stdin.readline()
    if not first:
        return
    projects = [tuple(int(value) for value in sys.stdin.readline().split()) for _ in range(int(first))]
    turn = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        me = parse_robot(line)
        opponent = parse_robot(sys.stdin.readline())
        available = tuple(int(value) for value in sys.stdin.readline().split())
        samples = []
        for _ in range(int(sys.stdin.readline())):
            parts = sys.stdin.readline().split()
            samples.append(Sample(int(parts[0]), int(parts[1]), int(parts[2]), parts[3], int(parts[4]), tuple(int(value) for value in parts[5:10])))
        print(choose_command(State(projects, turn, me, opponent, available, samples)), flush=True)
        turn += 1


def parse_robot(line: str) -> Robot:
    """Parses one robot input line.
    :param line: Raw robot line from stdin.
    :return: Parsed robot snapshot."""
    parts = line.split()
    values = [int(value) for value in parts[1:]]
    return Robot(parts[0], values[0], values[1], tuple(values[2:7]), tuple(values[7:12]))


def choose_command(state: State) -> str:
    """Chooses one arena command for the current turn.
    :param state: Current game state.
    :return: Command to print."""
    if state.me.eta > 0:
        return "WAIT"
    if state.me.target == SAMPLES:
        return command_at_samples(state)
    if state.me.target == DIAGNOSIS:
        return command_at_diagnosis(state)
    if state.me.target == MOLECULES:
        return command_at_molecules(state)
    if state.me.target == LABORATORY:
        return command_at_laboratory(state)
    return f"GOTO {SAMPLES}"


def command_at_samples(state: State) -> str:
    """Chooses a command while at SAMPLES.
    :param state: Current game state.
    :return: Command to print."""
    if len(state.mine) < 3:
        return f"CONNECT {choose_rank(state)}"
    return f"GOTO {DIAGNOSIS}"


def choose_rank(state: State) -> int:
    """Chooses which rank to request at SAMPLES.
    :param state: Current game state.
    :return: Rank number to request."""
    if state.turn > 165:
        return 1 if state.me.expertise_total < 8 else 2
    if state.me.expertise_total < 3:
        return 1
    if state.me.expertise_total < 10:
        return 2
    return 3


def command_at_diagnosis(state: State) -> str:
    """Chooses a command while at DIAGNOSIS.
    :param state: Current game state.
    :return: Command to print."""
    undiagnosed = [sample for sample in state.mine if not sample.diagnosed]
    if undiagnosed:
        return f"CONNECT {undiagnosed[0].id}"
    impossible = [sample for sample in state.mine if sample.diagnosed and not is_sample_possible(state, sample)]
    if impossible:
        return f"CONNECT {min(impossible, key=lambda sample: sample_value(state, sample)).id}"
    blocked = blocked_sample(state)
    if blocked is not None:
        return f"CONNECT {blocked.id}"
    cloud_sample = best_cloud_sample(state)
    if cloud_sample is not None and \
            (len(state.mine) < 2 or sample_value(state, cloud_sample) > min(sample_value(state, sample) for sample in state.mine if sample.diagnosed)):
        return f"CONNECT {cloud_sample.id}"
    if not best_batch(state) and not best_batch(state, ignore_available=True) and state.mine:
        return f"CONNECT {min(state.mine, key=lambda sample: sample_value(state, sample)).id}"
    return route_after_diagnosis(state)


def is_sample_possible(state: State, sample: Sample) -> bool:
    """Checks whether a diagnosed sample can fit in molecule storage.
    :param state: Current game state.
    :param sample: Sample to check.
    :return: True when the sample can be completed with current expertise."""
    return sample.diagnosed and sum(sample.need(state.me.expertise)) <= 10


def sample_value(state: State, sample: Sample) -> int:
    """Scores one diagnosed sample for planning.
    :param state: Current game state.
    :param sample: Sample to evaluate.
    :return: Larger value for samples worth prioritizing."""
    need = sample.need(state.me.expertise)
    return sample.health * 10 + project_gain_value(state, sample) + sample.rank * 2 - sum(need) * 3


def project_gain_value(state: State, sample: Sample) -> int:
    """Scores how useful a sample expertise gain is for science projects.
    :param state: Current game state.
    :param sample: Sample whose gain should be scored.
    :return: Heuristic value of the expertise gain."""
    if sample.gain not in TYPES:
        return 0
    gain_index = TYPES.index(sample.gain)
    projects = [project for project in state.projects if not all(state.me.expertise[index] >= project[index] for index in range(len(TYPES))) and
        not all(state.opponent.expertise[index] >= project[index] for index in range(len(TYPES)))]
    if not projects:
        return 0
    expertise = list(state.me.expertise)
    expertise[gain_index] += 1
    completes_project = any(all(expertise[index] >= project[index] for index in range(len(TYPES))) and \
        any(state.me.expertise[index] < project[index] for index in range(len(TYPES))) for project in projects)
    if completes_project:
        return 520
    return 35 if any(state.me.expertise[gain_index] < project[gain_index] for project in projects) else 5


def best_cloud_sample(state: State) -> Sample | None:
    """Chooses a useful cloud sample to take at DIAGNOSIS.
    :param state: Current game state.
    :return: Selected sample or None."""
    room = 3 - len(state.mine)
    if room <= 0:
        return None
    candidates = [sample for sample in state.cloud if is_sample_workable(state, sample)]
    scored = sorted(candidates, key=lambda sample: sample_value(state, sample), reverse=True)
    return scored[0] if scored else None


def is_sample_workable(state: State, sample: Sample) -> bool:
    """Checks whether a sample can be productively carried now.
    :param state: Current game state.
    :param sample: Sample to check.
    :return: True when the sample fits storage and is not currently blocked."""
    if not is_sample_possible(state, sample) or is_sample_blocked(state, sample):
        return False
    missing = batch_missing(sample.need(state.me.expertise), state.me.storage)
    return state.me.storage_total + sum(missing) <= 10


def is_sample_blocked(state: State, sample: Sample) -> bool:
    """Checks whether all missing molecules for a sample are unavailable.
    :param state: Current game state.
    :param sample: Sample to check.
    :return: True when progress on this sample is currently impossible."""
    need = sample.need(state.me.expertise)
    missing = tuple(max(need[index] - state.me.storage[index], 0) for index in range(len(TYPES)))
    return any(missing) and all(state.available[index] == 0 for index, count in enumerate(missing) if count > 0)


def blocked_sample(state: State) -> Sample | None:
    """Chooses a carried sample whose missing molecules are all unavailable.
    :param state: Current game state.
    :return: Selected sample to store in the cloud, or None."""
    if best_batch(state):
        return None
    candidates = []
    for sample in state.mine:
        if not sample.diagnosed:
            continue
        if is_sample_blocked(state, sample):
            candidates.append(sample)
    return min(candidates, key=lambda sample: sample_value(state, sample)) if candidates else None


def best_batch(state: State, samples: list[Sample] | None = None, ignore_available: bool = False) -> tuple[Sample, ...]:
    """Chooses the best feasible batch from diagnosed samples.
    :param state: Current game state.
    :param samples: Optional candidate samples, defaulting to our diagnosed carried samples.
    :param ignore_available: Whether molecule terminal availability should be ignored.
    :return: Chosen samples, possibly empty."""
    candidates = [sample for sample in (state.mine if samples is None else samples) if is_sample_possible(state, sample)]
    best = ()
    best_score = -10 ** 9
    for size in range(1, min(3, len(candidates)) + 1):
        for batch in combinations(candidates, size):
            need = batch_need(batch, state.me.expertise)
            missing = batch_missing(need, state.me.storage)
            if sum(need) > 10 or state.me.storage_total + sum(missing) > 10:
                continue
            if not ignore_available and any(missing[index] > state.available[index] for index in range(len(TYPES))):
                continue
            score = sum(sample_value(state, sample) for sample in batch) + sum(sample.health for sample in batch) - sum(missing) * 4 + size * 8
            if score > best_score:
                best = batch
                best_score = score
    return best


def batch_need(samples: tuple[Sample, ...], expertise: tuple[int, ...]) -> tuple[int, ...]:
    """Calculates total molecules needed for a batch of samples.
    :param samples: Samples to research before collecting more samples.
    :param expertise: Current expertise by molecule type.
    :return: Total molecule requirement by type."""
    return tuple(sum(sample.need(expertise)[index] for sample in samples) for index in range(len(TYPES)))


def batch_missing(need: tuple[int, ...], storage: tuple[int, ...]) -> tuple[int, ...]:
    """Calculates extra molecules needed for a batch beyond current storage.
    :param need: Batch molecule requirement by type.
    :param storage: Molecules already carried by type.
    :return: Missing molecule counts by type."""
    return tuple(max(need[index] - storage[index], 0) for index in range(len(TYPES)))


def route_after_diagnosis(state: State) -> str:
    """Chooses the next module after diagnosis work is done.
    :param state: Current game state.
    :return: Movement command."""
    if best_batch(state) or best_batch(state, ignore_available=True):
        return f"GOTO {MOLECULES}"
    if len(state.mine) < 3:
        return f"GOTO {SAMPLES}"
    return f"GOTO {DIAGNOSIS}"


def command_at_molecules(state: State) -> str:
    """Chooses a command while at MOLECULES.
    :param state: Current game state.
    :return: Command to print."""
    molecule = molecule_to_collect(state)
    if molecule is not None:
        return f"CONNECT {molecule}"
    if complete_sample(state) is not None:
        return f"GOTO {LABORATORY}"
    return f"GOTO {DIAGNOSIS}"


def molecule_to_collect(state: State) -> str | None:
    """Chooses the next molecule to collect for the current best batch.
    :param state: Current game state.
    :return: Molecule type or None."""
    batch = best_batch(state) or best_batch(state, ignore_available=True)
    if not batch:
        return None
    missing = batch_missing(batch_need(batch, state.me.expertise), state.me.storage)
    choices = [index for index, count in enumerate(missing) if count > 0 and state.available[index] > 0]
    if not choices:
        return None
    index = max(choices, key=lambda item: missing[item] * 10 - state.available[item] + state.opponent.storage[item])
    return TYPES[index]


def complete_sample(state: State) -> Sample | None:
    """Chooses a carried sample that can be researched immediately.
    :param state: Current game state.
    :return: Selected sample or None."""
    ready = [sample for sample in state.mine if sample.diagnosed and
        all(sample.need(state.me.expertise)[index] <= state.me.storage[index] for index in range(len(TYPES)))]
    scored = sorted(ready, key=lambda sample: sample_value(state, sample), reverse=True)
    return scored[0] if scored else None


def command_at_laboratory(state: State) -> str:
    """Chooses a command while at LABORATORY.
    :param state: Current game state.
    :return: Command to print."""
    sample = complete_sample(state)
    if sample is not None:
        return f"CONNECT {sample.id}"
    if any(sample.diagnosed for sample in state.mine):
        return f"GOTO {MOLECULES}"
    return f"GOTO {SAMPLES}"


if __name__ == "__main__":
    main()
