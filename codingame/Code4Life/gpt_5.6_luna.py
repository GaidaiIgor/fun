"""Plays the laboratory sample-and-molecule game with a greedy adaptive policy."""

from dataclasses import dataclass
import sys


MOLECULES = "ABCDE"


@dataclass(frozen=True, slots=True)
class Player:
    """Stores one player state for the current turn.

    :var target: Module the player occupies or is travelling toward.
    :var eta: Turns remaining before the player reaches the target module.
    :var score: Current health-point score.
    :var storage: Molecule counts held by the player in A-to-E order.
    :var expertise: Expertise counts in A-to-E order.
    """

    target: str
    eta: int
    score: int
    storage: tuple[int, ...]
    expertise: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Sample:
    """Stores one sample ownership, rank, reward, and molecule costs.

    :var identifier: Unique identifier of the sample.
    :var owner: 0 for this bot, 1 for the opponent, and -1 for the cloud.
    :var rank: Sample rank from 1 through 3.
    :var gain: Expertise type index, or -1 when undiagnosed.
    :var health: Health points, or -1 when undiagnosed.
    :var costs: Base molecule costs in A-to-E order.
    """

    identifier: int
    owner: int
    rank: int
    gain: int
    health: int
    costs: tuple[int, ...]


def parse_player(line: list[bytes]) -> Player:
    """Parses a player line.

    :param line: Tokenized player input line.
    :return: Parsed player state.
    """
    return Player(line[0].decode(), int(line[1]), int(line[2]), tuple(int(value) for value in line[3:8]),
                  tuple(int(value) for value in line[8:13]))


def parse_sample(line: list[bytes]) -> Sample:
    """Parses a sample line.

    :param line: Tokenized sample input line.
    :return: Parsed sample state.
    """
    health = int(line[4])
    gain = -1 if health < 0 else MOLECULES.index(line[3].decode())
    return Sample(int(line[0]), int(line[1]), int(line[2]), gain, health,
                  tuple(int(value) for value in line[5:10]))


def is_complete(expertise: tuple[int, ...], requirement: tuple[int, ...]) -> bool:
    """Checks whether expertise satisfies a science-project requirement.

    :param expertise: Current molecule expertise in A-to-E order.
    :param requirement: Required expertise in A-to-E order.
    :return: Whether every required expertise value is met.
    """
    return all(current >= required for current, required in zip(expertise, requirement))


def missing_molecules(sample: Sample, player: Player) -> tuple[int, ...]:
    """Calculates the molecules still missing for a sample.

    :param sample: Diagnosed sample to evaluate.
    :param player: Player whose storage and expertise are considered.
    :return: Molecule counts that must still be collected in A-to-E order.
    """
    return tuple(max(0, cost - expertise - stored) for cost, expertise, stored in
                 zip(sample.costs, player.expertise, player.storage))


def can_finish(sample: Sample, player: Player) -> bool:
    """Checks whether a sample can be completed without exceeding storage.

    :param sample: Diagnosed sample to evaluate.
    :param player: Player whose storage and expertise are considered.
    :return: Whether all missing molecules fit in the remaining storage.
    """
    return sum(player.storage) + sum(missing_molecules(sample, player)) <= 10


def sample_value(sample: Sample, player: Player, opponent: Player, projects: list[tuple[int, ...]],
                 completed_projects: set[int]) -> int:
    """Ranks a diagnosed sample by health, expertise, and project value.

    :param sample: Diagnosed sample to rank.
    :param player: Player who may research the sample.
    :param opponent: Opponent whose completed projects may remove project value.
    :param projects: Active science-project requirements.
    :param completed_projects: Projects already considered completed by this bot.
    :return: Relative desirability score.
    """
    value = sample.health * 100 + sample.rank * 10
    if sample.gain < 0:
        return value
    after = list(player.expertise)
    after[sample.gain] += 1
    for index, requirement in enumerate(projects):
        if index in completed_projects or is_complete(opponent.expertise, requirement):
            continue
        if is_complete(tuple(after), requirement):
            value += 5000
        else:
            value += max(0, 20 - 5 * sum(max(0, required - current) for current, required in
                                          zip(player.expertise, requirement)))
    return value


def choose_sample(samples: list[Sample], player: Player, opponent: Player,
                  projects: list[tuple[int, ...]], completed_projects: set[int]) -> Sample | None:
    """Chooses the best feasible diagnosed sample.

    :param samples: Diagnosed samples carried by this bot.
    :param player: Player whose storage and expertise are considered.
    :param opponent: Opponent used for project-race valuation.
    :param projects: Active science-project requirements.
    :param completed_projects: Projects already considered completed by this bot.
    :return: Highest-value sample that can fit in current storage, if one exists.
    """
    candidates = [sample for sample in samples if can_finish(sample, player)]
    if not candidates:
        return None
    ready = [sample for sample in candidates if not sum(missing_molecules(sample, player))]
    return max(ready or candidates, key=lambda sample: sample_value(sample, player, opponent, projects,
                                                                     completed_projects))


def choose_cloud_sample(samples: list[Sample], player: Player, opponent: Player,
                        projects: list[tuple[int, ...]], completed_projects: set[int],
                        attempted_cloud: set[int]) -> Sample | None:
    """Chooses an eligible diagnosed sample from the cloud.

    :param samples: All samples currently visible in the game.
    :param player: Player whose capacity and expertise are considered.
    :param opponent: Opponent used for project-race valuation.
    :param projects: Active science-project requirements.
    :param completed_projects: Projects already considered completed by this bot.
    :param attempted_cloud: Cloud sample identifiers that previously failed to transfer.
    :return: Highest-value transferable cloud sample, if one exists.
    """
    candidates = [sample for sample in samples if sample.owner == -1 and sample.identifier not in attempted_cloud
                  and can_finish(sample, player)]
    return max(candidates, key=lambda sample: sample_value(sample, player, opponent, projects,
                                                            completed_projects), default=None)


def choose_rank(player: Player, sample_count: int) -> int:
    """Selects a sample rank based on accumulated expertise and open slots.

    :param player: Player whose expertise determines the suitable difficulty.
    :param sample_count: Number of samples already carried by this bot.
    :return: Rank to request from the samples module.
    """
    level = sum(player.expertise) + sample_count
    return 1 if level < 3 else 2 if level < 7 else 3


def command_for_turn(player: Player, opponent: Player, available: tuple[int, ...], samples: list[Sample],
                     projects: list[tuple[int, ...]], completed_projects: set[int],
                     attempted_cloud: set[int]) -> str:
    """Chooses one valid action for the current turn.

    :param player: Current player state for this bot.
    :param opponent: Current player state for the opponent.
    :param available: Molecules currently available at the distributor.
    :param samples: All samples currently visible in the game.
    :param projects: Active science-project requirements.
    :param completed_projects: Projects already considered completed by this bot.
    :param attempted_cloud: Cloud transfers that previously failed.
    :return: One command accepted by the game protocol.
    """
    if player.eta:
        return "WAIT"

    carried = [sample for sample in samples if sample.owner == 0]
    diagnosed = [sample for sample in carried if sample.health >= 0]
    undiagnosed = [sample for sample in carried if sample.health < 0]

    if player.target == "SAMPLES":
        if len(carried) >= 3:
            return "GOTO DIAGNOSIS"
        return f"CONNECT {choose_rank(player, len(carried))}"

    if player.target == "DIAGNOSIS":
        if undiagnosed:
            return f"CONNECT {undiagnosed[0].identifier}"
        if diagnosed and choose_sample(diagnosed, player, opponent, projects, completed_projects) is None:
            discarded = min(diagnosed, key=lambda sample: sample_value(sample, player, opponent, projects, completed_projects))
            return f"CONNECT {discarded.identifier}"
        if len(carried) < 3:
            cloud_sample = choose_cloud_sample(samples, player, opponent, projects, completed_projects,
                                               attempted_cloud)
            if cloud_sample is not None:
                attempted_cloud.add(cloud_sample.identifier)
                return f"CONNECT {cloud_sample.identifier}"
        if diagnosed:
            return "GOTO SAMPLES" if len(carried) < 3 else "GOTO MOLECULES"
        return "GOTO SAMPLES"

    if player.target == "MOLECULES":
        if not diagnosed:
            return "GOTO DIAGNOSIS"
        target = choose_sample(diagnosed, player, opponent, projects, completed_projects)
        if target is None:
            return "GOTO DIAGNOSIS"
        missing = missing_molecules(target, player)
        if not sum(missing):
            return "GOTO LABORATORY"
        molecule = next((index for index, amount in enumerate(missing) if amount and available[index]), None)
        if molecule is None:
            return "WAIT"
        return f"CONNECT {MOLECULES[molecule]}"

    if player.target == "LABORATORY":
        if not diagnosed:
            return "GOTO DIAGNOSIS"
        target = choose_sample(diagnosed, player, opponent, projects, completed_projects)
        if target is None:
            return "GOTO MOLECULES"
        if sum(missing_molecules(target, player)):
            return "GOTO MOLECULES"
        return f"CONNECT {target.identifier}"

    return "GOTO SAMPLES"


def main():
    """Reads the game stream and prints one action per turn."""
    read_line = sys.stdin.buffer.readline
    first_line = read_line()
    if not first_line:
        return
    project_count = int(first_line)
    projects = [tuple(int(value) for value in read_line().split()) for _ in range(project_count)]
    completed_projects = set[int]()
    attempted_cloud = set[int]()

    while True:
        player_line = read_line().split()
        if not player_line:
            return
        player = parse_player(player_line)
        opponent = parse_player(read_line().split())
        available = tuple(int(value) for value in read_line().split())
        sample_count_line = read_line()
        if not sample_count_line:
            return
        sample_count = int(sample_count_line)
        samples = [parse_sample(read_line().split()) for _ in range(sample_count)]

        for index, requirement in enumerate(projects):
            if is_complete(player.expertise, requirement):
                completed_projects.add(index)
        print(command_for_turn(player, opponent, available, samples, projects, completed_projects,
                               attempted_cloud), flush=True)


if __name__ == "__main__":
    main()
