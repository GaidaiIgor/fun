"""Plays Code4Life by planning compatible sample-production batches."""

from dataclasses import dataclass
from itertools import permutations
import sys


TYPES = "ABCDE"
SAMPLES = "SAMPLES"
DIAGNOSIS = "DIAGNOSIS"
MOLECULES = "MOLECULES"
LABORATORY = "LABORATORY"


@dataclass(slots=True)
class Robot:
    """Stores the visible state of one robot.

    :var target: Names the module at which the robot is or is travelling.
    :var eta: Gives the number of turns remaining before arrival.
    :var score: Gives the health points already earned.
    :var storage: Gives the carried molecule counts in ABCDE order.
    :var expertise: Gives the molecule expertise counts in ABCDE order.
    """

    target: str
    eta: int
    score: int
    storage: tuple[int, int, int, int, int]
    expertise: tuple[int, int, int, int, int]

    def stored(self) -> int:
        """Calculates the total number of molecules carried.

        :return: Gives the number of occupied molecule slots.
        """
        return sum(self.storage)


@dataclass(slots=True, frozen=True)
class Sample:
    """Stores the visible state and recipe of one sample.

    :var sample_id: Identifies the sample in CONNECT commands.
    :var carried_by: Identifies its holder, or -1 for the cloud.
    :var rank: Gives the sample rank.
    :var gain: Names the expertise acquired when producing it.
    :var health: Gives the health points earned when producing it.
    :var cost: Gives the ABCDE molecule recipe.
    """

    sample_id: int
    carried_by: int
    rank: int
    gain: str
    health: int
    cost: tuple[int, int, int, int, int]

    def diagnosed(self) -> bool:
        """Determines whether the sample has a known recipe.

        :return: Gives whether the sample was diagnosed.
        """
        return self.health >= 0


@dataclass(slots=True)
class Plan:
    """Describes samples that can be supplied and produced in one batch.

    :var order: Gives the intended laboratory production order.
    :var required: Gives all molecules needed before entering the laboratory.
    :var additional: Gives molecules still needed beyond current storage.
    :var value: Gives the heuristic value used to compare batches.
    """

    order: tuple[Sample, ...]
    required: tuple[int, int, int, int, int]
    additional: tuple[int, int, int, int, int]
    value: int


class Bot:
    """Chooses commands from visible game state and short-term batch plans.

    :var projects: Stores the initial science-project expertise requirements.
    :var turn: Counts input turns in the current game.
    :var diagnosed_by_me: Stores samples diagnosed by this robot for cloud ties.
    :var molecule_stalls: Counts consecutive unproductive molecule turns.
    """

    def __init__(self, projects: list[tuple[int, int, int, int, int]]):
        self.projects: list[tuple[int, int, int, int, int]] = projects
        self.turn: int = 0
        self.diagnosed_by_me: set[int] = set()
        self.molecule_stalls: int = 0

    def command(self, me: Robot, opponent: Robot, available: tuple[int, int, int, int, int], samples: list[Sample]) -> str:
        """Selects this turn's command from the complete visible state.

        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param available: Gives molecules available from the distributor.
        :param samples: Gives all samples visible in the game.
        :return: Gives a valid arena command.
        """
        self.turn += 1
        if me.eta > 0:
            return "WAIT"
        if me.target != MOLECULES:
            self.molecule_stalls = 0
        own = [sample for sample in samples if sample.carried_by == 0]
        opponent_samples = [sample for sample in samples if sample.carried_by == 1]
        cloud = [sample for sample in samples if sample.carried_by == -1 and sample.diagnosed()]
        pressure = self.molecule_pressure(opponent, opponent_samples)
        if me.target == SAMPLES:
            return self.at_samples(me, opponent, own, available, pressure)
        if me.target == DIAGNOSIS:
            return self.at_diagnosis(me, opponent, own, cloud, available, pressure)
        if me.target == MOLECULES:
            return self.at_molecules(me, opponent, own, available, pressure)
        if me.target == LABORATORY:
            return self.at_laboratory(me, opponent, own, available, pressure)
        return "GOTO SAMPLES"

    def at_samples(self, me: Robot, opponent: Robot, own: list[Sample], available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int]) -> str:
        """Chooses an action while stationed at SAMPLES.

        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param own: Gives samples carried by this robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :return: Gives a valid arena command.
        """
        diagnosed = [sample for sample in own if sample.diagnosed()]
        if diagnosed:
            supplied = self.best_plan(diagnosed, me, opponent, available, pressure, True)
            collectable = self.best_plan(diagnosed, me, opponent, available, pressure, False, require_collectable=True)
            if any(self.ready(sample, me.expertise, me.storage) for sample in diagnosed):
                return "GOTO LABORATORY"
            if supplied or collectable:
                return "GOTO MOLECULES"
        if len(own) < 3 and self.turn <= 181:
            return f"CONNECT {self.sample_rank(me, opponent)}"
        if own:
            return "GOTO DIAGNOSIS"
        return "WAIT"

    def at_diagnosis(self, me: Robot, opponent: Robot, own: list[Sample], cloud: list[Sample], available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int]) -> str:
        """Chooses an action while stationed at DIAGNOSIS.

        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param own: Gives samples carried by this robot.
        :param cloud: Gives diagnosed samples in the cloud.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :return: Gives a valid arena command.
        """
        undiagnosed = [sample for sample in own if not sample.diagnosed()]
        if undiagnosed:
            sample = max(undiagnosed, key=lambda item: item.rank)
            self.diagnosed_by_me.add(sample.sample_id)
            return f"CONNECT {sample.sample_id}"
        diagnosed = [sample for sample in own if sample.diagnosed()]
        plan = self.best_plan(diagnosed, me, opponent, available, pressure, False)
        supplied = self.best_plan(diagnosed, me, opponent, available, pressure, True)
        collectable = self.best_plan(diagnosed, me, opponent, available, pressure, False, require_collectable=True)
        cloud_sample, replacement = self.cloud_move(diagnosed, cloud, plan, me, opponent, available, pressure)
        if replacement:
            return f"CONNECT {replacement.sample_id}"
        if cloud_sample:
            return f"CONNECT {cloud_sample.sample_id}"
        disposable = [sample for sample in diagnosed if self.disposable(sample, me, opponent)]
        if disposable:
            sample = min(disposable, key=lambda item: self.sample_value(item, me.expertise, opponent.expertise))
            return f"CONNECT {sample.sample_id}"
        if plan:
            if any(self.ready(sample, me.expertise, me.storage) for sample in diagnosed):
                return "GOTO LABORATORY"
            if supplied or collectable:
                return "GOTO MOLECULES"
            if len(own) < 3 and self.turn <= 181:
                return "GOTO SAMPLES"
            return "WAIT"
        if me.stored() == 10 and diagnosed:
            sample = min(diagnosed, key=lambda item: self.sample_value(item, me.expertise, opponent.expertise))
            return f"CONNECT {sample.sample_id}"
        if len(own) < 3 and self.turn <= 181:
            return "GOTO SAMPLES"
        return "WAIT"

    def at_molecules(self, me: Robot, opponent: Robot, own: list[Sample], available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int]) -> str:
        """Chooses an action while stationed at MOLECULES.

        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param own: Gives samples carried by this robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :return: Gives a valid arena command.
        """
        diagnosed = [sample for sample in own if sample.diagnosed()]
        ready = [sample for sample in diagnosed if self.ready(sample, me.expertise, me.storage)]
        supplied = self.best_plan(diagnosed, me, opponent, available, pressure, True)
        collectable = self.best_plan(diagnosed, me, opponent, available, pressure, False, require_collectable=True)
        plan = supplied or collectable
        if plan and all(need <= stored for need, stored in zip(plan.required, me.storage)):
            self.molecule_stalls = 0
            return "GOTO LABORATORY"
        if plan:
            molecule = self.molecule_choice(plan, me, available, pressure)
            if molecule is not None:
                self.molecule_stalls = 0
                return f"CONNECT {TYPES[molecule]}"
        if ready:
            self.molecule_stalls = 0
            return "GOTO LABORATORY"
        if self.molecule_stalls == 0:
            self.molecule_stalls = 1
            return "WAIT"
        self.molecule_stalls = 0
        return "GOTO DIAGNOSIS"

    def at_laboratory(self, me: Robot, opponent: Robot, own: list[Sample], available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int]) -> str:
        """Chooses an action while stationed at LABORATORY.

        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param own: Gives samples carried by this robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :return: Gives a valid arena command.
        """
        diagnosed = [sample for sample in own if sample.diagnosed()]
        plan = self.best_plan(diagnosed, me, opponent, available, pressure, False, require_stored=True)
        if plan:
            return f"CONNECT {plan.order[0].sample_id}"
        if any(not sample.diagnosed() for sample in own):
            return "GOTO DIAGNOSIS"
        if diagnosed:
            plan = self.best_plan(diagnosed, me, opponent, available, pressure, False, require_collectable=True)
            if plan:
                return "GOTO MOLECULES"
            return "GOTO DIAGNOSIS"
        if self.turn <= 181:
            return "GOTO SAMPLES"
        return "WAIT"

    def sample_rank(self, me: Robot, opponent: Robot) -> int:
        """Selects a sample rank matching current expertise, race state, and time.

        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :return: Gives the rank requested from SAMPLES.
        """
        closest_project = min((self.project_remaining(project, me.expertise) for project in self.projects if self.project_active(project, me.expertise, opponent.expertise)), default=99)
        expertise = sum(me.expertise)
        if self.turn > 165 or closest_project <= 2:
            return 1 if expertise < 9 else 2
        if me.score + 40 < opponent.score and expertise >= 6:
            return 3
        if expertise < 4:
            return 1
        if expertise < 10:
            return 2
        return 3

    def cloud_move(self, own: list[Sample], cloud: list[Sample], current: Plan | None, me: Robot, opponent: Robot, available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int]) -> tuple[Sample | None, Sample | None]:
        """Finds a cloud pickup or the carried sample to release for one.

        :param own: Gives diagnosed samples carried by this robot.
        :param cloud: Gives diagnosed samples in the cloud.
        :param current: Gives the best current carried-sample plan.
        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :return: Gives a cloud sample and optional sample that must first be released.
        """
        if len(own) == 3 and opponent.target == DIAGNOSIS and opponent.eta == 0:
            return None, None
        current_value = current.value if current else -10_000
        best_sample = None
        best_replacement = None
        best_value = current_value
        for candidate in cloud:
            if self.disposable(candidate, me, opponent):
                continue
            if len(own) < 3:
                plan = self.best_plan([*own, candidate], me, opponent, available, pressure, False)
                if plan:
                    value = plan.value + (4 if candidate.sample_id in self.diagnosed_by_me else 0)
                    if value > best_value:
                        best_sample = candidate
                        best_replacement = None
                        best_value = value
                continue
            for replacement in own:
                plan = self.best_plan([sample for sample in own if sample != replacement] + [candidate], me, opponent, available, pressure, False)
                if plan:
                    value = plan.value + (4 if candidate.sample_id in self.diagnosed_by_me else 0)
                    if value > best_value:
                        best_sample = candidate
                        best_replacement = replacement
                        best_value = value
        improvement = 20 if len(own) < 3 else 35
        if best_sample and best_value >= current_value + improvement:
            return best_sample, best_replacement
        return None, None

    def best_plan(self, samples: list[Sample], me: Robot, opponent: Robot, available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int], require_available: bool, require_stored: bool = False, require_collectable: bool = False) -> Plan | None:
        """Selects the highest-value compatible carried-sample production batch.

        :param samples: Gives diagnosed samples available for production.
        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :param require_available: Requires every missing molecule to be available now.
        :param require_stored: Requires every batch molecule to be in current storage.
        :param require_collectable: Requires at least one missing molecule to be available.
        :return: Gives the best feasible batch, if one exists.
        """
        best = None
        usable = [sample for sample in samples if not self.disposable(sample, me, opponent)]
        for count in range(1, len(usable) + 1):
            for order in permutations(usable, count):
                plan = self.make_plan(order, me, opponent, available, pressure, require_available)
                if plan and require_stored and any(needed > stored for needed, stored in zip(plan.required, me.storage)):
                    continue
                if plan and require_collectable and not self.collectable(plan, me, available):
                    continue
                if plan and (best is None or plan.value > best.value):
                    best = plan
        return best

    def make_plan(self, order: tuple[Sample, ...], me: Robot, opponent: Robot, available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int], require_available: bool) -> Plan | None:
        """Builds one ordered batch while accounting for acquired expertise.

        :param order: Gives samples in intended laboratory production order.
        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :param require_available: Requires every missing molecule to be available now.
        :return: Gives the feasible batch plan, if its storage fits.
        """
        expertise = list(me.expertise)
        required = [0, 0, 0, 0, 0]
        value = 0
        for sample in order:
            cost = [max(0, needed - known) for needed, known in zip(sample.cost, expertise)]
            required = [total + needed for total, needed in zip(required, cost)]
            value += sample.health * 12 - sum(cost) * 3 + self.project_bonus(sample.gain, tuple(expertise), opponent.expertise)
            expertise[TYPES.index(sample.gain)] += 1
        additional = tuple(max(0, needed - stored) for needed, stored in zip(required, me.storage))
        if me.stored() + sum(additional) > 10:
            return None
        if require_available and any(needed > stock for needed, stock in zip(additional, available)):
            return None
        value += (len(order) - 1) * 12
        for needed, stock, demand in zip(additional, available, pressure):
            value -= needed * max(0, demand - stock) * 2
            if needed and needed >= stock:
                value -= needed * 4
        return Plan(order, tuple(required), additional, value)

    def molecule_choice(self, plan: Plan, me: Robot, available: tuple[int, int, int, int, int], pressure: tuple[int, int, int, int, int]) -> int | None:
        """Selects the next molecule required by a planned batch.

        :param plan: Describes the selected production batch.
        :param me: Describes this robot.
        :param available: Gives molecules available from the distributor.
        :param pressure: Gives estimated opposing demand by molecule type.
        :return: Gives the ABCDE index of a molecule to take, if possible.
        """
        if me.stored() >= 10:
            return None
        first_cost = tuple(max(0, needed - known) for needed, known in zip(plan.order[0].cost, me.expertise))
        choices = [index for index in range(5) if plan.required[index] > me.storage[index] and available[index] > 0]
        if not choices:
            return None
        return max(choices, key=lambda index: (10_000 if first_cost[index] > me.storage[index] else 0) + pressure[index] * 20 + (5 - available[index]) * 10 + plan.required[index] - me.storage[index])

    def collectable(self, plan: Plan, me: Robot, available: tuple[int, int, int, int, int]) -> bool:
        """Determines whether a planned batch has a molecule that can be taken now.

        :param plan: Describes the selected production batch.
        :param me: Describes this robot.
        :param available: Gives molecules available from the distributor.
        :return: Gives whether a valid useful MOLECULES CONNECT is available.
        """
        return me.stored() < 10 and any(needed > stored and stock > 0 for needed, stored, stock in zip(plan.required, me.storage, available))

    def sample_value(self, sample: Sample, expertise: tuple[int, int, int, int, int], opponent_expertise: tuple[int, int, int, int, int]) -> int:
        """Scores one diagnosed sample outside a specific production order.

        :param sample: Gives the sample to score.
        :param expertise: Gives this robot's current molecule expertise.
        :param opponent_expertise: Gives the opposing robot's molecule expertise.
        :return: Gives the sample's heuristic value.
        """
        cost = sum(max(0, needed - known) for needed, known in zip(sample.cost, expertise))
        return sample.health * 12 - cost * 3 + self.project_bonus(sample.gain, expertise, opponent_expertise)

    def disposable(self, sample: Sample, me: Robot, opponent: Robot) -> bool:
        """Determines whether a sample is too weak to keep without an urgent project.

        :param sample: Gives the sample to assess.
        :param me: Describes this robot.
        :param opponent: Describes the opposing robot.
        :return: Gives whether the sample should be released when possible.
        """
        return sample.health <= 1 and self.project_bonus(sample.gain, me.expertise, opponent.expertise) < 150

    def project_bonus(self, gain: str, expertise: tuple[int, int, int, int, int], opponent_expertise: tuple[int, int, int, int, int]) -> int:
        """Scores expertise progress toward science projects.

        :param gain: Names the expertise that a sample would grant.
        :param expertise: Gives this robot's current molecule expertise.
        :param opponent_expertise: Gives the opposing robot's molecule expertise.
        :return: Gives the project-progress value of the expertise gain.
        """
        index = TYPES.index(gain)
        bonus = 0
        for project in self.projects:
            if not self.project_active(project, expertise, opponent_expertise) or project[index] <= expertise[index]:
                continue
            remaining = self.project_remaining(project, expertise)
            if remaining == 1:
                bonus += 600
            else:
                bonus += 300 // remaining
                if self.project_remaining(project, opponent_expertise) <= remaining:
                    bonus += 50 // remaining
        return bonus

    def project_active(self, project: tuple[int, int, int, int, int], expertise: tuple[int, int, int, int, int], opponent_expertise: tuple[int, int, int, int, int]) -> bool:
        """Determines whether neither robot has already completed a project.

        :param project: Gives the project's ABCDE expertise requirements.
        :param expertise: Gives this robot's molecule expertise.
        :param opponent_expertise: Gives the opposing robot's molecule expertise.
        :return: Gives whether the project can still reward this robot.
        """
        return self.project_remaining(project, expertise) > 0 and self.project_remaining(project, opponent_expertise) > 0

    def project_remaining(self, project: tuple[int, int, int, int, int], expertise: tuple[int, int, int, int, int]) -> int:
        """Calculates expertise still needed to complete a science project.

        :param project: Gives the project's ABCDE expertise requirements.
        :param expertise: Gives a robot's molecule expertise.
        :return: Gives the sum of outstanding expertise requirements.
        """
        return sum(max(0, needed - known) for needed, known in zip(project, expertise))

    def molecule_pressure(self, opponent: Robot, samples: list[Sample]) -> tuple[int, int, int, int, int]:
        """Estimates the molecules the opposing robot is likely to request next.

        :param opponent: Describes the opposing robot.
        :param samples: Gives samples carried by the opposing robot.
        :return: Gives estimated ABCDE demand after their current storage.
        """
        demand = [0, 0, 0, 0, 0]
        for sample in samples:
            if sample.diagnosed():
                demand = [total + max(0, cost - known) for total, cost, known in zip(demand, sample.cost, opponent.expertise)]
        return tuple(max(0, needed - stored) for needed, stored in zip(demand, opponent.storage))

    def ready(self, sample: Sample, expertise: tuple[int, int, int, int, int], storage: tuple[int, int, int, int, int]) -> bool:
        """Determines whether current molecules and expertise can produce a sample.

        :param sample: Gives the diagnosed sample to test.
        :param expertise: Gives current molecule expertise.
        :param storage: Gives current molecule storage.
        :return: Gives whether the laboratory can produce the sample now.
        """
        return all(cost <= known + held for cost, known, held in zip(sample.cost, expertise, storage))


def read_robot(fields: list[bytes]) -> Robot:
    """Parses one robot input line.

    :param fields: Gives whitespace-separated input tokens.
    :return: Gives the parsed robot state.
    """
    values = [int(value) for value in fields[1:]]
    return Robot(fields[0].decode(), values[0], values[1], tuple(values[2:7]), tuple(values[7:12]))


def read_sample(fields: list[bytes]) -> Sample:
    """Parses one sample input line.

    :param fields: Gives whitespace-separated input tokens.
    :return: Gives the parsed sample state.
    """
    return Sample(int(fields[0]), int(fields[1]), int(fields[2]), fields[3].decode(), int(fields[4]), tuple(int(value) for value in fields[5:10]))


def main():
    """Runs the arena input loop and writes one command per turn."""
    reader = sys.stdin.buffer
    project_count_line = reader.readline()
    if not project_count_line:
        return
    projects = [tuple(int(value) for value in reader.readline().split()) for _ in range(int(project_count_line))]
    bot = Bot(projects)
    while line := reader.readline():
        if not line.strip():
            continue
        me = read_robot(line.split())
        opponent = read_robot(reader.readline().split())
        available = tuple(int(value) for value in reader.readline().split())
        samples = [read_sample(reader.readline().split()) for _ in range(int(reader.readline()))]
        print(bot.command(me, opponent, available, samples), flush=True)


if __name__ == "__main__":
    main()
