"""Plays the Roche Code4Life arena game with deadline-aware batch planning."""

from dataclasses import dataclass
from itertools import permutations
from sys import stderr, stdin
from typing import TextIO


type Vector = tuple[int, int, int, int, int]

MOLECULE_NAMES = ("A", "B", "C", "D", "E")
MOLECULE_INDEX = {name: index for index, name in enumerate(MOLECULE_NAMES)} | {"0": -1}
GAME_TURNS = 200


@dataclass(frozen=True, slots=True)
class Robot:
    """Stores the authoritative state of one robot for a turn.
    :var target: Names the module occupied or approached by the robot.
    :var eta: Gives the turns remaining before the robot reaches its target.
    :var score: Gives accumulated robot health points.
    :var storage: Counts carried molecules by type.
    :var expertise: Counts permanent molecule expertise by type."""

    target: str
    eta: int
    score: int
    storage: Vector
    expertise: Vector


@dataclass(frozen=True, slots=True)
class Sample:
    """Stores the location, reward, and molecule profile of one sample.
    :var sample_id: Identifies the sample in CONNECT commands.
    :var carried_by: Identifies its robot carrier, or -1 for the cloud.
    :var rank: Gives the sample acquisition rank.
    :var gain: Identifies the expertise type awarded by production.
    :var health: Gives the health points awarded by production, or -1 while undiagnosed.
    :var cost: Gives required molecules by type, or -1 values while undiagnosed."""

    sample_id: int
    carried_by: int
    rank: int
    gain: int
    health: int
    cost: Vector


@dataclass(frozen=True, slots=True)
class Frame:
    """Stores all authoritative input for one game turn.
    :var me: Describes the IGPro robot.
    :var opponent: Describes the opposing robot.
    :var available: Counts molecules currently offered by the distributor.
    :var samples: Contains every sample currently in play."""

    me: Robot
    opponent: Robot
    available: Vector
    samples: tuple[Sample, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    """Describes an ordered batch that can share one trip to the laboratory.
    :var samples: Orders samples by their intended production sequence.
    :var required: Counts total molecules consumed by the ordered batch.
    :var pickups: Counts additional molecules still needed from the distributor.
    :var reward: Gives medicine points plus science projects reachable before the opponent.
    :var value: Adds strategic expertise value to the estimated reward.
    :var turns: Estimates turns needed to finish the batch from the current module."""

    samples: tuple[Sample, ...]
    required: Vector
    pickups: Vector
    reward: int
    value: int
    turns: int


class Bot:
    """Chooses one legal arena command from each authoritative frame.
    :var projects: Stores the three science-project expertise requirements.
    :var turn: Counts frames already processed in the current game.
    :var diagnosed_by_me: Tracks samples for which IGPro wins a simultaneous cloud race.
    :var rejected_until: Prevents recently discarded samples from causing diagnosis loops.
    :var molecule_waits: Limits waits for molecules visibly held by the opponent.
    :var release_abandoned_until: Prevents repeated travel toward an opponent that refuses to produce.
    :var project_eta_cache: Caches opponent science-project timing within one frame."""

    projects: tuple[Vector, ...]
    turn: int
    diagnosed_by_me: set[int]
    rejected_until: dict[int, int]
    molecule_waits: int
    release_abandoned_until: int
    project_eta_cache: dict[Vector, int]

    def __init__(self, projects: tuple[Vector, ...]):
        """Initializes persistent knowledge for one game.
        :param projects: Supplies science-project expertise requirements."""
        self.projects = projects
        self.turn = 0
        self.diagnosed_by_me = set()
        self.rejected_until = {}
        self.molecule_waits = 0
        self.release_abandoned_until = 0
        self.project_eta_cache = {}

    def decide(self, frame: Frame) -> str:
        """Chooses the command for the current authoritative frame.
        :param frame: Supplies both robots, molecule availability, and samples.
        :return: Provides one legal game command."""
        self.project_eta_cache.clear()
        if frame.me.eta > 0:
            self.molecule_waits = 0
            return "WAIT"
        owned = [sample for sample in frame.samples if sample.carried_by == 0]
        if frame.me.target == "START_POS":
            return "GOTO SAMPLES"
        if frame.me.target == "SAMPLES":
            return self._at_samples(frame, owned)
        if frame.me.target == "DIAGNOSIS":
            return self._at_diagnosis(frame, owned)
        if frame.me.target == "MOLECULES":
            return self._at_molecules(frame, owned)
        if frame.me.target == "LABORATORY":
            return self._at_laboratory(frame, owned)
        raise ValueError(f"Unknown module {frame.me.target}")

    def _at_samples(self, frame: Frame, owned: list[Sample]) -> str:
        """Chooses whether to draw another sample or visit diagnosis.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a SAMPLES-module command."""
        self.molecule_waits = 0
        if len(owned) == 3:
            return "GOTO DIAGNOSIS"
        expertise = sum(frame.me.expertise)
        ranks = (1, 1, 1) if expertise < 3 else (2, 2, 1) if expertise < 6 else (3, 2, 2) if expertise < 10 else (3, 3, 3)
        rank = ranks[len(owned)]
        if expertise < 6 and sum(max(amount, 0) for amount in frame.available) < 8:
            rank = max(1, rank - 1)
        if GAME_TURNS - self.turn < 35 and frame.me.score < frame.opponent.score:
            rank = min(3, rank + 1)
        unknown_after_draw = sum(sample.health < 0 for sample in owned) + 1
        minimum_pickups = max(0, (3, 5, 7)[rank - 1] - expertise - sum(frame.me.storage))
        minimum_finish = 9 + unknown_after_draw if minimum_pickups == 0 else 11 + unknown_after_draw + minimum_pickups
        cloud_plan = None
        if not owned:
            cloud = [sample for sample in frame.samples
                     if sample.carried_by == -1 and sample.health >= 0 and self.rejected_until.get(sample.sample_id, 0) <= self.turn]
            cloud_plan = self._best_plan(cloud, frame, "SAMPLES_CLOUD", True)
            if cloud_plan is None:
                potential = self._best_plan(cloud, frame, "SAMPLES_CLOUD", False)
                cloud_plan = potential if potential is not None \
                    and self._opponent_will_release(frame, potential) else None
        if cloud_plan is not None \
                and (GAME_TURNS - self.turn < minimum_finish or cloud_plan.reward >= (10, 20, 30)[rank - 1] or cloud_plan.value >= 35):
            return "GOTO DIAGNOSIS"
        if GAME_TURNS - self.turn < minimum_finish:
            return "GOTO DIAGNOSIS" if owned else "WAIT"
        return f"CONNECT {rank}"

    def _at_diagnosis(self, frame: Frame, owned: list[Sample]) -> str:
        """Diagnoses unknowns and exchanges samples for the best visible portfolio.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a DIAGNOSIS-module command."""
        self.molecule_waits = 0
        ready = self._ready_samples(owned, frame.me)
        if ready and GAME_TURNS - self.turn <= 4 + len(ready):
            return "GOTO LABORATORY"
        unknown = [sample for sample in owned if sample.health < 0]
        if unknown:
            sample = min(unknown, key=lambda item: item.sample_id)
            self.diagnosed_by_me.add(sample.sample_id)
            return f"CONNECT {sample.sample_id}"
        cloud = [sample for sample in frame.samples
                 if sample.carried_by == -1 and sample.health >= 0 and self.rejected_until.get(sample.sample_id, 0) <= self.turn]
        candidates = owned + cloud
        plan = self._best_plan(candidates, frame, "DIAGNOSIS", True)
        if plan is None:
            potential = self._best_plan(candidates, frame, "DIAGNOSIS", False)
            plan = potential if potential is not None and self._opponent_will_release(frame, potential) else None
        if plan is not None:
            target_ids = {sample.sample_id for sample in plan.samples}
            desired_cloud = [sample for sample in plan.samples if sample.carried_by == -1]
            if desired_cloud:
                if len(owned) < 3:
                    return f"CONNECT {desired_cloud[0].sample_id}"
                rejected = [sample for sample in owned if sample.sample_id not in target_ids]
                sample = min(rejected, key=lambda item: self._candidate_score(item, frame))
                target = desired_cloud[0]
                if frame.opponent.target != "DIAGNOSIS" or frame.opponent.eta > 1 \
                        or frame.opponent.eta == 1 and target.sample_id in self.diagnosed_by_me:
                    self.rejected_until[sample.sample_id] = self.turn + 8
                    return f"CONNECT {sample.sample_id}"
        owned_plan = self._best_plan(owned, frame, "DIAGNOSIS", True)
        if owned_plan is not None:
            if self._is_lemon(owned_plan):
                if len(owned) < 3:
                    return "GOTO SAMPLES"
                sample = min(owned, key=lambda item: self._candidate_score(item, frame))
                self.rejected_until[sample.sample_id] = self.turn + 8
                return f"CONNECT {sample.sample_id}"
            if not any(owned_plan.pickups) or GAME_TURNS - self.turn <= 12 and self._ready_samples(owned, frame.me):
                return "GOTO LABORATORY"
            return "GOTO MOLECULES"
        owned_plan = self._best_plan(owned, frame, "DIAGNOSIS", False)
        if owned_plan is not None and self._opponent_will_release(frame, owned_plan):
            return "GOTO MOLECULES"
        if GAME_TURNS - self.turn <= 8:
            return "GOTO LABORATORY" if self._ready_samples(owned, frame.me) else "WAIT"
        if len(owned) == 3:
            sample = min(owned, key=lambda item: self._candidate_score(item, frame))
            self.rejected_until[sample.sample_id] = self.turn + 8
            return f"CONNECT {sample.sample_id}"
        return "GOTO SAMPLES"

    def _at_molecules(self, frame: Frame, owned: list[Sample]) -> str:
        """Collects the scarcest required molecule for the best current batch.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a MOLECULES-module command."""
        plan = self._best_plan(owned, frame, "MOLECULES", True)
        if plan is None:
            potential = self._best_plan(owned, frame, "MOLECULES", False)
            plan = potential if potential is not None and self._opponent_will_release(frame, potential) else None
        if plan is not None:
            if self._is_lemon(plan):
                self.molecule_waits = 0
                return "GOTO SAMPLES" if len(owned) < 3 else "GOTO DIAGNOSIS"
            choices = [index for index, amount in enumerate(plan.pickups) if amount > 0 and frame.available[index] > 0]
            if choices:
                self.molecule_waits = 0
                opponent_need = self._opponent_need(frame)
                first = plan.samples[0]
                first_missing = tuple(max(first.cost[index] - frame.me.expertise[index] - frame.me.storage[index], 0) for index in range(5))
                index = min(choices, key=lambda item: (max(frame.available[item], 0) - plan.pickups[item] - opponent_need[item],
                                                       0 if first_missing[item] else 1, frame.available[item], -plan.pickups[item], item))
                return f"CONNECT {MOLECULE_NAMES[index]}"
            if not any(plan.pickups):
                self.molecule_waits = 0
                return "GOTO LABORATORY"
        if self._ready_samples(owned, frame.me):
            self.molecule_waits = 0
            return "GOTO LABORATORY"
        if plan is not None and self.molecule_waits < 3 and plan.turns < GAME_TURNS - self.turn \
                and self._opponent_will_release(frame, plan):
            self.molecule_waits += 1
            return "WAIT"
        if plan is not None and self._opponent_will_release(frame, plan):
            self.release_abandoned_until = self.turn + 8
        self.molecule_waits = 0
        if GAME_TURNS - self.turn <= 5:
            return "WAIT"
        return "GOTO DIAGNOSIS" if owned else "GOTO SAMPLES"

    def _at_laboratory(self, frame: Frame, owned: list[Sample]) -> str:
        """Produces the best ready medicine or starts the next profitable route.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a LABORATORY-module command."""
        self.molecule_waits = 0
        ready = self._ready_samples(owned, frame.me)
        if ready:
            return f"CONNECT {self._best_laboratory_order(owned, frame)[0].sample_id}"
        plan = self._best_plan(owned, frame, "LABORATORY", True)
        if plan is not None:
            if self._is_lemon(plan):
                return "GOTO SAMPLES" if len(owned) < 3 else "GOTO DIAGNOSIS"
            return "GOTO MOLECULES"
        plan = self._best_plan(owned, frame, "LABORATORY", False)
        if plan is not None and self._opponent_will_release(frame, plan):
            if self._is_lemon(plan):
                return "GOTO SAMPLES" if len(owned) < 3 else "GOTO DIAGNOSIS"
            return "GOTO MOLECULES"
        if owned:
            return "GOTO DIAGNOSIS"
        cloud = [sample for sample in frame.samples
                 if sample.carried_by == -1 and sample.health >= 0 and self.rejected_until.get(sample.sample_id, 0) <= self.turn]
        plan = self._best_plan(cloud, frame, "CLOUD", True)
        if plan is None:
            potential = self._best_plan(cloud, frame, "CLOUD", False)
            plan = potential if potential is not None and self._opponent_will_release(frame, potential) else None
        expertise = sum(frame.me.expertise)
        rank = 1 if expertise < 3 else 2 if expertise < 6 else 3
        minimum_pickups = max(0, (3, 5, 7)[rank - 1] - expertise - sum(frame.me.storage))
        minimum_finish = 13 if minimum_pickups == 0 else 15 + minimum_pickups
        if plan is not None \
                and (plan.reward >= 20 or plan.value >= 25 or GAME_TURNS - self.turn < minimum_finish):
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES" if GAME_TURNS - self.turn >= minimum_finish else "WAIT"

    def _best_laboratory_order(self, owned: list[Sample], frame: Frame) -> tuple[Sample, ...]:
        """Finds the highest-value sequence executable from molecules already held.
        :param owned: Supplies samples carried by IGPro.
        :param frame: Supplies current storage, expertise, projects, and remaining time.
        :return: Provides the best nonempty laboratory production order."""
        diagnosed = [sample for sample in owned if sample.health >= 0]
        best = ()
        best_key = None
        for size in range(1, min(3, len(diagnosed), GAME_TURNS - self.turn) + 1):
            for order in permutations(diagnosed, size):
                storage = list(frame.me.storage)
                expertise = list(frame.me.expertise)
                claimed = [self._dominates(frame.me.expertise, project) or self._dominates(frame.opponent.expertise, project)
                           for project in self.projects]
                reward = 0
                for step, sample in enumerate(order, 1):
                    need = tuple(max(sample.cost[index] - expertise[index], 0) for index in range(5))
                    if any(need[index] > storage[index] for index in range(5)):
                        break
                    for index in range(5):
                        storage[index] -= need[index]
                    reward += sample.health
                    expertise[sample.gain] += 1
                    for index, project in enumerate(self.projects):
                        if not claimed[index] and self._dominates(tuple(expertise), project):
                            claimed[index] = True
                            if step <= self._opponent_project_eta(frame, project):
                                reward += 50
                else:
                    expertise_value = size * (3 if GAME_TURNS - self.turn > 50 else 1 if GAME_TURNS - self.turn > 20 else 0)
                    progress = 0
                    for index, project in enumerate(self.projects):
                        if claimed[index] or self._dominates(frame.opponent.expertise, project):
                            continue
                        before = self._project_deficit(frame.me.expertise, project)
                        after = self._project_deficit(tuple(expertise), project)
                        opponent = self._project_deficit(frame.opponent.expertise, project)
                        progress += (before - after) * (6 if before <= opponent + 1 else 3)
                    key = (reward + expertise_value + progress, reward, size, sum(storage), tuple(-sample.sample_id for sample in order))
                    if best_key is None or key > best_key:
                        best, best_key = order, key
        return best

    def _best_plan(self, candidates: list[Sample], frame: Frame, origin: str, require_available: bool) -> Plan | None:
        """Finds the strongest ordered batch among visible diagnosed samples.
        :param candidates: Supplies samples eligible for the batch.
        :param frame: Supplies robots and current molecule availability.
        :param origin: Identifies the route whose turn cost should be estimated.
        :param require_available: Requires every planned pickup to be immediately available when true.
        :return: Provides the best feasible ordered batch, or None when no batch fits."""
        diagnosed = [sample for sample in candidates if sample.health >= 0]
        owned = sorted((sample for sample in diagnosed if sample.carried_by == 0), key=lambda item: item.sample_id)
        cloud = sorted((sample for sample in diagnosed if sample.carried_by == -1), key=lambda item: (-self._candidate_score(item, frame), item.sample_id))
        eligible = owned + cloud[:max(0, 12 - len(owned))]
        best = None
        best_key = None
        for size in range(1, min(3, len(eligible)) + 1):
            for order in permutations(eligible, size):
                plan = self._evaluate_order(order, frame, origin, require_available, len(owned))
                if plan is None:
                    continue
                rate = plan.value * 1000 // plan.turns
                stable = (sum(sample.carried_by == 0 for sample in order), -sum(plan.pickups), len(order), tuple(-sample.sample_id for sample in order))
                key = (plan.reward, rate, plan.value, *stable) if GAME_TURNS - self.turn <= 30 else (rate, plan.reward, plan.value, *stable)
                if best_key is None or key > best_key:
                    best, best_key = plan, key
        return best

    def _candidate_score(self, sample: Sample, frame: Frame) -> int:
        """Estimates standalone sample value for cloud prefiltering and rejection.
        :param sample: Supplies the diagnosed sample to assess.
        :param frame: Supplies expertise, projects, and opponent progress.
        :return: Provides a larger-is-better strategic score."""
        effective_cost = sum(max(sample.cost[index] - frame.me.expertise[index], 0) for index in range(5))
        score = sample.health * 10 - effective_cost * 4
        if GAME_TURNS - self.turn > 50:
            score += 24
        for project in self.projects:
            if not self._dominates(frame.me.expertise, project) and not self._dominates(frame.opponent.expertise, project) \
                    and frame.me.expertise[sample.gain] < project[sample.gain]:
                score += min(50, 100 // self._project_deficit(frame.me.expertise, project))
        if sample.carried_by == 0:
            score += 8
        if sample.sample_id in self.diagnosed_by_me:
            score += 2
        return score

    def _evaluate_order(self, order: tuple[Sample, ...], frame: Frame, origin: str, require_available: bool, owned_count: int) -> Plan | None:
        """Simulates expertise, molecule use, projects, capacity, and route time for one batch order.
        :param order: Supplies the proposed laboratory production order.
        :param frame: Supplies current robot and molecule state.
        :param origin: Identifies the route whose turn cost should be estimated.
        :param require_available: Requires all planned pickups to exist now when true.
        :param owned_count: Gives the number of samples already carried at diagnosis.
        :return: Provides a feasible batch plan, or None when capacity, supply, or time prevents it."""
        expertise = list(frame.me.expertise)
        required = [0, 0, 0, 0, 0]
        claimed = [self._dominates(frame.me.expertise, project) or self._dominates(frame.opponent.expertise, project) for project in self.projects]
        completed_projects = {}
        reward = 0
        for step, sample in enumerate(order, 1):
            for index in range(5):
                required[index] += max(sample.cost[index] - expertise[index], 0)
            reward += sample.health
            expertise[sample.gain] += 1
            for index, project in enumerate(self.projects):
                if not claimed[index] and self._dominates(tuple(expertise), project):
                    claimed[index] = True
                    completed_projects[index] = step
        pickups = tuple(max(required[index] - frame.me.storage[index], 0) for index in range(5))
        pre_lab = tuple(max(required[index], frame.me.storage[index]) for index in range(5))
        if sum(pre_lab) > 10:
            return None
        if require_available and any(pickups[index] > max(frame.available[index], 0) for index in range(5)):
            return None
        if not require_available \
                and any(pickups[index] > max(frame.available[index] + frame.opponent.storage[index], 0) for index in range(5)):
            return None
        cloud_count = sum(sample.carried_by == -1 for sample in order)
        switches = cloud_count + max(0, owned_count + cloud_count - 3) \
            if origin in {"DIAGNOSIS", "CLOUD", "SAMPLES_CLOUD"} else 0
        pickup_count = sum(pickups)
        route = {"DIAGNOSIS": (4, 6), "MOLECULES": (3, 3), "CLOUD": (8, 10), "SAMPLES_CLOUD": (7, 9),
                 "LABORATORY": (6, 6)}[origin][pickup_count > 0]
        turns = route + pickup_count + len(order) + switches
        if turns > GAME_TURNS - self.turn:
            return None
        preparation = route + pickup_count + switches
        reward += sum(50 for index, step in completed_projects.items()
                      if preparation + step <= self._opponent_project_eta(frame, self.projects[index]))
        expertise_value = len(order) * (3 if GAME_TURNS - self.turn > 50 else 1 if GAME_TURNS - self.turn > 20 else 0)
        progress = 0
        for index, project in enumerate(self.projects):
            if claimed[index] or self._dominates(frame.me.expertise, project) or self._dominates(frame.opponent.expertise, project):
                continue
            before = self._project_deficit(frame.me.expertise, project)
            after = self._project_deficit(tuple(expertise), project)
            opponent = self._project_deficit(frame.opponent.expertise, project)
            progress += (before - after) * (6 if before <= opponent + 1 else 3)
        leftovers = sum(pre_lab[index] - required[index] for index in range(5))
        ownership = sum(sample.carried_by == -1 and sample.sample_id in self.diagnosed_by_me for sample in order)
        batching = max(0, len(order) - 1) * 4
        return Plan(order, tuple(required), pickups, reward,
                    reward + expertise_value + progress + ownership + batching - leftovers * 2, turns)

    def _ready_samples(self, samples: list[Sample], robot: Robot) -> list[Sample]:
        """Selects diagnosed samples immediately producible from current storage.
        :param samples: Supplies samples carried by the robot.
        :param robot: Supplies robot storage and expertise.
        :return: Provides every sample whose effective costs are currently held."""
        return [sample for sample in samples if sample.health >= 0
                and all(max(sample.cost[index] - robot.expertise[index], 0) <= robot.storage[index] for index in range(5))]

    def _is_lemon(self, plan: Plan) -> bool:
        """Identifies an early maximum-cost one-point sample with no strategic progress.
        :param plan: Supplies the best currently available batch.
        :return: Indicates whether drawing alternatives is preferable to starting the batch."""
        return GAME_TURNS - self.turn > 50 and plan.reward == 1 and plan.value <= 4 and sum(plan.pickups) >= 5

    def _opponent_need(self, frame: Frame) -> Vector:
        """Estimates molecules the opponent can still consume from visible samples.
        :param frame: Supplies the opponent, samples, and current storage.
        :return: Provides an aggregate missing-molecule estimate by type."""
        samples = [sample for sample in frame.samples if sample.carried_by == 1 and sample.health >= 0]
        return tuple(max(sum(max(sample.cost[index] - frame.opponent.expertise[index], 0) for sample in samples)
                         - frame.opponent.storage[index], 0) for index in range(5))

    def _opponent_will_release(self, frame: Frame, plan: Plan) -> bool:
        """Checks whether executable opponent medicines cover every blocked pickup.
        :param frame: Supplies the opponent route, storage, expertise, and samples.
        :param plan: Supplies pickups that exceed current distributor inventory.
        :return: Indicates whether visible production can replenish every shortage."""
        if self.turn < self.release_abandoned_until or frame.opponent.target != "LABORATORY" or frame.opponent.eta > 2:
            return False
        required = tuple(max(plan.pickups[index] - frame.available[index], 0) if plan.pickups[index] > 0 else 0 for index in range(5))
        samples = [sample for sample in frame.samples if sample.carried_by == 1 and sample.health >= 0]
        for size in range(1, len(samples) + 1):
            for order in permutations(samples, size):
                storage = list(frame.opponent.storage)
                expertise = list(frame.opponent.expertise)
                released = [0, 0, 0, 0, 0]
                for sample in order:
                    need = tuple(max(sample.cost[index] - expertise[index], 0) for index in range(5))
                    if any(need[index] > storage[index] for index in range(5)):
                        break
                    for index in range(5):
                        storage[index] -= need[index]
                        released[index] += need[index]
                    expertise[sample.gain] += 1
                    if all(released[index] >= required[index] for index in range(5)):
                        return True
        return False

    def _opponent_project_eta(self, frame: Frame, project: Vector) -> int:
        """Estimates the earliest visible turn on which the opponent can claim a project.
        :param frame: Supplies opponent samples, route, storage, and shared molecule supply.
        :param project: Supplies the science-project expertise requirements.
        :return: Provides turns until completion, or a value beyond the game horizon when not visibly reachable."""
        if project in self.project_eta_cache:
            return self.project_eta_cache[project]
        if self._dominates(frame.opponent.expertise, project):
            self.project_eta_cache[project] = 0
            return 0
        samples = [sample for sample in frame.samples if sample.carried_by == 1 and sample.health >= 0]
        if not samples:
            self.project_eta_cache[project] = GAME_TURNS + 1
            return GAME_TURNS + 1
        laboratory_distance = {"START_POS": 2, "SAMPLES": 3, "DIAGNOSIS": 4, "MOLECULES": 3, "LABORATORY": 0}
        molecule_distance = {"START_POS": 2, "SAMPLES": 3, "DIAGNOSIS": 3, "MOLECULES": 0, "LABORATORY": 3}
        best = GAME_TURNS + 1
        for size in range(1, len(samples) + 1):
            for order in permutations(samples, size):
                storage = list(frame.opponent.storage)
                expertise = list(frame.opponent.expertise)
                for step, sample in enumerate(order, 1):
                    need = tuple(max(sample.cost[index] - expertise[index], 0) for index in range(5))
                    if any(need[index] > storage[index] for index in range(5)):
                        break
                    for index in range(5):
                        storage[index] -= need[index]
                    expertise[sample.gain] += 1
                    if self._dominates(tuple(expertise), project):
                        best = min(best, frame.opponent.eta + laboratory_distance[frame.opponent.target] + step)
                        break
                expertise = list(frame.opponent.expertise)
                required = [0, 0, 0, 0, 0]
                for step, sample in enumerate(order, 1):
                    for index in range(5):
                        required[index] += max(sample.cost[index] - expertise[index], 0)
                    expertise[sample.gain] += 1
                    if not self._dominates(tuple(expertise), project):
                        continue
                    pickups = tuple(max(required[index] - frame.opponent.storage[index], 0) for index in range(5))
                    pre_lab = tuple(max(required[index], frame.opponent.storage[index]) for index in range(5))
                    if sum(pre_lab) <= 10 and all(pickups[index] <= max(frame.available[index], 0) for index in range(5)):
                        eta = frame.opponent.eta + molecule_distance[frame.opponent.target] + sum(pickups) + 3 + step
                        best = min(best, eta)
                    break
        self.project_eta_cache[project] = best
        return best

    def _dominates(self, expertise: Vector, project: Vector) -> bool:
        """Checks whether an expertise vector completes a science project.
        :param expertise: Supplies current expertise by molecule type.
        :param project: Supplies required expertise by molecule type.
        :return: Indicates whether every project requirement is met."""
        return all(expertise[index] >= project[index] for index in range(5))

    def _project_deficit(self, expertise: Vector, project: Vector) -> int:
        """Counts expertise gains still required for a science project.
        :param expertise: Supplies current expertise by molecule type.
        :param project: Supplies required expertise by molecule type.
        :return: Provides the summed positive expertise deficit."""
        return sum(max(project[index] - expertise[index], 0) for index in range(5))


def main():
    """Reads arena frames, delegates decisions, and prints one flushed command per turn."""
    projects = tuple(tuple(map(int, stdin.readline().split())) for _ in range(int(stdin.readline())))
    bot = Bot(projects)
    while (frame := read_frame(stdin)) is not None:
        action = bot.decide(frame)
        print(action, flush=True)
        print(f"t={bot.turn} module={frame.me.target} action={action}", file=stderr)
        bot.turn += 1


def read_frame(stream: TextIO) -> Frame | None:
    """Parses one complete turn from the arena input stream.
    :param stream: Supplies the arena text protocol.
    :return: Provides a parsed frame, or None after clean end-of-file."""
    line = stream.readline()
    if not line:
        return None
    me = parse_robot(line)
    opponent = parse_robot(stream.readline())
    available = tuple(map(int, stream.readline().split()))
    samples = []
    for _ in range(int(stream.readline())):
        parts = stream.readline().split()
        sample_id, carried_by, rank = map(int, parts[:3])
        samples.append(Sample(sample_id, carried_by, rank, MOLECULE_INDEX[parts[3]], int(parts[4]), tuple(map(int, parts[5:]))))
    return Frame(me, opponent, available, tuple(samples))


def parse_robot(line: str) -> Robot:
    """Parses one robot-state protocol line.
    :param line: Supplies a target name followed by twelve integers.
    :return: Provides the parsed robot state."""
    parts = line.split()
    values = tuple(map(int, parts[1:]))
    return Robot(parts[0], values[0], values[1], values[2:7], values[7:12])


if __name__ == "__main__":
    main()
