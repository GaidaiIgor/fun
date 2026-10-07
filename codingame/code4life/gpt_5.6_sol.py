"""Plays the Roche Code4Life arena game with deadline-aware batch planning."""

from dataclasses import dataclass
from itertools import permutations, product
from sys import stderr, stdin
from typing import TextIO, TypeAlias


Vector: TypeAlias = tuple[int, int, int, int, int]

MOLECULE_NAMES = ("A", "B", "C", "D", "E")
MOLECULE_INDEX = {name: index for index, name in enumerate(MOLECULE_NAMES)} | {"0": -1}
GAME_TURNS = 200
MOLECULE_WAIT_LIMIT = 4
CLOUD_TARGET_TURNS = 12


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
    :var planned_samples: Preserves the production order of a committed owned batch.
    :var cloud_target: Preserves a valuable cloud sample and the turn through which diagnosis should wait for it.
    :var project_eta_cache: Caches opponent science-project timing within one frame."""

    projects: tuple[Vector, ...]
    turn: int
    diagnosed_by_me: set[int]
    rejected_until: dict[int, int]
    molecule_waits: int
    release_abandoned_until: int
    planned_samples: tuple[int, ...]
    cloud_target: tuple[int, int] | None
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
        self.planned_samples = ()
        self.cloud_target = None
        self.project_eta_cache = {}

    def decide(self, frame: Frame) -> str:
        """Chooses the command for the current authoritative frame.
        :param frame: Supplies both robots, molecule availability, and samples.
        :return: Provides one legal game command."""
        self.project_eta_cache.clear()
        owned = [sample for sample in frame.samples if sample.carried_by == 0]
        self.planned_samples = tuple(sample_id for sample_id in self.planned_samples if any(sample.sample_id == sample_id for sample in owned))
        if self.cloud_target is not None and not any(sample.sample_id == self.cloud_target[0] and sample.carried_by == -1 for sample in frame.samples):
            self.cloud_target = None
        if frame.me.eta > 0:
            self.molecule_waits = 0
            return "WAIT"
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
        self.planned_samples = ()
        if len(owned) == 3:
            return "GOTO DIAGNOSIS"
        remaining = GAME_TURNS - self.turn
        missing_ranks = list(self._sample_ranks(frame))
        for sample in owned:
            if sample.rank in missing_ranks:
                missing_ranks.remove(sample.rank)
        rank = missing_ranks[0]
        carried_ranks = tuple(sample.rank for sample in owned if sample.health < 0)
        for candidate in range(rank, 0, -1):
            unknown_ranks = carried_ranks + (candidate,)
            estimated_pickups = self._estimated_unknown_pickups(unknown_ranks, frame.me)
            estimated_finish = 8 + len(unknown_ranks) * 2 if estimated_pickups == 0 \
                else 10 + len(unknown_ranks) * 2 + estimated_pickups + 6 * ((estimated_pickups - 1) // 10)
            if remaining >= estimated_finish + 2 * bool(carried_ranks):
                rank = candidate
                break
        else:
            rank = 0
        cloud_plan = None
        if not owned:
            cloud = [sample for sample in frame.samples
                     if sample.carried_by == -1 and sample.health >= 0 and self.rejected_until.get(sample.sample_id, 0) <= self.turn]
            cloud_plan = self._best_plan(cloud, frame, "SAMPLES_CLOUD", True)
            if cloud_plan is None:
                cloud_plan = self._best_plan(cloud, frame, "SAMPLES_CLOUD", False)
        if cloud_plan is not None and (rank == 0 or cloud_plan.reward >= (10, 20, 30)[rank - 1] or cloud_plan.value >= 35):
            self.cloud_target = (cloud_plan.samples[0].sample_id, self.turn + CLOUD_TARGET_TURNS)
            return "GOTO DIAGNOSIS"
        if rank == 0:
            self.cloud_target = None
            return "GOTO DIAGNOSIS" if owned else "WAIT"
        self.cloud_target = None
        return f"CONNECT {rank}"

    def _at_diagnosis(self, frame: Frame, owned: list[Sample]) -> str:
        """Diagnoses unknowns and exchanges samples for the best visible portfolio.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a DIAGNOSIS-module command."""
        self.molecule_waits = 0
        self.planned_samples = ()
        ready = self._ready_samples(owned, frame.me)
        if ready and GAME_TURNS - self.turn <= 4 + len(ready):
            return "GOTO LABORATORY"
        unknown = [sample for sample in owned if sample.health < 0]
        if unknown:
            self.cloud_target = None
            sample = min(unknown, key=lambda item: item.sample_id)
            self.diagnosed_by_me.add(sample.sample_id)
            return f"CONNECT {sample.sample_id}"
        cloud = [sample for sample in frame.samples
                 if sample.carried_by == -1 and sample.health >= 0 and self.rejected_until.get(sample.sample_id, 0) <= self.turn]
        candidates = owned + cloud
        plan = self._best_routable_plan(candidates, frame, "DIAGNOSIS")
        if plan is not None:
            target_ids = {sample.sample_id for sample in plan.samples}
            desired_cloud = [sample for sample in plan.samples if sample.carried_by == -1]
            if desired_cloud:
                if len(owned) < 3:
                    self.cloud_target = None
                    return f"CONNECT {desired_cloud[0].sample_id}"
                rejected = [sample for sample in owned if sample.sample_id not in target_ids]
                sample = min(rejected, key=lambda item: (self._evaluate_order((item,), frame, "DIAGNOSIS", False, len(owned)) is not None,
                                                          self._candidate_score(item, frame)))
                target = desired_cloud[0]
                if frame.opponent.target != "DIAGNOSIS" or frame.opponent.eta > 1 or frame.opponent.eta == 1 and target.sample_id in self.diagnosed_by_me:
                    self.rejected_until[sample.sample_id] = self.turn + 8
                    return f"CONNECT {sample.sample_id}"
        if plan is None and self.cloud_target is not None and not owned and frame.opponent.target == "MOLECULES":
            retry = self._best_plan(cloud, frame, "SAMPLES_CLOUD", True)
            if retry is None:
                retry = self._best_plan(cloud, frame, "SAMPLES_CLOUD", False)
            rank = self._finishable_fresh_rank(frame)
            preferred = retry is not None and retry.samples[0].sample_id == self.cloud_target[0] \
                and (rank == 0 or retry.reward >= (10, 20, 30)[rank - 1] or retry.value >= 35)
            if preferred and self.turn < self.cloud_target[1]:
                return "WAIT"
            if preferred:
                self.rejected_until[self.cloud_target[0]] = self.turn + 8
        self.cloud_target = None
        owned_plan = self._best_routable_plan(owned, frame, "DIAGNOSIS")
        if owned_plan is not None:
            owned_plan = self._best_split_plan(owned, frame, owned_plan)
            if self._is_lemon(owned_plan):
                sample = owned_plan.samples[0]
                self.rejected_until[sample.sample_id] = self.turn + 20
                return f"CONNECT {sample.sample_id}"
            if any(owned_plan.pickups) and GAME_TURNS - self.turn <= 12 and ready:
                ready_plan = self._best_plan(ready, frame, "DIAGNOSIS", True)
                shortage = any(owned_plan.pickups[index] > max(frame.available[index], 0) for index in range(5))
                opponent_ready = self._ready_samples([sample for sample in frame.samples if sample.carried_by == 1], frame.opponent)
                released = shortage and len(opponent_ready) == 1 and frame.opponent.target == "LABORATORY" and frame.opponent.eta == 0 \
                    and owned_plan.reward > ready_plan.reward and self._release_delay(frame, owned_plan, "DIAGNOSIS") == 0
                secure = frame.opponent.target != "MOLECULES" and owned_plan.reward > ready_plan.reward \
                    and all(amount * 2 <= max(frame.available[index], 0) for index, amount in enumerate(owned_plan.pickups))
                if not released and not secure:
                    return "GOTO LABORATORY"
            expertise = list(frame.me.expertise)
            for sample in owned_plan.samples:
                expertise[sample.gain] += 1
            target_ids = {sample.sample_id for sample in owned_plan.samples}
            impossible = [sample for sample in owned if sample.sample_id not in target_ids and not self._physically_possible(sample, tuple(expertise))]
            if impossible and owned_plan.turns + self._plan_delay(frame, owned_plan, "DIAGNOSIS") < GAME_TURNS - self.turn:
                sample = min(impossible, key=lambda item: self._candidate_score(item, frame))
                self.rejected_until[sample.sample_id] = self.turn + 20
                return f"CONNECT {sample.sample_id}"
            if not any(owned_plan.pickups):
                self.planned_samples = tuple(sample.sample_id for sample in owned_plan.samples)
                return "GOTO LABORATORY"
            self.planned_samples = tuple(sample.sample_id for sample in owned_plan.samples)
            return "GOTO MOLECULES"
        if GAME_TURNS - self.turn <= 8:
            return "GOTO LABORATORY" if self._ready_samples(owned, frame.me) else "WAIT"
        if owned:
            if GAME_TURNS - self.turn <= 12:
                return "WAIT"
            impossible = [sample for sample in owned if not self._physically_possible(sample, frame.me.expertise)]
            if not impossible:
                if not self._finishable_fresh_rank(frame):
                    return "WAIT"
                if len(owned) < 3:
                    return "GOTO SAMPLES"
            if impossible:
                sample = min(impossible, key=lambda item: self._candidate_score(item, frame))
            else:
                scores = {item: self._candidate_score(item, frame) for item in owned}
                gaps = {item: 1 + sum(max(item.cost[index] - frame.me.expertise[index] - frame.me.storage[index] - max(frame.available[index], 0), 0)
                                           for index in range(5)) for item in owned}
                sample = min(owned, key=lambda item: (scores[item] / gaps[item], scores[item]))
            self.rejected_until[sample.sample_id] = self.turn + (20 if impossible else 8)
            return f"CONNECT {sample.sample_id}"
        return "GOTO SAMPLES" if self._finishable_fresh_rank(frame) else "WAIT"

    def _at_molecules(self, frame: Frame, owned: list[Sample]) -> str:
        """Collects the scarcest required molecule for the best current batch.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a MOLECULES-module command."""
        plan = self._committed_plan(owned, frame, "MOLECULES")
        if plan is None:
            self.molecule_waits = 0
            plan = self._best_routable_plan(owned, frame, "MOLECULES")
        if plan is not None:
            if self._is_lemon(plan):
                self.molecule_waits = 0
                self.planned_samples = ()
                return "GOTO DIAGNOSIS"
            choices = [index for index, amount in enumerate(plan.pickups) if amount > 0 and frame.available[index] > 0]
            if not choices and any(plan.pickups) and not self._ready_samples(owned, frame.me):
                plan_delay = self._plan_delay(frame, plan, "MOLECULES")
                project_reward = plan.reward - sum(sample.health for sample in plan.samples)
                alternatives = []
                for sample in owned:
                    alternative = self._best_routable_plan(owned, frame, "MOLECULES", (sample.sample_id,))
                    if alternative is None or self._is_lemon(alternative):
                        continue
                    alternative_delay = self._plan_delay(frame, alternative, "MOLECULES")
                    displaced = [item for item in plan.samples if item not in alternative.samples]
                    tail_turns = 0
                    if displaced:
                        expertise = list(frame.me.expertise)
                        for item in alternative.samples:
                            expertise[item.gain] += 1
                        storage = tuple(max(frame.me.storage[index] - alternative.required[index], 0) for index in range(5))
                        required = [0, 0, 0, 0, 0]
                        for item in displaced:
                            for index in range(5):
                                required[index] += max(item.cost[index] - expertise[index], 0)
                            expertise[item.gain] += 1
                        tail_pickups = sum(max(required[index] - storage[index], 0) for index in range(5))
                        tail_turns = plan_delay + len(displaced) + (6 + tail_pickups if tail_pickups else 0)
                    first_missing = tuple(max(alternative.samples[0].cost[index] - frame.me.expertise[index] - frame.me.storage[index], 0)
                                          for index in range(5))
                    if alternative_delay >= plan_delay or alternative.turns + alternative_delay > plan.turns + plan_delay + 1:
                        continue
                    if alternative.turns + alternative_delay + tail_turns > GAME_TURNS - self.turn \
                            or GAME_TURNS - self.turn <= 30 and alternative.reward < plan.reward:
                        continue
                    if alternative.reward - sum(item.health for item in alternative.samples) < project_reward \
                            or not any(first_missing[index] > 0 and frame.available[index] > 0 for index in range(5)):
                        continue
                    alternatives.append(alternative)
                if alternatives:
                    alternative = max(alternatives, key=lambda item: (-self._plan_delay(frame, item, "MOLECULES"),
                                                                       -item.turns, self._plan_key(item)))
                    plan = alternative
                    choices = [index for index, amount in enumerate(plan.pickups) if amount > 0 and frame.available[index] > 0]
            self.planned_samples = tuple(sample.sample_id for sample in plan.samples)
            denial = self._terminal_denial_choice(plan, frame)
            if denial >= 0:
                self.molecule_waits = 0
                return f"CONNECT {MOLECULE_NAMES[denial]}"
            if choices:
                self.molecule_waits = 0
                return f"CONNECT {MOLECULE_NAMES[self._molecule_choice(plan, frame, choices)]}"
            if not any(plan.pickups):
                self.molecule_waits = 0
                return "GOTO LABORATORY"
        ready = self._ready_samples(owned, frame.me)
        if ready:
            self.molecule_waits = 0
            if self.planned_samples and all(sample.sample_id != self.planned_samples[0] for sample in ready):
                self.planned_samples = ()
            return "GOTO LABORATORY"
        if frame.opponent.target == "LABORATORY" and frame.opponent.eta > 0:
            self.molecule_waits = 0
        release_delay = self._release_delay(frame, plan, "MOLECULES") if plan is not None else GAME_TURNS + 1
        if plan is not None and self.molecule_waits < MOLECULE_WAIT_LIMIT \
                and plan.turns + release_delay <= GAME_TURNS - self.turn and release_delay <= GAME_TURNS:
            self.molecule_waits += 1
            return "WAIT"
        if plan is not None and release_delay <= GAME_TURNS:
            self.release_abandoned_until = self.turn + 8
        self.molecule_waits = 0
        self.planned_samples = ()
        if GAME_TURNS - self.turn <= 5:
            return "WAIT"
        return "GOTO DIAGNOSIS" if owned else "GOTO SAMPLES"

    def _molecule_choice(self, plan: Plan, frame: Frame, choices: list[int]) -> int:
        """Chooses the scarcest planned molecule available for immediate collection.
        :param plan: Supplies the current batch and outstanding pickups.
        :param frame: Supplies shared inventory and opponent demand.
        :param choices: Lists available required molecule indices.
        :return: Provides the selected molecule index."""
        opponent_need = self._opponent_need(frame)
        first_missing = tuple(max(plan.samples[0].cost[index] - frame.me.expertise[index] - frame.me.storage[index], 0) for index in range(5))
        return min(choices, key=lambda item: (max(frame.available[item], 0) - plan.pickups[item] - opponent_need[item],
                                             0 if first_missing[item] else 1, frame.available[item], -plan.pickups[item], item))

    def _terminal_denial_choice(self, plan: Plan, frame: Frame) -> int:
        """Selects a spare molecule that permanently blocks the opponent without sacrificing our final batch.
        :param plan: Supplies the finishable final owned batch.
        :param frame: Supplies shared inventory and opponent samples.
        :return: Provides the molecule index to hoard, or -1 when denial is not guaranteed."""
        remaining = GAME_TURNS - self.turn
        owned_count = sum(sample.carried_by == 0 for sample in frame.samples)
        if remaining > 12 or frame.opponent.target != "MOLECULES" or len(plan.samples) != owned_count \
                or plan.reward != sum(sample.health for sample in plan.samples) \
                or any(plan.pickups[index] > max(frame.available[index], 0) for index in range(5)):
            return -1
        budget = min(remaining - plan.turns,
                     10 - sum(max(plan.required[index], frame.me.storage[index]) for index in range(5)))
        finishable = []
        for sample in frame.samples:
            if sample.carried_by != 1 or sample.health < 0:
                continue
            need = tuple(max(sample.cost[index] - frame.opponent.expertise[index] - frame.opponent.storage[index], 0)
                         for index in range(5))
            if sum(need) <= 10 - sum(frame.opponent.storage) and frame.opponent.eta + sum(need) + 4 <= remaining \
                    and all(need[index] <= max(frame.available[index], 0) for index in range(5)):
                finishable.append(need)
        if len(finishable) != 1 or not any(finishable[0]):
            return -1
        choices = []
        capacity = 10 - sum(frame.opponent.storage)
        for index, need in enumerate(finishable[0]):
            supply = max(frame.available[index], 0)
            denial_turns = supply - need + 1
            if plan.required[index] or not supply >= need > (supply + 1) // 2 or denial_turns > budget:
                continue
            for target, amount in enumerate(plan.pickups):
                if not amount:
                    continue
                pool = max(frame.available[target], 0)
                stolen = min(denial_turns, capacity, pool)
                if pool - stolen < amount + min(capacity - stolen, amount - 1):
                    break
            else:
                choices.append(index)
        return min(choices, key=lambda index: (frame.available[index] - finishable[0][index], index), default=-1)

    def _at_laboratory(self, frame: Frame, owned: list[Sample]) -> str:
        """Produces the best ready medicine or starts the next profitable route.
        :param frame: Supplies the current game state.
        :param owned: Supplies samples carried by IGPro.
        :return: Provides a LABORATORY-module command."""
        self.molecule_waits = 0
        if owned:
            self.cloud_target = None
        ready = self._ready_samples(owned, frame.me)
        if ready:
            if self.planned_samples:
                by_id = {sample.sample_id: sample for sample in owned}
                sample = by_id[self.planned_samples[0]]
                if sample in ready:
                    return f"CONNECT {sample.sample_id}"
                self.planned_samples = ()
            return f"CONNECT {self._best_laboratory_order(owned, frame)[0].sample_id}"
        plan = self._committed_plan(owned, frame, "LABORATORY")
        if plan is None:
            plan = self._best_routable_plan(owned, frame, "LABORATORY")
        if plan is not None:
            if self._is_lemon(plan):
                self.planned_samples = ()
                return "GOTO DIAGNOSIS"
            self.planned_samples = tuple(sample.sample_id for sample in plan.samples)
            return "GOTO MOLECULES"
        self.planned_samples = ()
        if owned:
            if GAME_TURNS - self.turn <= 12:
                return "WAIT"
            possible = all(self._physically_possible(sample, frame.me.expertise) for sample in owned)
            if possible and not self._finishable_fresh_rank(frame):
                return "WAIT"
            return "GOTO SAMPLES" if len(owned) < 3 and possible else "GOTO DIAGNOSIS"
        cloud = [sample for sample in frame.samples
                 if sample.carried_by == -1 and sample.health >= 0 and self.rejected_until.get(sample.sample_id, 0) <= self.turn]
        plan = self._best_plan(cloud, frame, "CLOUD", True)
        if plan is None:
            plan = self._best_plan(cloud, frame, "CLOUD", False)
        rank = self._finishable_fresh_rank(frame)
        if plan is not None and (plan.reward >= 20 or plan.value >= 25 or rank == 0):
            self.cloud_target = (plan.samples[0].sample_id, self.turn + CLOUD_TARGET_TURNS)
            return "GOTO DIAGNOSIS"
        self.cloud_target = None
        return "GOTO SAMPLES" if rank else "WAIT"

    def _best_split_plan(self, owned: list[Sample], frame: Frame, current: Plan) -> Plan:
        """Chooses a robust first load when three late medicines require two laboratory trips.
        :param owned: Supplies the three diagnosed carried samples.
        :param frame: Supplies expertise, storage, molecule inventory, and the deadline.
        :param current: Supplies the ordinary best routable first load.
        :return: Provides the first load maximizing the finishable two-trip portfolio."""
        if GAME_TURNS - self.turn > 50 or len(owned) != 3 or len(current.samples) != 2 or any(sample.health < 0 for sample in owned):
            return current
        best = current
        best_key = None
        current_total = current.reward + sum(sample.health for sample in owned if sample not in current.samples)
        for order in permutations(owned, 2):
            first = self._best_routable_plan(owned, frame, "DIAGNOSIS", tuple(sample.sample_id for sample in order))
            if first is None or first.samples != order:
                continue
            expertise = list(frame.me.expertise)
            for sample in first.samples:
                expertise[sample.gain] += 1
            leftover = next(sample for sample in owned if sample not in first.samples)
            need = tuple(max(leftover.cost[index] - expertise[index], 0) for index in range(5))
            storage = tuple(max(frame.me.storage[index] - first.required[index], 0) for index in range(5))
            pickups = tuple(max(need[index] - storage[index], 0) for index in range(5))
            available = tuple(max(frame.available[index] + min(frame.me.storage[index], first.required[index]), 0) for index in range(5))
            if sum(max(need[index], storage[index]) for index in range(5)) > 10 \
                    or any(pickups[index] > available[index] for index in range(5)):
                continue
            tail_turns = 1 if not any(pickups) else 7 + sum(pickups)
            turns = first.turns + self._plan_delay(frame, first, "DIAGNOSIS") + tail_turns
            if turns > GAME_TURNS - self.turn or first.reward + leftover.health < current_total:
                continue
            key = (first.reward + leftover.health, -turns, -sum(pickups), first.reward, first.value,
                   tuple(-sample.sample_id for sample in first.samples))
            if best_key is None or key > best_key:
                best, best_key = first, key
        return best

    def _committed_plan(self, owned: list[Sample], frame: Frame, origin: str) -> Plan | None:
        """Revalidates the exact owned batch previously selected for production.
        :param owned: Supplies samples still carried by IGPro.
        :param frame: Supplies the current game state.
        :param origin: Identifies the current route origin.
        :return: Provides the committed plan while feasible, or None after clearing it."""
        if not self.planned_samples:
            return None
        plan = self._best_routable_plan(owned, frame, origin, self.planned_samples)
        if plan is None:
            self.planned_samples = ()
        return plan

    def _sample_ranks(self, frame: Frame) -> tuple[int, int, int]:
        """Chooses an ordered three-sample portfolio from typed expertise and remaining time.
        :param frame: Supplies expertise, storage, and the current turn horizon.
        :return: Provides desired sample ranks in draw order."""
        expertise = sum(frame.me.expertise)
        if expertise < 6:
            return (1, 1, 1)
        if expertise < 10 or min(frame.me.expertise) == 0:
            return (2, 2, 2)
        if GAME_TURNS - self.turn < 45:
            return (2, 2, 3)
        return (3, 3, 3) if min(frame.me.expertise) >= 2 and self._estimated_unknown_pickups((3, 3, 3), frame.me) <= 10 else (3, 2, 2)

    def _finishable_fresh_rank(self, frame: Frame) -> int:
        """Finds the strongest fresh single-sample route that fits the remaining game.
        :param frame: Supplies expertise, storage, and the current turn horizon.
        :return: Provides the selected rank, or zero when none fits."""
        for rank in range(self._sample_ranks(frame)[0], 0, -1):
            pickups = self._estimated_unknown_pickups((rank,), frame.me)
            if pickups > 10:
                continue
            finish = 13 if pickups == 0 else 15 + pickups
            if GAME_TURNS - self.turn >= finish:
                return rank
        return 0

    def _physically_possible(self, sample: Sample, expertise: Vector) -> bool:
        """Checks whether one medicine can ever fit the molecule tray after expertise discounts.
        :param sample: Supplies the diagnosed medicine costs.
        :param expertise: Supplies permanent molecule discounts.
        :return: Indicates whether effective costs respect tray and distributor limits."""
        need = tuple(max(sample.cost[index] - expertise[index], 0) for index in range(5))
        return sum(need) <= 10 and max(need) <= 5

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
                claimed = [self._dominates(frame.me.expertise, project) or self._dominates(frame.opponent.expertise, project) for project in self.projects]
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
                        progress += (before - after) * (10 if before <= opponent + 1 else 5)
                    key = (reward + expertise_value + progress, reward, size, sum(storage), tuple(-sample.sample_id for sample in order))
                    if best_key is None or key > best_key:
                        best, best_key = order, key
        return best

    def _best_routable_plan(self, candidates: list[Sample], frame: Frame, origin: str, prefix: tuple[int, ...] = ()) -> Plan | None:
        """Finds the best currently supplied or safely release-backed batch.
        :param candidates: Supplies samples eligible for the batch.
        :param frame: Supplies robots and current molecule availability.
        :param origin: Identifies the route whose turn cost should be estimated.
        :param prefix: Requires these committed sample IDs at the start of the batch.
        :return: Provides the best safely routable batch, or None when no batch fits."""
        available = self._best_plan(candidates, frame, origin, True, prefix)
        if available is not None and prefix and origin == "MOLECULES":
            by_id = {sample.sample_id: sample for sample in candidates}
            available = self._evaluate_order(tuple(by_id[sample_id] for sample_id in prefix), frame, origin, True, len(candidates))
            while available is not None and len(available.samples) < 3:
                available_ids = tuple(sample.sample_id for sample in available.samples)
                extension = self._best_plan(candidates, frame, origin, True, available_ids, False, len(available.samples) + 1)
                added = extension.samples[len(available.samples):] if extension is not None else ()
                if extension is None or extension.turns - available.turns >= 6 + len(added) \
                        or extension.reward - sum(sample.health for sample in added) < available.reward:
                    break
                available = extension
        if available is None:
            return self._best_plan(candidates, frame, origin, False, prefix)
        if origin in {"DIAGNOSIS", "LABORATORY"} and any(available.pickups) and candidates \
                and all(sample.carried_by == 0 for sample in candidates):
            forecast = self._best_plan(candidates, frame, origin, True, prefix, True)
            if forecast is not None and self._plan_key(forecast) > self._plan_key(available):
                available = forecast
        cloud_fallback = not any(available.pickups) and any(sample.carried_by == -1 for sample in available.samples) \
            and any(sample.carried_by == 0 for sample in candidates)
        if origin != "MOLECULES" and not prefix and cloud_fallback:
            released = self._best_plan([sample for sample in candidates if sample.carried_by == 0], frame, origin, False, wait_limit=0)
            if released is not None and self._plan_key(released) > self._plan_key(available):
                available = released
        late_release = GAME_TURNS - self.turn <= 30 and frame.opponent.target == "LABORATORY" and frame.opponent.eta == 0 \
            and len(self._ready_samples([sample for sample in frame.samples if sample.carried_by == 1], frame.opponent)) == 1
        if origin != "MOLECULES" and not prefix and candidates and all(sample.carried_by == 0 for sample in candidates) \
                and (any(available.pickups) or late_release):
            released = self._best_plan(candidates, frame, origin, False, wait_limit=0)
            if released is not None and self._plan_key(released) > self._plan_key(available):
                available = released
        if len(available.samples) == 3 or not any(available.pickups) and origin != "MOLECULES":
            return available
        available_ids = tuple(sample.sample_id for sample in available.samples)
        released = self._best_plan(candidates, frame, origin, False, available_ids, wait_limit=0)
        if released is None or len(released.samples) == len(available.samples):
            return available
        return released if self._plan_key(released) > self._plan_key(available) else available

    def _plan_delay(self, frame: Frame, plan: Plan, origin: str) -> int:
        """Finds the extra wait before a plan can collect every missing molecule.
        :param frame: Supplies current and visibly held molecule inventory.
        :param plan: Supplies the proposed pickups.
        :param origin: Identifies the current route origin.
        :return: Provides zero for current supply or the visible release delay."""
        if all(plan.pickups[index] <= max(frame.available[index], 0) for index in range(5)):
            return 0
        return self._release_delay(frame, plan, origin)

    def _best_plan(self, candidates: list[Sample], frame: Frame, origin: str, require_available: bool,
                   prefix: tuple[int, ...] = (), forecast_contention: bool = False, minimum_size: int = 1,
                   wait_limit: int = MOLECULE_WAIT_LIMIT) -> Plan | None:
        """Finds the strongest ordered batch among visible diagnosed samples.
        :param candidates: Supplies samples eligible for the batch.
        :param frame: Supplies robots and current molecule availability.
        :param origin: Identifies the route whose turn cost should be estimated.
        :param require_available: Requires current supply when true, or visible timely opponent release when false.
        :param prefix: Requires these committed sample IDs at the start of the batch.
        :param forecast_contention: Allocates the opponent pickup window across types when true.
        :param minimum_size: Requires at least this many samples in the returned batch.
        :param wait_limit: Gives the maximum acceptable visible-release delay.
        :return: Provides the best feasible ordered batch, or None when no batch fits."""
        diagnosed = [sample for sample in candidates if sample.health >= 0]
        owned = sorted((sample for sample in diagnosed if sample.carried_by == 0), key=lambda item: item.sample_id)
        cloud = sorted((sample for sample in diagnosed if sample.carried_by == -1), key=lambda item: (-self._candidate_score(item, frame), item.sample_id))
        eligible = owned + cloud[:max(0, 12 - len(owned))]
        best = None
        best_key = None
        for size in range(minimum_size, min(3, len(eligible)) + 1):
            for order in permutations(eligible, size):
                if tuple(sample.sample_id for sample in order[:len(prefix)]) != prefix:
                    continue
                plan = self._evaluate_order(order, frame, origin, require_available, len(owned), forecast_contention)
                delay = self._release_delay(frame, plan, origin) if plan is not None and not require_available else 0
                if plan is None or delay > wait_limit or plan.turns + delay > GAME_TURNS - self.turn:
                    continue
                if delay:
                    plan = self._evaluate_order(order, frame, origin, require_available, len(owned), forecast_contention, delay)
                stable = (sum(sample.carried_by == 0 for sample in order), -sum(plan.pickups), len(order), tuple(-sample.sample_id for sample in order))
                key = (*self._plan_key(plan, delay), *stable)
                if best_key is None or key > best_key:
                    best, best_key = plan, key
        return best

    def _plan_key(self, plan: Plan, delay: int = 0) -> tuple[int, int, int]:
        """Ranks a plan by deadline-sensitive reward and adjusted production rate.
        :param plan: Supplies the batch value and ordinary route duration.
        :param delay: Supplies extra turns spent waiting for released molecules.
        :return: Provides the primary larger-is-better plan comparison key."""
        rate = plan.value * 1000 // (plan.turns + delay)
        return (plan.reward, rate, plan.value) if GAME_TURNS - self.turn <= 30 else (rate, plan.reward, plan.value)

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

    def _evaluate_order(self, order: tuple[Sample, ...], frame: Frame, origin: str, require_available: bool, owned_count: int,
                        forecast_contention: bool = False, preparation_delay: int = 0) -> Plan | None:
        """Simulates expertise, molecule use, projects, capacity, and route time for one batch order.
        Treats one payoff-qualified opponent production as an expected, not guaranteed, routing delay.
        :param order: Supplies the proposed laboratory production order.
        :param frame: Supplies current robot and molecule state.
        :param origin: Identifies the route whose turn cost should be estimated.
        :param require_available: Requires all planned pickups to exist now when true.
        :param owned_count: Gives the number of samples already carried at diagnosis.
        :param forecast_contention: Allocates the opponent pickup window across types when true.
        :param preparation_delay: Adds release waits before valuing project completion timing.
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
        cloud_count = sum(sample.carried_by == -1 for sample in order)
        switches = cloud_count + max(0, owned_count + cloud_count - 3) if origin in {"DIAGNOSIS", "CLOUD", "SAMPLES_CLOUD"} else 0
        if require_available and any(pickups[index] > max(frame.available[index], 0) for index in range(5)):
            return None
        singleton_race = require_available and len(order) == 1 and order[0].carried_by == 0 \
            and origin in {"DIAGNOSIS", "LABORATORY"}
        if singleton_race and frame.opponent.target == "MOLECULES":
            opponent_need = self._opponent_need(frame)
            capacity = 10 - sum(frame.opponent.storage)
            lead = min(max(3 - frame.opponent.eta, 0), capacity)
            for index, amount in enumerate(pickups):
                supply = max(frame.available[index], 0)
                demand = opponent_need[index]
                taken = min(demand, lead, supply)
                competing = min(demand - taken, capacity - taken)
                supply -= taken
                if amount and supply < amount + min(competing, amount - 1):
                    return None
        if singleton_race and frame.opponent.target in {"DIAGNOSIS", "LABORATORY"}:
            opponent_need = self._opponent_need(frame)
            capacity = 10 - sum(frame.opponent.storage)
            opponent_ready = self._ready_samples([sample for sample in frame.samples if sample.carried_by == 1 and sample.health >= 0],
                                                 frame.opponent) if frame.opponent.target == "LABORATORY" else []
            production_delay = not completed_projects and any(sample.health >= order[0].health for sample in opponent_ready)
            for index, amount in enumerate(pickups):
                initial_supply = max(frame.available[index], 0)
                competing = min(opponent_need[index], capacity)
                secured = min(amount, frame.opponent.eta, initial_supply)
                amount, supply = amount - secured, initial_supply - secured
                if amount and supply < amount + min(competing, amount - 1):
                    return None
                if opponent_need[index] or not capacity or pickups[index] != initial_supply:
                    continue
                secured = min(pickups[index], frame.opponent.eta + production_delay, initial_supply)
                amount, supply = pickups[index] - secured, initial_supply - secured
                if amount and supply < amount + min(1, amount - 1):
                    return None
        if origin in {"DIAGNOSIS", "LABORATORY", "MOLECULES", "CLOUD"} and frame.opponent.target == "MOLECULES":
            arrival = {"MOLECULES": 0, "DIAGNOSIS": 3, "LABORATORY": 3, "CLOUD": 7}[origin] + switches
            opponent_window = min(max(0, arrival - frame.opponent.eta), 10 - sum(frame.opponent.storage))
            opponent_need = self._opponent_need(frame)
            forecast = [max(amount, 0) for amount in frame.available]
            if len(order) == 1 and origin != "CLOUD":
                visible_need = sum(opponent_need)
                prior_need = max(visible_need - 1, 0)
                immediate_race = arrival == frame.opponent.eta == 0 and visible_need <= 2
                denial_supply = tuple(max(forecast[index] - min(opponent_need[index], opponent_window), 0) for index in range(5))
                future_need = tuple(max(opponent_need[index] - opponent_window, 0) for index in range(5))
                opponent_storage = sum(frame.opponent.storage)
                deadlines = {}
                for index in range(5):
                    targeted = opponent_need[index] > 0
                    if pickups[index] < 2 or pickups[index] != denial_supply[index]:
                        continue
                    preceding = min(opponent_need[index], opponent_window) if targeted else 0 if immediate_race else prior_need
                    denial_turn = frame.opponent.eta + preceding
                    if opponent_storage + preceding < 10:
                        deadlines[index] = max(denial_turn - arrival, 0) + 1
                schedule = sorted((index for index in range(5) if pickups[index] > 0),
                                  key=lambda index: (denial_supply[index] - pickups[index] - future_need[index], denial_supply[index], -pickups[index], index))
                elapsed = 0
                for index in schedule:
                    elapsed += pickups[index]
                    if index in deadlines and elapsed > deadlines[index]:
                        return None
            if require_available:
                if forecast_contention:
                    opponent_need = list(opponent_need)
                    for _ in range(opponent_window):
                        choices = [index for index in range(5) if opponent_need[index] > 0 and forecast[index] > 0]
                        if not choices:
                            break
                        index = min(choices, key=lambda item: (forecast[item] - opponent_need[item], forecast[item], item))
                        forecast[index] -= 1
                        opponent_need[index] -= 1
                else:
                    forecast = [max(forecast[index] - min(opponent_need[index], opponent_window), 0) for index in range(5)]
                if any(pickups[index] > forecast[index] for index in range(5)):
                    return None
        if not require_available and any(pickups[index] > max(frame.available[index] + frame.opponent.storage[index], 0) for index in range(5)):
            return None
        pickup_count = sum(pickups)
        route = {"DIAGNOSIS": (4, 6), "MOLECULES": (3, 3), "CLOUD": (8, 10), "SAMPLES_CLOUD": (7, 9), "LABORATORY": (6, 6)}[origin][pickup_count > 0]
        turns = route + pickup_count + len(order) + switches
        if turns > GAME_TURNS - self.turn:
            return None
        preparation = route + pickup_count + switches + preparation_delay
        reward += sum(50 for index, step in completed_projects.items() if preparation + step <= self._opponent_project_eta(frame, self.projects[index]))
        expertise_value = len(order) * (3 if GAME_TURNS - self.turn > 50 else 1 if GAME_TURNS - self.turn > 20 else 0)
        progress = 0
        for index, project in enumerate(self.projects):
            if claimed[index] or self._dominates(frame.me.expertise, project) or self._dominates(frame.opponent.expertise, project):
                continue
            before = self._project_deficit(frame.me.expertise, project)
            after = self._project_deficit(tuple(expertise), project)
            opponent = self._project_deficit(frame.opponent.expertise, project)
            progress += (before - after) * (10 if before <= opponent + 1 else 5)
        leftovers = sum(pre_lab[index] - required[index] for index in range(5))
        ownership = sum(sample.carried_by == -1 and sample.sample_id in self.diagnosed_by_me for sample in order)
        batching = max(0, len(order) - 1) * 8
        return Plan(order, tuple(required), pickups, reward, reward + expertise_value + progress + ownership + batching - leftovers * 2, turns)

    def _estimated_unknown_pickups(self, ranks: tuple[int, ...], robot: Robot) -> int:
        """Estimates batch pickups without treating expertise types as interchangeable.
        :param ranks: Supplies ranks of the undiagnosed samples in the prospective batch.
        :param robot: Supplies current expertise and molecule storage.
        :return: Provides a symmetric representative pickup estimate."""
        return sum(max(sum(ranks) - len(ranks) * robot.expertise[index] - robot.storage[index], 0) for index in range(5))

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

    def _release_delay(self, frame: Frame, plan: Plan, origin: str) -> int:
        """Calculates a release delay guaranteed across every maximal valid opponent production branch.
        :param frame: Supplies the opponent module and arrival time.
        :param plan: Supplies planned samples and pickups.
        :param origin: Identifies the current route origin.
        :return: Provides additional turns beyond the ordinary batch route, or an unreachable sentinel."""
        releasing = frame.opponent.target == "LABORATORY" or frame.opponent.target == "MOLECULES" and frame.opponent.eta == 0
        required = tuple(max(plan.pickups[index] - frame.available[index], 0) if plan.pickups[index] > 0 else 0 for index in range(5))
        if self.turn < self.release_abandoned_until or not releasing or not any(required):
            return GAME_TURNS + 1
        cloud_count = sum(sample.carried_by == -1 for sample in plan.samples)
        switches = cloud_count + max(0, sum(sample.carried_by == 0 for sample in frame.samples) + cloud_count - 3) \
            if origin in {"DIAGNOSIS", "CLOUD", "SAMPLES_CLOUD"} else 0
        approach = {"MOLECULES": 0, "DIAGNOSIS": 3, "LABORATORY": 3, "CLOUD": 7, "SAMPLES_CLOUD": 6}[origin] + switches
        pickup_count = sum(plan.pickups)
        available = tuple(max(amount, 0) for amount in frame.available)
        theft_delay = 1 if frame.opponent.target == "MOLECULES" and frame.opponent.eta == 0 and origin != "MOLECULES" \
            and sum(frame.opponent.storage) < 10 and any(available) else 0
        first_production = 4 + theft_delay if frame.opponent.target == "MOLECULES" else frame.opponent.eta + 1
        initial = sum(min(plan.pickups[index], available[index]) for index in range(5))
        worst = 0
        release_pools = []
        samples = [sample for sample in frame.samples if sample.carried_by == 1 and sample.health >= 0]
        states = [(tuple(samples), frame.opponent.storage, frame.opponent.expertise, (0, 0, 0, 0, 0), (0,) * initial)]
        while states:
            remaining, storage, expertise, released, ready = states.pop()
            produced = False
            production = first_production + len(samples) - len(remaining)
            for sample_index, sample in enumerate(remaining):
                need = tuple(max(sample.cost[index] - expertise[index], 0) for index in range(5))
                if any(need[index] > storage[index] for index in range(5)):
                    continue
                produced = True
                next_storage = tuple(storage[index] - need[index] for index in range(5))
                next_expertise = tuple(expertise[index] + (index == sample.gain) for index in range(5))
                next_released = tuple(released[index] + need[index] for index in range(5))
                unlocked = sum(min(plan.pickups[index], max(frame.available[index] + next_released[index], 0)) for index in range(5))
                next_ready = ready + (production,) * (unlocked - len(ready))
                if len(next_ready) < pickup_count:
                    states.append((remaining[:sample_index] + remaining[sample_index + 1:], next_storage, next_expertise, next_released, next_ready))
                    continue
                worst = max(worst, max((turn - approach - index for index, turn in enumerate(next_ready)), default=0))
                release_pools.append(tuple(max(frame.available[index] + next_released[index], 0) for index in range(5)))
            if not produced:
                return GAME_TURNS + 1
        if frame.opponent.target == "MOLECULES" and frame.opponent.eta == 0 and sum(frame.opponent.storage) < 10:
            standalone = [tuple(max(sample.cost[index] - frame.me.expertise[index] - frame.me.storage[index], 0) for index in range(5))
                          for sample in plan.samples]
            choices = [index for index, amount in enumerate(plan.pickups) if amount > 0 and frame.available[index] > 0]
            secured = self._molecule_choice(plan, frame, choices) if origin == "MOLECULES" and choices else -1
            opportunities = sum(min(plan.pickups[index], max(frame.available[index], 0)) for index in range(5))
            theft_limit = min(10 - sum(frame.opponent.storage), opportunities if origin == "MOLECULES" else 1)
            for pool in release_pools:
                costs = [tuple(max(pool[index] - need[index] + 1, 0) for index in range(5)) for need in standalone]
                for assignment in product(range(5), repeat=len(plan.samples)):
                    theft = [0, 0, 0, 0, 0]
                    for sample_index, index in enumerate(assignment):
                        cost = costs[sample_index][index]
                        if cost and index == secured and standalone[sample_index][index] <= 1:
                            cost = GAME_TURNS + 1
                        theft[index] = max(theft[index], cost)
                    if sum(theft) <= theft_limit and all(theft[index] <= available[index] for index in range(5)):
                        return GAME_TURNS + 1
        return max(worst, 0)

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
        owned = ";".join(f"{sample.sample_id}:{sample.rank}:{sample.gain}:{sample.health}:{sample.cost}" for sample in frame.samples if sample.carried_by == 0)
        theirs = ";".join(f"{sample.sample_id}:{sample.rank}:{sample.gain}:{sample.health}:{sample.cost}" for sample in frame.samples if sample.carried_by == 1)
        me = f"{frame.me.score}:{frame.me.storage}:{frame.me.expertise}"
        opponent = f"{frame.opponent.target}:{frame.opponent.eta}:{frame.opponent.score}:{frame.opponent.storage}:{frame.opponent.expertise}"
        print(f"t={bot.turn} module={frame.me.target} action={action} me={me} available={frame.available} " \
              f"opponent={opponent} owned={owned} theirs={theirs} planned={bot.planned_samples} projects={bot.projects if bot.turn == 0 else ()}", file=stderr)
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
