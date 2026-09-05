"""Plays Code4Life by planning ordered batches, expertise gains, and molecule collection."""

import sys
from dataclasses import dataclass
from itertools import permutations
from time import perf_counter


TYPES = "ABCDE"
MODULES = ("SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY")
DISTANCES = {
    "START_POS": dict.fromkeys(MODULES, 2),
    "SAMPLES": {"SAMPLES": 0, "DIAGNOSIS": 3, "MOLECULES": 3, "LABORATORY": 3},
    "DIAGNOSIS": {"SAMPLES": 3, "DIAGNOSIS": 0, "MOLECULES": 3, "LABORATORY": 4},
    "MOLECULES": {"SAMPLES": 3, "DIAGNOSIS": 3, "MOLECULES": 0, "LABORATORY": 3},
    "LABORATORY": {"SAMPLES": 3, "DIAGNOSIS": 4, "MOLECULES": 3, "LABORATORY": 0}}


@dataclass(slots=True)
class Robot:
    """Stores a robot's observed module, travel time, score, molecules, and expertise."""
    target: str
    eta: int
    score: int
    storage: tuple[int, ...]
    expertise: tuple[int, ...]


@dataclass(slots=True)
class Sample:
    """Stores sample identity, owner, rank, expertise type, health, and diagnosed costs."""
    id: int
    carried_by: int
    rank: int
    gain: int
    health: int
    cost: tuple[int, ...]


@dataclass(slots=True)
class Plan:
    """Records a production order, next action, first collection, timing, and value."""
    order: tuple[int, ...]
    action: str
    takes: tuple[int, ...]
    duration: int
    points: int
    rating: float
    completions: tuple[int, ...]
    batch: tuple[int, ...] = ()


class Bot:
    """Chooses actions using persistent project, sample, timing, and plan observations."""

    def __init__(self, projects: list[tuple[int, ...]]):
        self.projects = projects
        self.claimed = set()
        self.turn = 0
        self.travel_extra = 1
        self.last_move = None
        self.commitment = ()
        self.collection_order = ()
        self.collection_batch = ()
        self.deferred_diagnosis = ()
        self.catalog = {1: {}, 2: {}, 3: {}}
        self.seen = set()
        self.stalled = 0
        self.last_stock = None
        self.enemy_idle = 0
        self.last_enemy = None
        self.debug = ""

    def decide(self, me: Robot, opponent: Robot, available: tuple[int, ...], samples: list[Sample]) -> str:
        """Returns the next protocol command for the supplied complete turn observation."""
        started = perf_counter()
        self.deadline = started + 0.032
        self.turn += 1
        self.me, self.opponent, self.available = me, opponent, available
        self.remaining = 201 - self.turn
        self.own = [sample for sample in samples if sample.carried_by == 0]
        self.known = [sample for sample in self.own if sample.health >= 0]
        self.cloud = [sample for sample in samples if sample.carried_by == -1 and sample.health >= 0]
        self.enemy = [sample for sample in samples if sample.carried_by == 1 and sample.health >= 0]
        for index, project in enumerate(self.projects):
            if any(all(expertise[i] >= project[i] for i in range(5)) for expertise in (me.expertise, opponent.expertise)):
                self.claimed.add(index)
        revealed = []
        for sample in samples:
            if sample.health >= 0 and sample.id not in self.seen:
                self.seen.add(sample.id)
                self.catalog[sample.rank][(sample.cost, sample.gain, sample.health)] = sample
                revealed.append((sample.id, sample.carried_by, sample.rank, sample.gain, sample.health, sample.cost))
        if self.last_move is not None:
            target, distance = self.last_move
            if me.target == target:
                self.travel_extra = me.eta + 1 - distance
            self.last_move = None
        stock = (me.storage, available, tuple(sample.id for sample in self.own), opponent.storage)
        self.stalled = self.stalled + 1 if self.last_stock == stock and me.eta == 0 else 0
        self.last_stock = stock
        self.enemy_picking = tuple(max(0, opponent.storage[i] - self.last_enemy[2][i]) if self.last_enemy is not None else 0 for i in range(5))
        enemy_state = (opponent.target, opponent.eta, opponent.storage, opponent.expertise, tuple(sample.id for sample in self.enemy))
        self.enemy_idle = self.enemy_idle + 1 if enemy_state == self.last_enemy else 0
        self.last_enemy = enemy_state
        if me.target != "MOLECULES":
            self.collection_order = ()
            self.collection_batch = ()
        self.deferred_diagnosis = tuple(identity for identity in self.deferred_diagnosis if any(sample.id == identity for sample in self.known))
        self.enemy_need = tuple(max((max(0, sample.cost[i] - opponent.expertise[i] - opponent.storage[i])
                                    for sample in self.enemy), default=0) for i in range(5))
        self.project_deadlines = self.opponent_project_times()
        self.reason = ""
        self.selected = None
        action = self.act()
        if action.startswith("GOTO "):
            target = action.split()[1]
            self.last_move = (target, DISTANCES[me.target][target])
        plan = self.selected
        details = ";".join(f"{sample.id}:r{sample.rank}:h{sample.health}:g{sample.gain}:c{','.join(map(str, sample.cost))}" for sample in self.own)
        self.debug = f"t={self.turn} at={me.target}/{me.eta} score={me.score}:{opponent.score} xp={me.expertise} stock={me.storage} pool={available}"
        self.debug += f" enemy={opponent.target}/{opponent.eta}:{opponent.expertise}:{opponent.storage}"
        self.debug += f" own=[{details}] projects={sorted(self.claimed)} travel+={self.travel_extra} why={self.reason}"
        if self.turn == 1:
            self.debug += f" goals={self.projects}"
        if revealed:
            self.debug += f" revealed={revealed}"
        if plan is not None:
            self.debug += f" plan={plan.order} batch={plan.batch} take={plan.takes} finish={plan.completions} pts={plan.points} rate={plan.rating:.2f}"
        self.debug += f" action={action} ms={(perf_counter() - started) * 1000:.2f}"
        return action

    def act(self) -> str:
        """Selects diagnosis, collection, production, replenishment, or endgame denial."""
        if self.me.eta > 0:
            self.reason = "travel"
            return "WAIT"
        if self.me.target not in MODULES:
            self.reason = "opening"
            return "GOTO SAMPLES"
        unknown = [sample for sample in self.own if sample.health < 0]
        if self.me.target == "LABORATORY" and self.known:
            ready = self.search(self.known)
            if ready is not None and ready.action.startswith("CONNECT"):
                return self.use(ready, "produce")
        if self.me.target == "SAMPLES":
            if len(self.own) < 3 and self.remaining >= self.new_sample_time():
                self.reason = "draw"
                return f"CONNECT {self.choose_rank()}"
            if unknown:
                self.reason = "diagnose"
                return "GOTO DIAGNOSIS"
        if unknown and self.known and (self.deferred_diagnosis or self.me.target == "DIAGNOSIS" and self.remaining <= 30):
            known_plan = self.search(self.known)
            if known_plan is not None:
                delayed = self.search(self.known, delay=len(unknown))
                if self.deferred_diagnosis or delayed is None or delayed.points < known_plan.points or \
                        known_plan.points >= 30 and self.remaining - known_plan.duration <= len(unknown) + 1:
                    self.deferred_diagnosis = known_plan.order
                    return self.use(known_plan, "deliver-before-diagnosis")
        if unknown and (self.remaining > 10 or not self.known and self.me.target != "DIAGNOSIS"):
            diagnosis_time = self.travel(self.me.target, "DIAGNOSIS") + self.travel("DIAGNOSIS", "LABORATORY") + 2
            if self.remaining < diagnosis_time:
                return self.deny()
            if self.me.target != "DIAGNOSIS":
                self.reason = "diagnose"
                return "GOTO DIAGNOSIS"
            self.reason = "diagnose"
            return f"CONNECT {max(unknown, key=lambda sample: sample.rank).id}"
        if self.me.target == "DIAGNOSIS":
            return self.at_diagnosis(unknown)
        plan = self.search(self.known)
        if self.known and self.me.target == "MOLECULES":
            waiting = self.wait_for_release(plan)
            if waiting is not None:
                return waiting
        if plan is not None:
            if self.me.target == "LABORATORY" and not plan.action.startswith("CONNECT") and self.should_refill(plan):
                self.reason = "refill-batch"
                self.commitment = ()
                return "GOTO SAMPLES"
            return self.use(plan, "produce" if plan.action.startswith("CONNECT") and self.me.target == "LABORATORY" else "batch")
        if self.known and self.me.target != "MOLECULES":
            forecast = self.release_forecast()
            if forecast is not None:
                pool, delay = forecast
                arrival = self.travel(self.me.target, "MOLECULES")
                future = self.search(self.known, pool=pool, release_after=delay)
                if future is not None and delay <= arrival + 2:
                    return self.use(future, "route-to-release")
        if self.known or self.cloud:
            candidate = self.search(self.diagnosis_candidates(), position="DIAGNOSIS", delay=self.travel(self.me.target, "DIAGNOSIS"), exchange=True)
            if candidate is not None:
                self.reason = "exchange"
                self.commitment = ()
                return "GOTO DIAGNOSIS"
        if len(self.own) < 3 and self.remaining >= self.travel(self.me.target, "SAMPLES") + self.new_sample_time():
            self.reason = "restock"
            self.commitment = ()
            return "GOTO SAMPLES"
        if self.known and self.remaining > 25:
            self.reason = "exchange"
            self.commitment = ()
            return "GOTO DIAGNOSIS"
        return self.deny()

    def at_diagnosis(self, unknown: list[Sample]) -> str:
        """Chooses a feasible combination of held and cloud samples, paying exchange costs."""
        if not unknown and self.remaining > 30:
            unreachable = self.unreachable_samples()
            if unreachable:
                self.reason = "drop-locked-sample"
                return f"CONNECT {min(unreachable, key=self.sample_value).id}"
        plan = self.search(self.diagnosis_candidates(), exchange=True)
        if plan is not None:
            missing = [sample_id for sample_id in plan.order if all(sample.id != sample_id for sample in self.own)]
            if missing:
                self.selected = plan
                if len(self.own) == 3:
                    unwanted = [sample for sample in self.own if sample.id not in plan.order]
                    self.reason = "swap-out"
                    dropped = min(unwanted, key=lambda sample: (sample.health < 0, self.sample_value(sample)))
                    return f"CONNECT {dropped.id}"
                self.reason = "cloud-pickup"
                return f"CONNECT {missing[0]}"
            return self.use(plan, "diagnosed-batch")
        if unknown and self.remaining > self.travel("DIAGNOSIS", "LABORATORY") + 1:
            self.reason = "diagnose-last"
            return f"CONNECT {unknown[0].id}"
        if self.known and self.remaining > self.new_sample_time() + 4:
            # A rival already heading to production may make this hand feasible shortly.
            returned = self.release_forecast()
            if returned is not None:
                pool, delay = returned
                future = self.search(self.known, pool=pool, release_after=delay)
                if future is not None and delay <= self.travel("DIAGNOSIS", "MOLECULES") + 2:
                    return self.use(future, "incoming-release")
            if len(self.own) < 3 and self.remaining >= self.travel("DIAGNOSIS", "SAMPLES") + self.new_sample_time():
                viable = [sample for sample in self.known if max(sample.cost[i] - self.me.expertise[i] for i in range(5)) <= 5]
                if len(viable) == len(self.known):
                    self.reason = "refill-blocked-hand"
                    return "GOTO SAMPLES"
            self.reason = "drop-unworkable"
            self.commitment = ()
            return f"CONNECT {min(self.known, key=self.sample_value).id}"
        if len(self.own) < 3 and self.remaining >= self.travel("DIAGNOSIS", "SAMPLES") + self.new_sample_time():
            self.reason = "replace"
            return "GOTO SAMPLES"
        return self.deny()

    def unreachable_samples(self) -> list[Sample]:
        """Finds samples that no expertise sequence in the current hand can make producible."""
        pending = self.known.copy()
        expertise = list(self.me.expertise)
        while pending:
            for sample in pending:
                need = [max(0, sample.cost[i] - expertise[i]) for i in range(5)]
                if sum(need) <= 10 and all(need[i] <= max(5, self.me.storage[i]) for i in range(5)):
                    expertise[sample.gain] += 1
                    pending.remove(sample)
                    break
            else:
                break
        return pending

    def search(self, candidates: list[Sample], position: str = "", delay: int = 0, exchange: bool = False,
               pool: tuple[int, ...] = (), release_after: int = 0) -> Plan | None:
        """Finds the highest-value feasible ordered subset, including split laboratory visits."""
        position = position or self.me.target
        pool = pool or self.available
        owned = {sample.id for sample in self.own}
        best = None
        committed = None
        exponent = min(1, max(0, (self.remaining - 25) / 50))
        for count in range(1, min(3, len(candidates)) + 1):
            for iteration, order in enumerate(permutations(candidates, count)):
                if iteration % 16 == 0 and perf_counter() > self.deadline:
                    return best
                cloud_count = sum(sample.id not in owned for sample in order)
                transaction = 0
                if exchange:
                    drops = max(0, len(self.own) + cloud_count - 3)
                    diagnosed_drops = sum(sample not in order for sample in self.known)
                    transaction = cloud_count + drops + max(0, drops - diagnosed_drops)
                expertise = list(self.me.expertise)
                requirements = []
                rewards = []
                local_projects = set(self.claimed)
                for sample in order:
                    requirements.append(tuple(max(0, sample.cost[i] - expertise[i]) for i in range(5)))
                    old_expertise = expertise.copy()
                    expertise[sample.gain] += 1
                    points = sample.health
                    shaping = 0
                    completed_projects = []
                    for index, project in enumerate(self.projects):
                        if index in local_projects:
                            continue
                        deficit = sum(max(0, project[i] - expertise[i]) for i in range(5))
                        if deficit == 0:
                            completed_projects.append(index)
                            local_projects.add(index)
                        elif old_expertise[sample.gain] < project[sample.gain] and deficit <= self.remaining / 10:
                            enemy_deficit = sum(max(0, project[i] - self.opponent.expertise[i]) for i in range(5))
                            shaping += 12 / (deficit + 1) * (0.35 if enemy_deficit + 2 < deficit else 1)
                    rewards.append((points, shaping, old_expertise[sample.gain], completed_projects))
                for cuts in range(1 << (count - 1)):
                    storage = list(self.me.storage)
                    supply = list(pool)
                    elapsed = delay + transaction
                    location = position
                    begin = 0
                    first_takes = None
                    first_batch = ()
                    all_takes = [0] * 5
                    action = ""
                    completions = []
                    utility = 0
                    points = 0
                    for end in range(1, count + 1):
                        if end < count and not cuts & (1 << (end - 1)):
                            continue
                        demand = [sum(requirements[j][i] for j in range(begin, end)) for i in range(5)]
                        takes = tuple(max(0, demand[i] - storage[i]) for i in range(5))
                        pickups = sum(takes)
                        if sum(storage) + pickups > 10 or any(takes[i] > 0 and takes[i] > supply[i] for i in range(5)):
                            break
                        if first_takes is None:
                            first_takes = takes
                            first_batch = tuple(sample.id for sample in order[:end])
                            if pickups:
                                action = f"CONNECT {TYPES[self.pick_type(takes)]}" if location == "MOLECULES" else "GOTO MOLECULES"
                            else:
                                action = f"CONNECT {order[0].id}" if location == "LABORATORY" else "GOTO LABORATORY"
                        if pickups:
                            elapsed += self.travel(location, "MOLECULES")
                            safe_pickups = sum(min(takes[i], max(0, self.available[i])) for i in range(5))
                            if safe_pickups < pickups:
                                elapsed = max(elapsed, release_after - safe_pickups)
                            elapsed += pickups
                            location = "MOLECULES"
                            for i in range(5):
                                all_takes[i] += takes[i]
                        elapsed += self.travel(location, "LABORATORY")
                        location = "LABORATORY"
                        for j in range(begin, end):
                            elapsed += 1
                            completions.append(elapsed)
                            reward, shaping, previous_expertise, completed_projects = rewards[j]
                            reward += 50 * sum(self.project_deadlines[index] >= elapsed for index in completed_projects)
                            future = min(20, max(0, (self.remaining - elapsed - 22) / 8))
                            utility += reward + future * 0.7 ** previous_expertise + shaping * min(1, future / 5)
                            points += reward
                        if elapsed > self.remaining:
                            break
                        for i in range(5):
                            storage[i] += takes[i] - demand[i]
                            supply[i] += demand[i] - takes[i]
                        begin = end
                    else:
                        risk = 0
                        if self.opponent.target == "MOLECULES" and self.opponent.eta <= 3 and first_takes is not None:
                            risk = sum(min(all_takes[i], self.enemy_need[i]) / (max(0, pool[i] - all_takes[i]) + 1) for i in range(5))
                        rating = utility / (elapsed + 9 + min(4, risk)) ** exponent
                        if self.remaining > 30:
                            rating += 0.025 * (sum(self.me.storage) - sum(storage))
                            if tuple(sample.id for sample in order) == self.commitment:
                                rating *= 1.035
                        if best is None or (rating, -elapsed) > (best.rating, -best.duration):
                            best = Plan(tuple(sample.id for sample in order), action, first_takes, elapsed, points, rating, tuple(completions), first_batch)
                        if not exchange and position == "MOLECULES" and tuple(sample.id for sample in order) == self.collection_order:
                            if (not self.collection_batch or first_batch == self.collection_batch) and \
                                    (committed is None or (rating, -elapsed) > (committed.rating, -committed.duration)):
                                committed = Plan(self.collection_order, action, first_takes, elapsed, points, rating, tuple(completions), first_batch)
        if committed is not None:
            preserves_batch = all(identity in best.batch for identity in committed.batch)
            dominates = best.points > committed.points and best.duration <= committed.duration
            if not preserves_batch and not dominates:
                return committed
        return best

    def use(self, plan: Plan, reason: str) -> str:
        """Records the selected plan and returns its next module action."""
        self.selected = plan
        self.commitment = plan.order
        if self.me.target == "MOLECULES" and plan.action.startswith("CONNECT"):
            self.collection_order = plan.order
            self.collection_batch = plan.batch
        self.reason = reason
        return plan.action

    def opponent_project_times(self) -> list[int]:
        """Estimates project races that the rival can finish with molecules already held."""
        deadlines = [201] * len(self.projects)
        start = self.opponent.eta + self.travel(self.opponent.target, "LABORATORY")
        for count in range(1, len(self.enemy) + 1):
            for order in permutations(self.enemy, count):
                expertise = list(self.opponent.expertise)
                storage = list(self.opponent.storage)
                for elapsed, sample in enumerate(order, start + 1):
                    need = [max(0, sample.cost[i] - expertise[i]) for i in range(5)]
                    if any(need[i] > storage[i] for i in range(5)):
                        break
                    for i in range(5):
                        storage[i] -= need[i]
                    expertise[sample.gain] += 1
                    for index, project in enumerate(self.projects):
                        if all(expertise[i] >= project[i] for i in range(5)):
                            deadlines[index] = min(deadlines[index], elapsed)
        return deadlines

    def travel(self, origin: str, destination: str) -> int:
        """Returns observed movement duration between the supplied modules."""
        return 0 if origin == destination else DISTANCES[origin][destination] + self.travel_extra

    def pick_type(self, takes: tuple[int, ...]) -> int:
        """Prioritizes required molecules that have little spare supply or rival demand."""
        competing = self.opponent.target == "MOLECULES" and self.opponent.eta <= 3
        return max((i for i in range(5) if takes[i] > 0), key=lambda i: (
            competing and self.enemy_picking[i] > 0 and self.available[i] < 2 * takes[i],
            6 / (max(0, self.available[i] - takes[i]) + 1) + competing * (2 * self.enemy_picking[i] + 0.4 * self.enemy_need[i]) + 0.2 * takes[i]))

    def diagnosis_candidates(self) -> list[Sample]:
        """Keeps held samples and a bounded shortlist of promising cloud alternatives."""
        cloud = sorted(self.cloud, key=self.sample_value, reverse=True)
        # Preserve cheap expertise opportunities alongside valuable high-rank samples.
        shortlist = cloud[:5]
        if len(cloud) > 5:
            extra = min(cloud[5:], key=lambda sample: sum(max(0, sample.cost[i] - self.me.expertise[i]) for i in range(5)))
            shortlist.append(extra)
        return self.known + shortlist

    def sample_value(self, sample: Sample) -> float:
        """Estimates a sample's individual value for shortlisting and exchanging."""
        if sample.health < 0:
            return 0
        need = [max(0, sample.cost[i] - self.me.expertise[i]) for i in range(5)]
        missing = sum(max(0, need[i] - self.me.storage[i]) for i in range(5))
        impossible = sum(max(0, need[i] - self.me.storage[i] - max(0, 5 - self.opponent.storage[i])) for i in range(5))
        project = sum(5 / (1 + sum(max(0, target[i] - self.me.expertise[i]) for i in range(5)))
                      for index, target in enumerate(self.projects) if index not in self.claimed and self.me.expertise[sample.gain] < target[sample.gain])
        growth = min(20, max(0, (self.remaining - 30) / 8)) * 0.7 ** self.me.expertise[sample.gain]
        return (sample.health + growth + project) / (missing + 5 + 4 * impossible)

    def new_sample_time(self) -> int:
        """Estimates the minimum useful horizon for drawing and delivering another sample."""
        return self.travel("SAMPLES", "DIAGNOSIS") + self.travel("DIAGNOSIS", "MOLECULES") + self.travel("MOLECULES", "LABORATORY") + 5

    def choose_rank(self) -> int:
        """Balances early expertise growth with larger medicines as expertise accumulates."""
        expertise = sum(self.me.expertise)
        if expertise < 6:
            rank = 1
        elif sum(min(value, 3) for value in self.me.expertise) >= 12 and min(self.me.expertise) >= 1:
            rank = 3
        else:
            rank = 2
        if sum(self.me.storage) >= 7 and expertise < 7:
            rank = 1
        # Observed recipes improve rank selection without assuming an undocumented catalog.
        recipes = list(self.catalog[rank].values())
        if rank > 1 and len(recipes) >= 5:
            feasible = 0
            for sample in recipes:
                need = [max(0, sample.cost[i] - self.me.expertise[i]) for i in range(5)]
                if max(need) <= 5 and sum(max(0, need[i] - self.me.storage[i]) for i in range(5)) + sum(self.me.storage) <= 10:
                    feasible += 1
            if feasible < len(recipes) / 2:
                rank -= 1
        return rank

    def should_refill(self, plan: Plan) -> bool:
        """Recognizes cheap leftovers worth combining with a fresh batch on a long horizon."""
        if len(self.own) != 1 or self.remaining < 65 or sum(self.me.storage) > 2:
            return False
        return self.known[0].health <= 10 and sum(self.me.expertise) >= 5 and plan.points < 50

    def release_forecast(self) -> tuple[tuple[int, ...], int] | None:
        """Forecasts molecules a rival already at or approaching the laboratory can return."""
        if self.opponent.target != "LABORATORY" or self.enemy_idle >= 12:
            return None
        choices = []
        for count in range(1, len(self.enemy) + 1):
            for order in permutations(self.enemy, count):
                expertise = list(self.opponent.expertise)
                storage = list(self.opponent.storage)
                for sample in order:
                    need = [max(0, sample.cost[i] - expertise[i]) for i in range(5)]
                    if any(need[i] > storage[i] for i in range(5)):
                        break
                    for i in range(5):
                        storage[i] -= need[i]
                    expertise[sample.gain] += 1
                else:
                    choices.append((sum(sample.health + 10 for sample in order), count, tuple(self.opponent.storage[i] - storage[i] for i in range(5))))
        if not choices:
            return None
        value = max(choice[0] for choice in choices)
        choices = [choice for choice in choices if choice[0] == value]
        returned = tuple(min(choice[2][i] for choice in choices) for i in range(5))
        return tuple(self.available[i] + returned[i] for i in range(5)), self.opponent.eta + max(choice[1] for choice in choices)

    def wait_for_release(self, current: Plan | None = None) -> str | None:
        """Collects safe requirements or waits briefly for an imminent useful molecule return."""
        forecast = self.release_forecast()
        if forecast is None or self.stalled >= 4:
            return None
        pool, delay = forecast
        if delay > 4:
            return None
        plan = self.search(self.known, pool=pool, release_after=delay)
        if plan is None or current is not None and plan.rating <= current.rating * 1.015:
            return None
        if not any(plan.takes):
            return self.use(plan, "release-batch")
        self.selected = plan
        takes = tuple(plan.takes[i] if self.available[i] > 0 else 0 for i in range(5))
        if any(takes) and sum(self.me.storage) < 10:
            plan.action = f"CONNECT {TYPES[self.pick_type(takes)]}"
            return self.use(plan, "collect-before-release")
        self.reason = "wait-release"
        return "WAIT"

    def deny(self) -> str:
        """Uses otherwise idle final turns to withhold molecules needed by a rival."""
        if self.me.target != "MOLECULES":
            if self.enemy and self.remaining > self.travel(self.me.target, "MOLECULES") and sum(self.me.storage) < 10:
                self.reason = "endgame-denial-travel"
                return "GOTO MOLECULES"
        elif sum(self.me.storage) < 10:
            needed = [i for i in range(5) if self.enemy_need[i] > 0 and self.available[i] > 0]
            if needed:
                index = max(needed, key=lambda i: self.enemy_need[i] / self.available[i])
                self.reason = "endgame-denial"
                return f"CONNECT {TYPES[index]}"
        self.reason = "no-completable-work"
        return "WAIT"


def read_robot(line: str) -> Robot:
    """Parses a robot observation from one protocol input line."""
    fields = line.split()
    values = list(map(int, fields[1:]))
    return Robot(fields[0], values[0], values[1], tuple(values[2:7]), tuple(values[7:12]))


def main():
    """Reads arena observations until EOF and flushes one legal command per turn."""
    stream = sys.stdin
    line = stream.readline()
    if not line:
        return
    projects = [tuple(map(int, stream.readline().split())) for _ in range(int(line))]
    bot = Bot(projects)
    while line := stream.readline():
        me = read_robot(line)
        opponent = read_robot(stream.readline())
        available = tuple(map(int, stream.readline().split()))
        samples = []
        for _ in range(int(stream.readline())):
            fields = stream.readline().split()
            sample_id, owner, rank = map(int, fields[:3])
            gain = TYPES.index(fields[3]) if fields[3] in TYPES else -1
            samples.append(Sample(sample_id, owner, rank, gain, int(fields[4]), tuple(map(int, fields[5:10]))))
        command = bot.decide(me, opponent, available, samples)
        print(bot.debug, file=sys.stderr, flush=True)
        print(command, flush=True)


if __name__ == "__main__":
    main()
