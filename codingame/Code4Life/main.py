"""Plays a two-robot medicine production arena from standard input."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from sys import stderr


MOLECULES = ("A", "B", "C", "D", "E")
MODULES = {"SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY"}
MOLECULE_INDEX = {molecule: index for index, molecule in enumerate(MOLECULES)}
FULL_STOCK = (5, 5, 5, 5, 5)
NO_STOCK = (0, 0, 0, 0, 0)
MAX_SAMPLES = 3
MAX_STORAGE = 10
GAME_TURNS = 200
STOCK_WAIT_LIMIT = 3
SAMPLE_TOP_UP_LIMIT = GAME_TURNS - 45
Vector = tuple[int, ...]
Projects = list[Vector]


@dataclass(slots=True)
class Robot:
    target: str
    eta: int
    score: int
    storage: Vector
    expertise: Vector


@dataclass(slots=True)
class Sample:
    sample_id: int
    carried_by: int
    rank: int
    gain: str
    health: int
    cost: Vector

    @property
    def diagnosed(self) -> bool:
        return self.health > 0 and min(self.cost) >= 0


@dataclass(slots=True)
class Plan:
    samples: tuple[Sample, ...]
    missing: Vector
    value: float


def main():
    try:
        project_count = int(input())
    except EOFError:
        return
    projects = [tuple(map(int, input().split())) for _ in range(project_count)]
    print(f"projects={";".join(vector_text(project) for project in projects)}", file=stderr)
    rejected_ids = set()
    stalled_ids = set()
    stock_waits = {}
    turn = 0
    while True:
        try:
            robots = [read_robot(input()) for _ in range(2)]
        except EOFError:
            return
        available = tuple(map(int, input().split()))
        samples = [read_sample(input()) for _ in range(int(input()))]
        action, reason = choose_action(turn, projects, rejected_ids, stalled_ids, stock_waits, robots, available, samples)
        log_turn(turn, robots, available, samples, action, reason)
        print(action)
        turn += 1


def choose_action(turn: int, projects: Projects, rejected_ids: set[int], stalled_ids: set[int], stock_waits: dict[tuple[int, ...], int], \
        robots: list[Robot], available: Vector, samples: list[Sample]) -> tuple[str, str]:
    me = robots[0]
    mine = [sample for sample in samples if sample.carried_by == 0]
    diagnosed = [sample for sample in mine if sample.diagnosed]
    undiagnosed = [sample for sample in mine if not sample.diagnosed]
    if me.eta > 0:
        return "WAIT", f"traveling eta={me.eta}"
    if me.target not in MODULES:
        return "GOTO SAMPLES", "leaving start"
    if me.target == "SAMPLES":
        return choose_samples_action(turn, me, mine)
    if me.target == "DIAGNOSIS":
        return choose_diagnosis_action(turn, projects, rejected_ids, stalled_ids, me, available, samples, mine, diagnosed, undiagnosed)
    if me.target == "MOLECULES":
        return choose_molecules_action(turn, projects, stalled_ids, stock_waits, me, robots[1], available, diagnosed)
    return choose_laboratory_action(turn, projects, rejected_ids, stalled_ids, me, available, diagnosed, undiagnosed)


def choose_samples_action(turn: int, robot: Robot, mine: list[Sample]) -> tuple[str, str]:
    if turn > GAME_TURNS - 22 and not mine:
        return "WAIT", "too late for a new sample cycle"
    if len(mine) < desired_sample_count(turn):
        rank = desired_rank(turn, robot)
        return f"CONNECT {rank}", f"taking rank {rank}"
    return "GOTO DIAGNOSIS", "sample rack full enough"


def choose_diagnosis_action(turn: int, projects: Projects, rejected_ids: set[int], stalled_ids: set[int], robot: Robot, available: Vector, \
        samples: list[Sample], mine: list[Sample], diagnosed: list[Sample], undiagnosed: list[Sample]) -> tuple[str, str]:
    if undiagnosed:
        sample = max(undiagnosed, key=lambda item: (item.rank, -item.sample_id))
        return f"CONNECT {sample.sample_id}", f"diagnosing {sample.sample_id}"
    stalled = [sample for sample in diagnosed if sample.sample_id in stalled_ids]
    if stalled:
        sample = worst_sample(projects, robot, stalled)
        rejected_ids.add(sample.sample_id)
        stalled_ids.discard(sample.sample_id)
        return f"CONNECT {sample.sample_id}", f"dropping stalled sample {sample.sample_id}"
    bad = worst_rejected_sample(projects, robot, diagnosed)
    if bad is not None:
        rejected_ids.add(bad.sample_id)
        return f"CONNECT {bad.sample_id}", f"dropping weak sample {bad.sample_id}"
    if len(mine) < MAX_SAMPLES:
        cloud = choose_cloud_sample(turn, projects, rejected_ids, robot, samples, diagnosed)
        if cloud is not None:
            return f"CONNECT {cloud.sample_id}", f"taking cloud sample {cloud.sample_id}"
    plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, available, True, False)
    if plan is not None:
        if any(sample_can_finish(sample, robot.storage, robot.expertise) for sample in diagnosed):
            return "GOTO LABORATORY", "already have medicine ready"
        return "GOTO MOLECULES", "diagnosed work is viable"
    plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, FULL_STOCK, True, False)
    if plan is not None:
        if next_molecule(plan, available) is not None:
            return "GOTO MOLECULES", "collecting partial blocked plan"
        if len(mine) < MAX_SAMPLES and turn <= SAMPLE_TOP_UP_LIMIT:
            return "GOTO SAMPLES", "stock blocked, filling open sample slot"
    if diagnosed:
        sample = worst_sample(projects, robot, diagnosed)
        rejected_ids.add(sample.sample_id)
        return f"CONNECT {sample.sample_id}", f"no timely viable plan, dropping {sample.sample_id}"
    if turn > GAME_TURNS - 22:
        return "WAIT", "no time to restart"
    return "GOTO SAMPLES", "need samples"


def choose_molecules_action(turn: int, projects: Projects, stalled_ids: set[int], stock_waits: dict[tuple[int, ...], int], robot: Robot, \
        opponent: Robot, available: Vector, diagnosed: list[Sample]) -> tuple[str, str]:
    ready_plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, NO_STOCK, True, True)
    plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, FULL_STOCK, True, False)
    if plan is None:
        if ready_plan is not None:
            return "GOTO LABORATORY", "finish reachable sample"
        return "GOTO DIAGNOSIS", "molecule stock blocks current samples"
    if sum(plan.missing) == 0:
        return "GOTO LABORATORY", "all planned molecules ready"
    molecule = next_molecule(plan, available)
    if molecule is not None:
        stock_waits.clear()
        return f"CONNECT {molecule}", f"collecting {molecule} for {[sample.sample_id for sample in plan.samples]}"
    if ready_plan is not None:
        return "GOTO LABORATORY", "finish ready subset before stock wait"
    key = tuple(sample.sample_id for sample in plan.samples) + (-1,) + \
        tuple(index for index in range(len(MOLECULES)) if plan.missing[index] > 0 and available[index] == 0)
    stock_waits[key] = stock_waits.get(key, 0) + 1
    wait_limit = STOCK_WAIT_LIMIT + (3 if opponent.target == "LABORATORY" and opponent.eta <= 2 else 0)
    if stock_waits[key] <= wait_limit and turn <= GAME_TURNS - 28:
        return "WAIT", f"waiting for stock {stock_waits[key]}"
    if len(diagnosed) < MAX_SAMPLES and turn <= SAMPLE_TOP_UP_LIMIT:
        stock_waits.clear()
        return "GOTO SAMPLES", "stock blocked, filling open sample slot"
    sample = worst_sample(projects, robot, list(plan.samples))
    stalled_ids.add(sample.sample_id)
    return "GOTO DIAGNOSIS", f"stock stalled sample {sample.sample_id}"


def choose_laboratory_action(turn: int, projects: Projects, rejected_ids: set[int], stalled_ids: set[int], robot: Robot, available: Vector, \
        diagnosed: list[Sample], undiagnosed: list[Sample]) -> tuple[str, str]:
    plan = completion_plan(projects, robot, diagnosed, NO_STOCK, True, True)
    if plan is not None:
        return f"CONNECT {plan.samples[0].sample_id}", f"researching {plan.samples[0].sample_id}"
    if diagnosed:
        current_plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, available, True, False)
        if current_plan is not None:
            return "GOTO MOLECULES", "need more molecules"
        future_plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, FULL_STOCK, True, False)
        if future_plan is None:
            sample = worst_sample(projects, robot, diagnosed)
            rejected_ids.add(sample.sample_id)
            stalled_ids.add(sample.sample_id)
            return "GOTO DIAGNOSIS", f"dropping lab leftover {sample.sample_id}"
        if next_molecule(future_plan, available) is None and len(diagnosed) + len(undiagnosed) < MAX_SAMPLES and turn <= SAMPLE_TOP_UP_LIMIT:
            return "GOTO SAMPLES", "leftover stock blocked, filling sample slot"
        return "GOTO MOLECULES", "need more molecules"
    if undiagnosed:
        return "GOTO DIAGNOSIS", "carrying undiagnosed samples"
    return "GOTO SAMPLES", "cycle complete"


def completion_plan(projects: Projects, robot: Robot, samples: list[Sample], available: Vector, require_stock: bool, require_ready: bool) -> Plan | None:
    best = None
    for size in range(1, len(samples) + 1):
        for order in permutations(samples, size):
            plan = evaluate_order(projects, robot, order, available, require_stock)
            if plan is not None and (not require_ready or sum(plan.missing) == 0) and better_plan(plan, best):
                best = plan
    return best


def timely_completion_plan(turn: int, module: str, projects: Projects, robot: Robot, samples: list[Sample], available: Vector, \
        require_stock: bool, require_ready: bool) -> Plan | None:
    best = None
    for size in range(1, len(samples) + 1):
        for order in permutations(samples, size):
            plan = evaluate_order(projects, robot, order, available, require_stock)
            if plan is not None and (not require_ready or sum(plan.missing) == 0) and enough_time(turn, module, plan) and better_plan(plan, best):
                best = plan
    return best


def choose_cloud_sample(turn: int, projects: Projects, rejected_ids: set[int], robot: Robot, samples: list[Sample], diagnosed: list[Sample]) -> Sample | None:
    best = None
    best_plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed, FULL_STOCK, True, False)
    for sample in samples:
        if sample.carried_by == -1 and sample.diagnosed and sample.sample_id not in rejected_ids and not sample_is_bad(projects, robot, sample):
            plan = timely_completion_plan(turn, robot.target, projects, robot, diagnosed + [sample], FULL_STOCK, True, False)
            if plan is not None and sample in plan.samples and better_plan(plan, best_plan):
                best = sample
                best_plan = plan
    return best


def worst_rejected_sample(projects: Projects, robot: Robot, samples: list[Sample]) -> Sample | None:
    bad_samples = [sample for sample in samples if sample_is_bad(projects, robot, sample)]
    if not bad_samples:
        return None
    return worst_sample(projects, robot, bad_samples)


def sample_is_bad(projects: Projects, robot: Robot, sample: Sample) -> bool:
    need = effective_need(sample, robot.expertise)
    if sum(need) > MAX_STORAGE or any(need[index] > FULL_STOCK[index] + robot.storage[index] for index in range(len(MOLECULES))):
        return True
    return sample.health <= 1 and sum(robot.expertise) >= 3 and gain_bonus(projects, robot.expertise, sample.gain) < 10


def worst_sample(projects: Projects, robot: Robot, samples: list[Sample]) -> Sample:
    return min(samples, key=lambda sample: sample_value(projects, robot.expertise, sample) - sum(effective_need(sample, robot.expertise)))


def evaluate_order(projects: Projects, robot: Robot, order: tuple[Sample, ...], available: Vector, require_stock: bool) -> Plan | None:
    inventory = list(robot.storage)
    expertise = list(robot.expertise)
    missing = [0, 0, 0, 0, 0]
    value = 0
    for sample in order:
        if not sample.diagnosed:
            return None
        need = effective_need(sample, tuple(expertise))
        for index in range(len(MOLECULES)):
            if inventory[index] < need[index]:
                missing[index] += need[index] - inventory[index]
                inventory[index] = need[index]
            inventory[index] -= need[index]
        value += sample_value(projects, tuple(expertise), sample)
        expertise[MOLECULE_INDEX[sample.gain]] += 1
    if sum(robot.storage) + sum(missing) > MAX_STORAGE:
        return None
    if require_stock and any(missing[index] > available[index] for index in range(len(MOLECULES))):
        return None
    return Plan(order, tuple(missing), value)


def better_plan(plan: Plan, best: Plan | None) -> bool:
    if best is None:
        return True
    plan_missing = sum(plan.missing)
    best_missing = sum(best.missing)
    return (plan.value - plan_missing, plan.value, len(plan.samples), -plan_missing) > (best.value - best_missing, best.value, len(best.samples), -best_missing)


def next_molecule(plan: Plan, available: Vector) -> str | None:
    indexes = [index for index in range(len(MOLECULES)) if plan.missing[index] > 0 and available[index] > 0]
    if not indexes:
        return None
    index = min(indexes, key=lambda item: (available[item], -plan.missing[item]))
    return MOLECULES[index]


def enough_time(turn: int, module: str, plan: Plan) -> bool:
    return turn + estimated_plan_turns(module, plan) < GAME_TURNS


def estimated_plan_turns(module: str, plan: Plan) -> int:
    missing = sum(plan.missing)
    if module == "DIAGNOSIS":
        return (3 + missing + 3 if missing > 0 else 4) + len(plan.samples)
    if module == "MOLECULES":
        return missing + 3 + len(plan.samples)
    if module == "LABORATORY":
        return len(plan.samples) if missing == 0 else 3 + missing + 3 + len(plan.samples)
    return 2 + 3 + missing + 3 + len(plan.samples)


def sample_can_finish(sample: Sample, storage: Vector, expertise: Vector) -> bool:
    need = effective_need(sample, expertise)
    return all(storage[index] >= need[index] for index in range(len(MOLECULES)))


def effective_need(sample: Sample, expertise: Vector) -> Vector:
    return tuple(max(0, sample.cost[index] - expertise[index]) for index in range(len(MOLECULES)))


def sample_value(projects: Projects, expertise: Vector, sample: Sample) -> float:
    return sample.health + gain_bonus(projects, expertise, sample.gain)


def gain_bonus(projects: Projects, expertise: Vector, gain: str) -> float:
    index = MOLECULE_INDEX[gain]
    bonus = 3 if expertise[index] < 3 else 0
    for project in projects:
        missing = [max(0, project[item] - expertise[item]) for item in range(len(MOLECULES))]
        total_missing = sum(missing)
        if total_missing == 0 or missing[index] == 0:
            continue
        bonus = max(bonus, 50 if total_missing == 1 else 8 + 32 / total_missing)
    return bonus


def desired_sample_count(turn: int) -> int:
    if turn > GAME_TURNS - 30:
        return 1
    return 2 if turn > GAME_TURNS - 50 else MAX_SAMPLES


def desired_rank(turn: int, robot: Robot) -> int:
    expertise = sum(robot.expertise)
    if turn > GAME_TURNS - 28:
        return 1 if expertise < 8 else 2
    if expertise < 3:
        return 1
    if expertise < 8:
        return 2
    return 3


def read_robot(line: str) -> Robot:
    parts = line.split()
    return Robot(parts[0], int(parts[1]), int(parts[2]), tuple(map(int, parts[3:8])), tuple(map(int, parts[8:13])))


def read_sample(line: str) -> Sample:
    parts = line.split()
    return Sample(int(parts[0]), int(parts[1]), int(parts[2]), parts[3], int(parts[4]), tuple(map(int, parts[5:10])))


def log_turn(turn: int, robots: list[Robot], available: Vector, samples: list[Sample], action: str, reason: str):
    robot = robots[0]
    opponent = robots[1]
    mine = [sample for sample in samples if sample.carried_by == 0]
    sample_text = ";".join(sample_log(sample) for sample in mine) or "none"
    print(f"T{turn} {robot.target}/{robot.eta} score={robot.score} opp={opponent.target}/{opponent.eta}/{opponent.score} " \
          f"avail={vector_text(available)} storage={vector_text(robot.storage)} exp={vector_text(robot.expertise)} " \
          f"samples={sample_text} action={action} reason={reason}", file=stderr)


def sample_log(sample: Sample) -> str:
    status = "diag" if sample.diagnosed else "raw"
    return f"{sample.sample_id}:r{sample.rank}:{status}:h{sample.health}:g{sample.gain}:c{vector_text(sample.cost)}"


def vector_text(values: Vector) -> str:
    return ",".join(str(value) for value in values)


if __name__ == "__main__":
    main()
