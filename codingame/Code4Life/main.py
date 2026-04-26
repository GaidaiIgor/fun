import sys

MOLECULES = "ABCDE"
IDX = {c: i for i, c in enumerate(MOLECULES)}
TURN_LIMIT = 200


class Robot:
    def __init__(self, target, eta, score, storage, expertise):
        self.target = target
        self.eta = eta
        self.score = score
        self.storage = storage
        self.expertise = expertise

    @property
    def total_storage(self):
        return sum(self.storage)

    @property
    def total_expertise(self):
        return sum(self.expertise)


class Sample:
    def __init__(self, sample_id, carried_by, rank, gain, health, costs):
        self.id = sample_id
        self.carried_by = carried_by
        self.rank = rank
        self.gain = gain if gain in IDX else None
        self.health = health
        self.costs = costs

    @property
    def diagnosed(self):
        return self.health != -1


def normalize_module(name):
    if name in ("SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY"):
        return name
    return "START_POS"


project_count = int(input())
science_projects = []
for _ in range(project_count):
    science_projects.append([int(x) for x in input().split()])


def effective_need(sample, robot):
    return [max(0, sample.costs[i] - robot.expertise[i]) for i in range(5)]


def missing_need(sample, robot):
    need = effective_need(sample, robot)
    return [max(0, need[i] - robot.storage[i]) for i in range(5)]


def total_missing(sample, robot):
    return sum(missing_need(sample, robot))


def can_finish_now(sample, robot):
    if not sample.diagnosed:
        return False
    for i in range(5):
        if sample.costs[i] > robot.expertise[i] + robot.storage[i]:
            return False
    return True


def can_finish_with_empty_bag(sample, robot):
    return sample.diagnosed and sum(effective_need(sample, robot)) <= 10


def project_progress_bonus(sample, robot, projects):
    if sample.gain is None:
        return 0
    idx = IDX[sample.gain]
    after = robot.expertise[:]
    after[idx] += 1
    best = 0
    for project in projects:
        before_missing = sum(max(0, project[i] - robot.expertise[i]) for i in range(5))
        after_missing = sum(max(0, project[i] - after[i]) for i in range(5))
        if after_missing < before_missing:
            bonus = 14 * (before_missing - after_missing)
            if after_missing == 0:
                bonus += 80
            best = max(best, bonus)
    return best


def desired_rank(robot, turn):
    remaining = TURN_LIMIT - turn
    exp_total = robot.total_expertise
    if remaining <= 28:
        return 1 if exp_total < 7 else 2
    if exp_total < 4:
        return 1
    if exp_total < 12:
        return 2
    if remaining > 70 and exp_total >= 12:
        return 3
    return 2


def desired_sample_count(turn):
    remaining = TURN_LIMIT - turn
    if remaining <= 20:
        return 0
    if remaining <= 35:
        return 1
    if remaining <= 55:
        return 2
    return 3


def sample_value(sample, robot, projects, available, turn):
    if not sample.diagnosed:
        return -10_000
    need = effective_need(sample, robot)
    need_total = sum(need)
    if need_total > 10:
        return -5_000 - 50 * (need_total - 10)

    missing = [max(0, need[i] - robot.storage[i]) for i in range(5)]
    missing_total = sum(missing)
    scarcity = sum(max(0, missing[i] - available[i]) for i in range(5))
    stage = robot.total_expertise
    health_weight = 2.1 if stage < 6 else 2.7 if stage < 10 else 3.2
    expertise_bonus = 16 if stage < 4 else 10 if stage < 8 else 5
    rank_pref = 10 - 6 * abs(sample.rank - desired_rank(robot, turn))
    project_bonus = project_progress_bonus(sample, robot, projects)

    value = (
        health_weight * sample.health
        + expertise_bonus
        + rank_pref
        + project_bonus
        - 3.2 * need_total
        - 1.2 * missing_total
        - 8.0 * scarcity
    )

    remaining = TURN_LIMIT - turn
    rough_finish_time = 7 + missing_total
    if not can_finish_now(sample, robot) and rough_finish_time > remaining:
        value -= 100

    if sample.rank == 1 and sample.health == 1 and stage >= 6 and project_bonus == 0:
        value -= 14

    return value


def lab_value(sample, robot, projects):
    return 4 * sample.health + project_progress_bonus(sample, robot, projects)


def find_bad_sample(samples, robot, projects, available, turn):
    diagnosed = [s for s in samples if s.diagnosed]
    if not diagnosed:
        return None
    worst = min(diagnosed, key=lambda s: sample_value(s, robot, projects, available, turn))
    value = sample_value(worst, robot, projects, available, turn)
    if value < 12:
        return worst
    return None


def best_cloud_sample(cloud_samples, robot, projects, available, turn):
    good = [s for s in cloud_samples if can_finish_with_empty_bag(s, robot)]
    if not good:
        return None
    best = max(good, key=lambda s: sample_value(s, robot, projects, available, turn))
    if sample_value(best, robot, projects, available, turn) >= 24:
        return best
    return None


def best_lab_sample(samples, robot, projects):
    ready = [s for s in samples if s.diagnosed and can_finish_now(s, robot)]
    if not ready:
        return None
    return max(ready, key=lambda s: lab_value(s, robot, projects))


def opponent_remaining_needs(opp_samples, opp_robot):
    needs = [0] * 5
    for sample in opp_samples:
        if not sample.diagnosed:
            continue
        for i in range(5):
            rem = sample.costs[i] - opp_robot.expertise[i] - opp_robot.storage[i]
            if rem > 0:
                needs[i] += rem
    return needs


def best_molecule_type(my_samples, opp_samples, me, opp, available, projects, turn):
    contenders = [s for s in my_samples if s.diagnosed and can_finish_with_empty_bag(s, me)]
    if not contenders or me.total_storage >= 10:
        return None

    opp_needs = opponent_remaining_needs(opp_samples, opp)
    opp_pressure = opp.target == "MOLECULES" and opp.eta <= 2

    best_type = None
    best_score = -10**9
    for i, molecule in enumerate(MOLECULES):
        if available[i] <= 0:
            continue
        score = 0.0
        for sample in contenders:
            missing = missing_need(sample, me)
            if missing[i] <= 0:
                continue
            sample_priority = 3.5 * sample.health + 1.5 * project_progress_bonus(sample, me, projects)
            sample_priority += 15 if sum(missing) == 1 else 0
            sample_priority += 4 * sample.rank
            score += sample_priority / max(1, sum(missing))
        if score <= 0:
            continue

        score += 3 * (5 - available[i])
        if opp_pressure:
            score += 6 * opp_needs[i]
            if opp_needs[i] > 0 and available[i] <= opp_needs[i]:
                score += 12
        if best_type is None or score > best_score:
            best_type = molecule
            best_score = score
    return best_type


def choose_action(turn, me, opp, available, samples, projects):
    my_samples = [s for s in samples if s.carried_by == 0]
    opp_samples = [s for s in samples if s.carried_by == 1]
    cloud_samples = [s for s in samples if s.carried_by == -1 and s.diagnosed]
    undiagnosed = [s for s in my_samples if not s.diagnosed]
    diagnosed = [s for s in my_samples if s.diagnosed]
    ready = [s for s in diagnosed if can_finish_now(s, me)]
    bad_sample = find_bad_sample(my_samples, me, projects, available, turn)
    desired_count = desired_sample_count(turn)

    if me.target == "START_POS":
        if my_samples:
            return "GOTO DIAGNOSIS"
        if desired_count == 0:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES"

    if me.target == "SAMPLES":
        if len(my_samples) < desired_count:
            return f"CONNECT {desired_rank(me, turn)}"
        if undiagnosed or bad_sample is not None:
            return "GOTO DIAGNOSIS"
        if ready:
            return "GOTO LABORATORY"
        if diagnosed:
            return "GOTO MOLECULES"
        return "GOTO DIAGNOSIS"

    if me.target == "DIAGNOSIS":
        if ready:
            return "GOTO LABORATORY"
        if undiagnosed:
            return f"CONNECT {undiagnosed[0].id}"
        if bad_sample is not None:
            return f"CONNECT {bad_sample.id}"
        cloud = best_cloud_sample(cloud_samples, me, projects, available, turn)
        if len(my_samples) < desired_count and cloud is not None:
            return f"CONNECT {cloud.id}"
        if diagnosed:
            return "GOTO MOLECULES"
        if len(my_samples) < desired_count:
            return "GOTO SAMPLES"
        return "GOTO SAMPLES"

    if me.target == "MOLECULES":
        if ready:
            return "GOTO LABORATORY"
        if undiagnosed or not diagnosed:
            return "GOTO DIAGNOSIS"
        molecule = best_molecule_type(my_samples, opp_samples, me, opp, available, projects, turn)
        if molecule is not None:
            return f"CONNECT {molecule}"
        if me.total_storage >= 10:
            return "GOTO DIAGNOSIS"
        if bad_sample is not None:
            return "GOTO DIAGNOSIS"
        return "GOTO LABORATORY" if ready else "GOTO DIAGNOSIS"

    if me.target == "LABORATORY":
        best = best_lab_sample(my_samples, me, projects)
        if best is not None:
            return f"CONNECT {best.id}"
        if undiagnosed or bad_sample is not None:
            return "GOTO DIAGNOSIS"
        if diagnosed:
            return "GOTO MOLECULES"
        if len(my_samples) < desired_count:
            return "GOTO SAMPLES"
        return "GOTO DIAGNOSIS"

    return "WAIT"


turn = 0
while True:
    turn += 1
    try:
        robot_lines = [input() for _ in range(2)]
    except EOFError:
        break

    robots = []
    for line in robot_lines:
        parts = line.split()
        target = normalize_module(parts[0])
        eta = int(parts[1])
        score = int(parts[2])
        storage = list(map(int, parts[3:8]))
        expertise = list(map(int, parts[8:13]))
        robots.append(Robot(target, eta, score, storage, expertise))
    me, opp = robots

    available = list(map(int, input().split()))
    sample_count = int(input())
    samples = []
    for _ in range(sample_count):
        parts = input().split()
        sample_id = int(parts[0])
        carried_by = int(parts[1])
        rank = int(parts[2])
        gain = parts[3]
        health = int(parts[4])
        costs = list(map(int, parts[5:10]))
        samples.append(Sample(sample_id, carried_by, rank, gain, health, costs))

    if me.eta > 0:
        print("WAIT")
        continue

    action = choose_action(turn, me, opp, available, samples, science_projects)
    my_diag = [s.id for s in samples if s.carried_by == 0 and s.diagnosed]
    my_undiag = [s.id for s in samples if s.carried_by == 0 and not s.diagnosed]
    print(action)
    print(
        f"t={turn} me={me.target} score={me.score} exp={me.expertise} st={me.storage} diag={my_diag} und={my_undiag} act={action}",
        file=sys.stderr,
    )