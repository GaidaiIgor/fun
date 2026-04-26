import sys

MOLECULES = "ABCDE"
IDX = {c: i for i, c in enumerate(MOLECULES)}
TURN_LIMIT = 200
DIST = {
    "START_POS": {"SAMPLES": 2, "DIAGNOSIS": 2, "MOLECULES": 2, "LABORATORY": 2},
    "SAMPLES": {"SAMPLES": 0, "DIAGNOSIS": 3, "MOLECULES": 3, "LABORATORY": 3},
    "DIAGNOSIS": {"SAMPLES": 3, "DIAGNOSIS": 0, "MOLECULES": 3, "LABORATORY": 4},
    "MOLECULES": {"SAMPLES": 3, "DIAGNOSIS": 3, "MOLECULES": 0, "LABORATORY": 3},
    "LABORATORY": {"SAMPLES": 3, "DIAGNOSIS": 4, "MOLECULES": 3, "LABORATORY": 0},
}


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
    return name if name in DIST else "START_POS"


project_count = int(input())
science_projects = [list(map(int, input().split())) for _ in range(project_count)]

prev_signature = None
stall_turns = 0
prev_transition_signature = None
transition_turns = 0
last_blocked_state = None
blocked_waits = 0


def effective_need(sample, robot):
    return [max(0, sample.costs[i] - robot.expertise[i]) for i in range(5)]


def missing_need(sample, robot):
    need = effective_need(sample, robot)
    return [max(0, need[i] - robot.storage[i]) for i in range(5)]


def total_missing(sample, robot):
    return sum(missing_need(sample, robot))


def storage_overlap(sample, robot):
    need = effective_need(sample, robot)
    return sum(min(need[i], robot.storage[i]) for i in range(5))


def wasted_storage(robot, samples):
    if not samples:
        return robot.total_storage
    useful = 0
    for i in range(5):
        useful += min(robot.storage[i], max(effective_need(s, robot)[i] for s in samples))
    return robot.total_storage - useful


def can_finish_now(sample, robot):
    return sample.diagnosed and all(sample.costs[i] <= robot.expertise[i] + robot.storage[i] for i in range(5))


def can_finish_with_empty_bag(sample, robot):
    return sample.diagnosed and sum(effective_need(sample, robot)) <= 10


def can_finish_from_here(sample, robot):
    return sample.diagnosed and total_missing(sample, robot) <= 10 - robot.total_storage


def finish_eta(sample, robot, current_module):
    if not sample.diagnosed:
        return 999
    if can_finish_now(sample, robot):
        return 1 if current_module == "LABORATORY" else DIST[current_module]["LABORATORY"] + 1
    add = total_missing(sample, robot)
    if current_module == "MOLECULES":
        return add + DIST["MOLECULES"]["LABORATORY"] + 1
    if current_module == "LABORATORY":
        return DIST["LABORATORY"]["MOLECULES"] + add + DIST["MOLECULES"]["LABORATORY"] + 1
    return DIST[current_module]["MOLECULES"] + add + DIST["MOLECULES"]["LABORATORY"] + 1


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
            bonus = 10 * (before_missing - after_missing)
            if after_missing == 0:
                bonus += 65
            best = max(best, bonus)
    return best


def desired_rank(robot, turn):
    remaining = TURN_LIMIT - turn
    exp_total = robot.total_expertise
    if remaining <= 22:
        return 1 if exp_total < 4 else 2
    if exp_total < 2 and robot.score < 10:
        return 1
    if exp_total >= 8 and robot.score >= 35 and remaining > 60:
        return 3
    return 2


def desired_sample_count(robot, turn):
    remaining = TURN_LIMIT - turn
    if remaining <= 8:
        return 0
    if remaining <= 22:
        return 1
    if remaining <= 55:
        return 2
    if robot.total_expertise < 2 and robot.score < 10:
        return 2
    return 3


def sample_value(sample, robot, projects, available, turn):
    if not sample.diagnosed:
        return -10_000
    need = effective_need(sample, robot)
    need_total = sum(need)
    if need_total > 10:
        return -20_000 - 50 * (need_total - 10)
    missing = missing_need(sample, robot)
    missing_total = sum(missing)
    scarcity = sum(max(0, missing[i] - available[i]) for i in range(5))
    project_bonus = project_progress_bonus(sample, robot, projects)
    remaining = TURN_LIMIT - turn
    overlap = storage_overlap(sample, robot)

    value = 5.8 * sample.health - 2.3 * need_total - 1.5 * missing_total - 7.5 * scarcity
    value += project_bonus
    value += 8.0 * overlap
    value += 7 - 3 * abs(sample.rank - desired_rank(robot, turn))

    if robot.total_storage > 0 and overlap == 0:
        value -= 25
    if robot.total_storage > 0:
        value -= 8 * wasted_storage(robot, [sample])

    if sample.rank == 1 and sample.health == 1 and (robot.total_expertise >= 2 or robot.score >= 3):
        value -= 42
    elif sample.rank == 1 and robot.total_expertise >= 4 and project_bonus == 0:
        value -= 18

    eta = finish_eta(sample, robot, robot.target)
    if eta > remaining:
        value -= 220
    elif eta + 4 > remaining:
        value -= 40

    if not can_finish_from_here(sample, robot):
        value -= 100
    return value


def best_ready_sample(samples, robot, projects):
    ready = [s for s in samples if s.diagnosed and can_finish_now(s, robot)]
    if not ready:
        return None
    return max(ready, key=lambda s: 4 * s.health + project_progress_bonus(s, robot, projects))


def viable_to_keep(sample, robot, turn):
    return can_finish_with_empty_bag(sample, robot) and finish_eta(sample, robot, "DIAGNOSIS") <= (TURN_LIMIT - turn)


def choose_keep_set(diagnosed, robot, projects, available, turn):
    if not diagnosed:
        return set()
    target_count = min(3, max(1, desired_sample_count(robot, turn)))
    viable = [s for s in diagnosed if viable_to_keep(s, robot, turn)] or diagnosed[:]
    viable.sort(key=lambda s: sample_value(s, robot, projects, available, turn), reverse=True)
    kept = viable[:target_count]
    if robot.total_storage > 0:
        overlapful = [s for s in viable if storage_overlap(s, robot) > 0]
        if overlapful:
            best_overlap = max(overlapful, key=lambda s: (storage_overlap(s, robot), sample_value(s, robot, projects, available, turn)))
            if best_overlap not in kept:
                kept[-1] = best_overlap
    return {s.id for s in kept}


def best_cloud_sample(cloud_samples, robot, projects, available, turn):
    if not cloud_samples:
        return None

    def key(sample):
        return sample_value(sample, robot, projects, available, turn) + 6.0 * storage_overlap(sample, robot)

    good = [s for s in cloud_samples if viable_to_keep(s, robot, turn)]
    if robot.total_storage > 0:
        overlap_good = [s for s in good if storage_overlap(s, robot) > 0 and can_finish_from_here(s, robot)]
        if overlap_good:
            return max(overlap_good, key=key)
    if good:
        best = max(good, key=key)
        if key(best) >= 28:
            return best
    return None


def target_sample(diagnosed, robot, projects, available, turn):
    candidates = [s for s in diagnosed if can_finish_with_empty_bag(s, robot) and can_finish_from_here(s, robot)]
    if not candidates:
        return None

    def key(sample):
        overlap = storage_overlap(sample, robot)
        missing = total_missing(sample, robot)
        scarcity = sum(max(0, missing_need(sample, robot)[i] - available[i]) for i in range(5))
        score = sample.health * 6 + project_progress_bonus(sample, robot, projects)
        score += 9 * overlap - 5 * missing - 7 * scarcity - 3 * abs(sample.rank - desired_rank(robot, turn))
        if robot.total_storage > 0 and overlap == 0:
            score -= 25
        return score

    return max(candidates, key=key)


def best_molecule_choice(diagnosed, opp_samples, me, opp, available, projects, turn):
    if me.total_storage >= 10:
        return None
    opp_pressure = opp.target == "MOLECULES" and opp.eta <= 1
    opp_needs = [0] * 5
    for s in opp_samples:
        if not s.diagnosed:
            continue
        miss = missing_need(s, opp)
        for i in range(5):
            opp_needs[i] += miss[i]

    best_type = None
    best_score = -10**9
    for i, molecule in enumerate(MOLECULES):
        if available[i] <= 0:
            continue
        future = Robot(me.target, me.eta, me.score, me.storage[:], me.expertise[:])
        future.storage[i] += 1
        sample_scores = []
        for s in diagnosed:
            if not can_finish_with_empty_bag(s, future) or not can_finish_from_here(s, future):
                continue
            overlap = storage_overlap(s, future)
            before_overlap = storage_overlap(s, me)
            missing = total_missing(s, future)
            scarcity = sum(max(0, missing_need(s, future)[j] - available[j] + (1 if j == i else 0)) for j in range(5))
            score = 8 * s.health + project_progress_bonus(s, future, projects)
            score += 10 * overlap - 6 * missing - 7 * scarcity
            if can_finish_now(s, future):
                score += 60
            if overlap > before_overlap:
                score += 18
            if me.total_storage > 0 and before_overlap == 0 and overlap == 0:
                score -= 30
            sample_scores.append(score)
        if not sample_scores:
            continue
        score = max(sample_scores)
        if opp_pressure and opp_needs[i] > 0:
            score += 5 * opp_needs[i]
            if available[i] <= opp_needs[i]:
                score += 10
        if score > best_score:
            best_score = score
            best_type = molecule
    return best_type


def worst_diagnosed_to_drop(diagnosed, robot, projects, available, turn):
    if not diagnosed:
        return None

    def key(sample):
        bad = -sample_value(sample, robot, projects, available, turn)
        if not can_finish_from_here(sample, robot):
            bad += 100
        if storage_overlap(sample, robot) == 0 and robot.total_storage > 0:
            bad += 60
        return bad

    return max(diagnosed, key=key)


def should_force_drop_due_to_bag(diagnosed, robot):
    return robot.total_storage > 0 and diagnosed and max(storage_overlap(s, robot) for s in diagnosed) == 0


def anti_stall_override(turn, me, diagnosed, undiagnosed, ready, projects, available):
    if stall_turns < 2:
        return None
    remaining = TURN_LIMIT - turn
    if me.target == "MOLECULES":
        if ready:
            return "GOTO LABORATORY"
        if diagnosed:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES" if remaining > 6 else "WAIT"
    if me.target == "LABORATORY":
        if diagnosed or undiagnosed:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES" if remaining > 6 else "WAIT"
    if me.target == "DIAGNOSIS":
        if diagnosed:
            return f"CONNECT {worst_diagnosed_to_drop(diagnosed, me, projects, available, turn).id}"
        return "GOTO SAMPLES" if remaining > 6 else "WAIT"
    return None


def choose_action(turn, me, opp, available, samples, projects):
    global prev_transition_signature, transition_turns, last_blocked_state, blocked_waits
    remaining = TURN_LIMIT - turn
    my_samples = [s for s in samples if s.carried_by == 0]
    opp_samples = [s for s in samples if s.carried_by == 1]
    cloud_samples = [s for s in samples if s.carried_by == -1 and s.diagnosed]
    undiagnosed = [s for s in my_samples if not s.diagnosed]
    diagnosed = [s for s in my_samples if s.diagnosed]
    ready = [s for s in diagnosed if can_finish_now(s, me)]
    keep_set = choose_keep_set(diagnosed, me, projects, available, turn)
    forced_bad = [s for s in diagnosed if not can_finish_from_here(s, me)]
    bag_bad = should_force_drop_due_to_bag(diagnosed, me)
    to_drop = forced_bad + [s for s in diagnosed if s.id not in keep_set and s not in forced_bad]
    primary = target_sample(diagnosed, me, projects, available, turn)
    target_count = desired_sample_count(me, turn)
    cloud = best_cloud_sample(cloud_samples, me, projects, available, turn)

    transition_sig = (me.target, tuple(me.storage), tuple(sorted(s.id for s in diagnosed)))
    if transition_sig == prev_transition_signature:
        transition_turns += 1
    else:
        transition_turns = 0
    prev_transition_signature = transition_sig

    override = anti_stall_override(turn, me, diagnosed, undiagnosed, ready, projects, available)
    if override is not None:
        return override

    if remaining <= 4 and not my_samples and not ready:
        return "WAIT"

    if me.target == "START_POS":
        if ready:
            return "GOTO LABORATORY"
        if my_samples:
            return "GOTO DIAGNOSIS" if undiagnosed or to_drop or bag_bad else "GOTO MOLECULES"
        if cloud is not None and me.total_storage > 0:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES" if target_count > 0 else "WAIT"

    if me.target == "SAMPLES":
        if ready:
            return "GOTO LABORATORY"
        if undiagnosed and len(my_samples) >= target_count:
            return "GOTO DIAGNOSIS"
        if len(my_samples) < target_count:
            return f"CONNECT {desired_rank(me, turn)}"
        return "GOTO DIAGNOSIS"

    if me.target == "DIAGNOSIS":
        if ready:
            return "GOTO LABORATORY"
        if undiagnosed:
            return f"CONNECT {undiagnosed[0].id}"
        if bag_bad:
            if cloud is not None and storage_overlap(cloud, me) > 0:
                blocked_waits = 0
                return f"CONNECT {cloud.id}"
            blocked_waits = 0
            return f"CONNECT {worst_diagnosed_to_drop(diagnosed, me, projects, available, turn).id}"
        if blocked_waits >= 1 and diagnosed and me.total_storage > 0:
            overlap_cloud = cloud is not None and storage_overlap(cloud, me) > max(storage_overlap(s, me) for s in diagnosed)
            if overlap_cloud:
                blocked_waits = 0
                return f"CONNECT {cloud.id}"
            blocked_waits = 0
            return f"CONNECT {worst_diagnosed_to_drop(diagnosed, me, projects, available, turn).id}"
        if transition_turns >= 2 and diagnosed:
            blocked_waits = 0
            return f"CONNECT {worst_diagnosed_to_drop(diagnosed, me, projects, available, turn).id}"
        if to_drop:
            blocked_waits = 0
            return f"CONNECT {worst_diagnosed_to_drop(to_drop, me, projects, available, turn).id}"
        if diagnosed and primary is None:
            blocked_waits = 0
            return f"CONNECT {worst_diagnosed_to_drop(diagnosed, me, projects, available, turn).id}"
        if len(my_samples) < target_count:
            if cloud is not None:
                blocked_waits = 0
                return f"CONNECT {cloud.id}"
            if remaining > 24 and not diagnosed:
                return "GOTO SAMPLES"
        if diagnosed:
            return "GOTO MOLECULES"
        if me.total_storage > 0 and cloud is not None:
            return f"CONNECT {cloud.id}"
        if len(my_samples) < target_count:
            return "GOTO SAMPLES"
        return "GOTO SAMPLES" if remaining > 6 else "WAIT"

    if me.target == "MOLECULES":
        best_ready = best_ready_sample(my_samples, me, projects)
        if best_ready is not None:
            return "GOTO LABORATORY"
        if undiagnosed:
            return "GOTO DIAGNOSIS"
        if bag_bad:
            return "GOTO DIAGNOSIS"
        if primary is None:
            if diagnosed:
                return "GOTO DIAGNOSIS"
            if me.total_storage > 0 and cloud is not None and remaining > 10:
                return "GOTO DIAGNOSIS"
            return "GOTO SAMPLES" if target_count > 0 else ("GOTO DIAGNOSIS" if me.total_storage > 0 else "WAIT")
        molecule = best_molecule_choice(diagnosed, opp_samples, me, opp, available, projects, turn)
        if molecule is not None:
            last_blocked_state = None
            blocked_waits = 0
            return f"CONNECT {molecule}"
        blocked_state = (tuple(sorted(s.id for s in diagnosed)), tuple(me.storage), tuple(available))
        if blocked_state == last_blocked_state:
            blocked_waits += 1
        else:
            last_blocked_state = blocked_state
            blocked_waits = 0
        overlap = max(storage_overlap(s, me) for s in diagnosed) if diagnosed else 0
        if overlap > 0 and blocked_waits == 0 and remaining > 8:
            return "WAIT"
        return "GOTO DIAGNOSIS"

    if me.target == "LABORATORY":
        best_ready = best_ready_sample(my_samples, me, projects)
        if best_ready is not None:
            return f"CONNECT {best_ready.id}"
        if undiagnosed or to_drop or bag_bad:
            return "GOTO DIAGNOSIS"
        if diagnosed:
            return "GOTO MOLECULES" if primary is not None else "GOTO DIAGNOSIS"
        if cloud is not None and me.total_storage > 0 and remaining > 10:
            return "GOTO DIAGNOSIS"
        if len(my_samples) < target_count or (not my_samples and me.total_storage > 0 and remaining > 12):
            return "GOTO SAMPLES"
        return "GOTO DIAGNOSIS" if me.total_storage > 0 and remaining > 6 else "WAIT"

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

    my_diag = tuple(sorted(s.id for s in samples if s.carried_by == 0 and s.diagnosed))
    my_und = tuple(sorted(s.id for s in samples if s.carried_by == 0 and not s.diagnosed))
    signature = (me.target, tuple(me.storage), my_diag, my_und)
    if signature == prev_signature:
        stall_turns += 1
    else:
        stall_turns = 0
    prev_signature = signature

    action = choose_action(turn, me, opp, available, samples, science_projects)
    print(action)
    print(
        f"t={turn} me={me.target} score={me.score} exp={me.expertise} st={me.storage} diag={list(my_diag)} und={list(my_und)} stall={stall_turns} trans={transition_turns} bw={blocked_waits} act={action}",
        file=sys.stderr,
    )
