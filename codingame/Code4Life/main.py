import sys
from itertools import combinations, permutations

MOLS = "ABCDE"
SAMP = "SAMPLES"
DIAG = "DIAGNOSIS"
MOLMOD = "MOLECULES"
LAB = "LABORATORY"

# Persistent memory across turns.
turn = 0
dropped_ids = set()
blocked_molecule_turns = 0


def mi(g):
    return MOLS.find(g)


class Robot:
    def __init__(self, line):
        p = line.split()
        self.target = p[0]
        self.eta = int(p[1])
        self.score = int(p[2])
        self.storage = list(map(int, p[3:8]))
        self.expertise = list(map(int, p[8:13]))


class Sample:
    def __init__(self, line):
        p = line.split()
        self.id = int(p[0])
        self.carried_by = int(p[1])
        self.rank = int(p[2])
        self.gain = p[3]
        self.health = int(p[4])
        self.cost = list(map(int, p[5:10]))

    @property
    def diagnosed(self):
        return self.health >= 0 and all(c >= 0 for c in self.cost)


def storage_total(storage):
    return sum(storage)


def can_make(sample, storage, exp):
    return sample.diagnosed and all(storage[i] + exp[i] >= sample.cost[i] for i in range(5))


def effective_cost(sample, exp):
    return [max(0, sample.cost[i] - exp[i]) for i in range(5)]


def order_requirements(order, exp):
    """Total molecules needed for this completion order, including later expertise."""
    e = exp[:]
    req = [0] * 5

    for s in order:
        for i in range(5):
            req[i] += max(0, s.cost[i] - e[i])

        gi = mi(s.gain)
        if gi >= 0:
            e[gi] += 1

    return req, e


def feasible_with_storage(req, storage):
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]
    return sum(storage) + sum(to_collect) <= 10


def project_bonus(exp, final_exp, projects):
    bonus = 0.0

    for pr in projects:
        before_done = all(exp[i] >= pr[i] for i in range(5))
        after_done = all(final_exp[i] >= pr[i] for i in range(5))

        if after_done and not before_done:
            # Full value is 50, discounted because opponent can also complete projects.
            bonus += 32.0

        before_gap = sum(max(0, pr[i] - exp[i]) for i in range(5))
        after_gap = sum(max(0, pr[i] - final_exp[i]) for i in range(5))
        bonus += 1.2 * max(0, before_gap - after_gap)

    return bonus


def sample_gain_bonus(sample, exp, projects):
    gi = mi(sample.gain)
    if gi < 0:
        return 0.0

    b = 0.0

    if exp[gi] == 0:
        b += 8.0
    elif exp[gi] == 1:
        b += 5.0
    elif exp[gi] < 4:
        b += 2.0

    for pr in projects:
        if exp[gi] < pr[gi]:
            b += 2.0
            break

    return b


def plan_value(order, req, final_exp, storage, avail, exp, projects, remaining):
    health = sum(s.health for s in order)
    total_req = sum(req)
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]
    shortage = [max(0, to_collect[i] - avail[i]) for i in range(5)]

    low_health_penalty = 0.0
    for s in order:
        if s.health <= 1 and remaining < 120:
            low_health_penalty += 4.0
        if s.health <= 1 and remaining < 70:
            low_health_penalty += 12.0

    scarcity_penalty = sum(max(0, 2 - avail[i]) * to_collect[i] * 0.8 for i in range(5))
    shortage_penalty = 3.0 * sum(shortage)
    batch_bonus = 2.5 * (len(order) - 1)

    return (
        health
        + sum(sample_gain_bonus(s, exp, projects) for s in order)
        + project_bonus(exp, final_exp, projects)
        + batch_bonus
        - 0.18 * total_req
        - 0.25 * sum(to_collect)
        - scarcity_penalty
        - shortage_penalty
        - low_health_penalty
    )


def select_best_plan(samples, exp, storage, avail, projects, remaining, require_available=False):
    diagnosed = [s for s in samples if s.diagnosed]
    best = None

    for r in range(1, len(diagnosed) + 1):
        for subset in combinations(diagnosed, r):
            for order in permutations(subset):
                req, final_exp = order_requirements(order, exp)

                if not feasible_with_storage(req, storage):
                    continue

                to_collect = [max(0, req[i] - storage[i]) for i in range(5)]

                if require_available and any(to_collect[i] > avail[i] for i in range(5)):
                    continue

                val = plan_value(order, req, final_exp, storage, avail, exp, projects, remaining)

                if best is None or val > best["value"]:
                    best = {
                        "order": list(order),
                        "ids": [s.id for s in order],
                        "req": req,
                        "to_collect": to_collect,
                        "final_exp": final_exp,
                        "value": val,
                    }

    return best


def best_single_value(s, exp, storage, avail, projects, remaining):
    if not s.diagnosed:
        return -10**9

    p = select_best_plan([s], exp, storage, avail, projects, remaining, False)
    return p["value"] if p else -10**9


def impossible_sample(s, exp):
    ec = effective_cost(s, exp)
    return max(ec) > 5 or sum(ec) > 10


def choose_rank(exp, remaining):
    et = sum(exp)

    if remaining < 30:
        return 1 if et < 5 else 2

    if et < 2:
        return 1

    if et < 7:
        return 2

    if et >= 9 and remaining > 55:
        return 3

    return 2


def choose_cloud_sample(cloud, carried, exp, storage, avail, projects, remaining):
    if len(carried) >= 3:
        return None

    current = select_best_plan(carried, exp, storage, avail, projects, remaining, False)
    current_val = current["value"] if current else 0.0

    best = None

    for s in cloud:
        if s.id in dropped_ids or not s.diagnosed:
            continue

        single_val = best_single_value(s, exp, storage, avail, projects, remaining)

        if s.health <= 1 and single_val < 8:
            continue

        combined = select_best_plan(carried + [s], exp, storage, avail, projects, remaining, False)

        if combined is None:
            continue

        improvement = combined["value"] - current_val
        score = single_val + 0.8 * improvement

        if best is None or score > best[0]:
            best = (score, s)

    if best is not None and best[0] > 12:
        return best[1]

    return None


def choose_drop_sample(carried, exp, storage, avail, projects, remaining):
    diagnosed = [s for s in carried if s.diagnosed]

    if not diagnosed:
        return None

    possible = select_best_plan(diagnosed, exp, storage, avail, projects, remaining, False)

    if possible is not None:
        return None

    candidates = []

    for s in diagnosed:
        val = best_single_value(s, exp, storage, avail, projects, remaining)
        badness = -val

        if impossible_sample(s, exp):
            badness += 100

        if s.health <= 1:
            badness += 10

        candidates.append((badness, s))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def makeable_samples(carried, storage, exp):
    return [s for s in carried if s.diagnosed and can_make(s, storage, exp)]


def choose_molecule(plan, storage, avail, exp):
    if plan is None or storage_total(storage) >= 10:
        return None

    req = plan["req"]
    need = [max(0, req[i] - storage[i]) for i in range(5)]

    if sum(need) == 0:
        return None

    first_need = [0] * 5

    if plan["order"]:
        first_req, _ = order_requirements([plan["order"][0]], exp)
        first_need = [max(0, first_req[i] - storage[i]) for i in range(5)]

    best_i = None
    best_score = -10**9

    for i in range(5):
        if need[i] <= 0 or avail[i] <= 0:
            continue

        score = 40 * min(1, first_need[i]) + 10 * need[i] + 3 * max(0, 4 - avail[i])

        if score > best_score:
            best_score = score
            best_i = i

    return MOLS[best_i] if best_i is not None else None


def goto(module, me):
    if me.target == module:
        return "WAIT"

    return "GOTO " + module


def summarize_samples(samples):
    out = []

    for s in samples:
        if s.diagnosed:
            out.append(
                "{}:r{}:{}:{}:{}".format(
                    s.id,
                    s.rank,
                    s.health,
                    s.gain,
                    "".join(map(str, s.cost)),
                )
            )
        else:
            out.append("{}:r{}:?".format(s.id, s.rank))

    return "[" + ",".join(out) + "]"


def decide(projects, robots, avail, samples):
    global blocked_molecule_turns, dropped_ids, turn

    me = robots[0]
    remaining = 200 - turn

    carried = [s for s in samples if s.carried_by == 0]
    diagnosed = [s for s in carried if s.diagnosed]
    undiagnosed = [s for s in carried if not s.diagnosed]
    cloud = [s for s in samples if s.carried_by == -1 and s.diagnosed]

    if me.eta > 0:
        return "WAIT", None

    strict_plan = select_best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        remaining,
        True,
    )

    loose_plan = strict_plan or select_best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        remaining,
        False,
    )

    makeable = makeable_samples(diagnosed, me.storage, me.expertise)

    if me.target == LAB:
        blocked_molecule_turns = 0

        if makeable:
            if loose_plan:
                for s in loose_plan["order"]:
                    if can_make(s, me.storage, me.expertise):
                        return "CONNECT {}".format(s.id), loose_plan

            best = max(
                makeable,
                key=lambda s: best_single_value(
                    s,
                    me.expertise,
                    me.storage,
                    avail,
                    projects,
                    remaining,
                ),
            )

            return "CONNECT {}".format(best.id), loose_plan

        if diagnosed:
            return goto(MOLMOD, me), loose_plan

        if undiagnosed:
            return goto(DIAG, me), loose_plan

        return goto(SAMP, me), loose_plan

    if me.target == MOLMOD:
        if makeable:
            blocked_molecule_turns = 0
            return goto(LAB, me), loose_plan

        if not diagnosed:
            blocked_molecule_turns = 0
            return goto(DIAG if undiagnosed else SAMP, me), loose_plan

        mol = choose_molecule(loose_plan, me.storage, avail, me.expertise)

        if mol is not None:
            blocked_molecule_turns = 0
            return "CONNECT {}".format(mol), loose_plan

        if loose_plan is not None and blocked_molecule_turns < 3 and remaining > 20:
            blocked_molecule_turns += 1
            return "WAIT", loose_plan

        blocked_molecule_turns = 0

        if loose_plan is not None and storage_total(me.storage) > 0:
            return goto(LAB, me), loose_plan

        return goto(DIAG, me), loose_plan

    if me.target == DIAG:
        blocked_molecule_turns = 0

        if undiagnosed:
            s = max(undiagnosed, key=lambda x: (x.rank, x.id))
            return "CONNECT {}".format(s.id), loose_plan

        # Critical fix:
        # If any viable carried plan exists, leave Diagnosis.
        # Do not upload diagnosed samples merely because they are not in the current best subset.
        if makeable:
            return goto(LAB, me), loose_plan

        if loose_plan is not None:
            return goto(MOLMOD, me), loose_plan

        if len(carried) < 3:
            s = choose_cloud_sample(
                cloud,
                carried,
                me.expertise,
                me.storage,
                avail,
                projects,
                remaining,
            )

            if s is not None:
                return "CONNECT {}".format(s.id), loose_plan

        bad = choose_drop_sample(
            carried,
            me.expertise,
            me.storage,
            avail,
            projects,
            remaining,
        )

        if bad is not None:
            dropped_ids.add(bad.id)
            return "CONNECT {}".format(bad.id), loose_plan

        if len(carried) < 3 and remaining > 18:
            return goto(SAMP, me), loose_plan

        return "WAIT", loose_plan

    if me.target == SAMP:
        blocked_molecule_turns = 0

        if len(carried) < 3 and remaining > 18:
            return "CONNECT {}".format(choose_rank(me.expertise, remaining)), loose_plan

        return goto(DIAG, me), loose_plan

    blocked_molecule_turns = 0

    if makeable:
        return goto(LAB, me), loose_plan

    if diagnosed:
        return goto(MOLMOD, me), loose_plan

    if undiagnosed:
        return goto(DIAG, me), loose_plan

    return goto(SAMP, me), loose_plan


project_count = int(input())
projects = [list(map(int, input().split())) for _ in range(project_count)]

while True:
    try:
        robots = [Robot(input()), Robot(input())]
    except EOFError:
        break

    available = list(map(int, input().split()))
    sample_count = int(input())
    samples = [Sample(input()) for _ in range(sample_count)]

    action, plan = decide(projects, robots, available, samples)

    me = robots[0]
    carried = [s for s in samples if s.carried_by == 0]

    plan_s = "None" if plan is None else "ids={} req={} val={:.1f}".format(
        plan["ids"],
        plan["req"],
        plan["value"],
    )

    print(
        "T{} {} eta={} score={} store={} exp={} avail={} carried={} plan={} dropped={} -> {}".format(
            turn,
            me.target,
            me.eta,
            me.score,
            me.storage,
            me.expertise,
            available,
            summarize_samples(carried),
            plan_s,
            sorted(dropped_ids),
            action,
        ),
        file=sys.stderr,
    )

    print(action)
    turn += 1