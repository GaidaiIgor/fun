import sys
from itertools import combinations, permutations

MOLS = "ABCDE"
LAB = "LABORATORY"
DIAG = "DIAGNOSIS"
MOLMOD = "MOLECULES"
SAMP = "SAMPLES"


def mi(g):
    return MOLS.find(g)


class Robot:
    def __init__(self, line):
        parts = line.split()
        self.target = parts[0]
        self.eta = int(parts[1])
        self.score = int(parts[2])
        self.storage = list(map(int, parts[3:8]))
        self.expertise = list(map(int, parts[8:13]))


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


def can_make(sample, storage, exp):
    return sample.diagnosed and all(storage[i] + exp[i] >= sample.cost[i] for i in range(5))


def eff_cost(sample, exp):
    return [max(0, sample.cost[i] - exp[i]) for i in range(5)]


def order_requirements(order, exp):
    """
    Molecules needed to finish samples in this order, accounting for expertise
    gained after each completed sample.
    """
    e = exp[:]
    req = [0] * 5

    for s in order:
        for i in range(5):
            req[i] += max(0, s.cost[i] - e[i])

        gi = mi(s.gain)
        if gi >= 0:
            e[gi] += 1

    return req, e


def feasible_capacity(req, storage):
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]
    return sum(storage) + sum(to_collect) <= 10


def project_gain_bonus(exp, final_exp, projects):
    # Approximation: we do not know whether opponent has already completed a project.
    bonus = 0.0

    for pr in projects:
        before_ok = all(exp[i] >= pr[i] for i in range(5))
        after_ok = all(final_exp[i] >= pr[i] for i in range(5))

        if after_ok and not before_ok:
            bonus += 35.0

        before_def = sum(max(0, pr[i] - exp[i]) for i in range(5))
        after_def = sum(max(0, pr[i] - final_exp[i]) for i in range(5))
        bonus += 1.2 * max(0, before_def - after_def)

    return bonus


def gain_bonus_for_sample(sample, exp, projects):
    gi = mi(sample.gain)
    if gi < 0:
        return 0.0

    b = 0.0

    if exp[gi] < 2:
        b += 5.0
    elif exp[gi] < 5:
        b += 2.5
    else:
        b += 0.5

    if any(exp[gi] < pr[gi] for pr in projects):
        b += 3.0

    return b


def plan_value(order, req, final_exp, storage, avail, exp, projects):
    health = sum(s.health for s in order)
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]
    shortage = [max(0, to_collect[i] - avail[i]) for i in range(5)]

    scarcity_penalty = sum(
        req[i] * max(0, 2 - avail[i]) * 0.45
        for i in range(5)
    )

    molecule_penalty = (
        0.20 * sum(req)
        + 0.35 * sum(to_collect)
        + 2.0 * sum(shortage)
    )

    gain_bonus = sum(gain_bonus_for_sample(s, exp, projects) for s in order)
    science_bonus = project_gain_bonus(exp, final_exp, projects)
    batch_bonus = 1.5 * (len(order) - 1)

    return health + gain_bonus + science_bonus + batch_bonus - molecule_penalty - scarcity_penalty


def select_best_plan(samples, exp, storage, avail, projects, require_available):
    diagnosed = [s for s in samples if s.diagnosed]
    best = None

    for r in range(1, len(diagnosed) + 1):
        for subset in combinations(diagnosed, r):
            for order in permutations(subset):
                req, final_exp = order_requirements(order, exp)

                if not feasible_capacity(req, storage):
                    continue

                to_collect = [max(0, req[i] - storage[i]) for i in range(5)]

                if require_available and any(to_collect[i] > avail[i] for i in range(5)):
                    continue

                val = plan_value(order, req, final_exp, storage, avail, exp, projects)

                if best is None or val > best["value"]:
                    best = {
                        "order": list(order),
                        "ids": [s.id for s in order],
                        "req": req,
                        "to_collect": to_collect,
                        "value": val,
                        "final_exp": final_exp,
                    }

    return best


def sample_value(sample, exp, storage, avail, projects):
    if not sample.diagnosed:
        return -999.0

    req, final_exp = order_requirements([sample], exp)

    if not feasible_capacity(req, storage):
        return -999.0

    return plan_value([sample], req, final_exp, storage, avail, exp, projects)


def choose_rank(exp, turn):
    et = sum(exp)
    remaining = 200 - turn

    if remaining < 30:
        return 1 if et < 8 else 2

    if et < 5:
        return 1

    if et < 11:
        return 2

    if remaining > 55 and et >= 11:
        return 3

    return 2


def choose_cloud_sample(cloud, carried, exp, storage, avail, projects):
    if len(carried) >= 3:
        return None

    current_plan = select_best_plan(
        carried, exp, storage, avail, projects, require_available=False
    )
    current_val = current_plan["value"] if current_plan else -50

    best = None

    for s in cloud:
        if not s.diagnosed:
            continue

        val = sample_value(s, exp, storage, avail, projects)

        # Avoid taking low-value junk from the cloud.
        if val < 5 and s.health <= 10:
            continue

        combined_plan = select_best_plan(
            carried + [s], exp, storage, avail, projects, require_available=False
        )
        combined_val = combined_plan["value"] if combined_plan else val
        improvement = combined_val - current_val
        score = val + 0.5 * improvement

        if best is None or score > best[0]:
            best = (score, s)

    return best[1] if best else None


def choose_bad_sample_to_drop(carried, exp, storage, avail, projects, plan):
    plan_ids = set(plan["ids"]) if plan else set()
    candidates = []

    for s in carried:
        if not s.diagnosed:
            continue

        val = sample_value(s, exp, storage, avail, projects)
        req = eff_cost(s, exp)

        impossible_now = sum(req) > 10
        trash_health = s.health <= 1 and sum(req) > 2
        not_in_plan_and_weak = s.id not in plan_ids and val < 8

        if impossible_now or trash_health or not_in_plan_and_weak:
            candidates.append((val, s))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def choose_molecule(plan, storage, avail, exp):
    if not plan or sum(storage) >= 10:
        return None

    req = plan["req"]
    need = [max(0, req[i] - storage[i]) for i in range(5)]

    first = plan["order"][0] if plan["order"] else None
    first_need = [0] * 5

    if first is not None:
        first_req, _ = order_requirements([first], exp)
        first_need = [max(0, first_req[i] - storage[i]) for i in range(5)]

    best_i = None
    best_score = -10**9

    for i in range(5):
        if need[i] <= 0 or avail[i] <= 0:
            continue

        # Enable the next medicine first, but also grab scarce molecules early.
        score = (
            25 * min(1, first_need[i])
            + 8 * need[i]
            + 4 * max(0, 3 - avail[i])
        )

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None

    return MOLS[best_i]


def goto(module, me):
    if me.target == module:
        return "WAIT"
    return "GOTO " + module


def action_for_turn(turn, projects, robots, avail, samples):
    me = robots[0]

    carried = [s for s in samples if s.carried_by == 0]
    cloud = [s for s in samples if s.carried_by == -1 and s.diagnosed]

    diagnosed = [s for s in carried if s.diagnosed]
    undiagnosed = [s for s in carried if not s.diagnosed]

    if me.eta > 0:
        return "WAIT"

    makeable = [
        s for s in diagnosed
        if can_make(s, me.storage, me.expertise)
    ]

    strict_plan = select_best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        require_available=True,
    )

    loose_plan = strict_plan or select_best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        require_available=False,
    )

    if me.target == LAB:
        if makeable:
            if loose_plan:
                for s in loose_plan["order"]:
                    if can_make(s, me.storage, me.expertise):
                        return "CONNECT {}".format(s.id)

            best = max(
                makeable,
                key=lambda s: sample_value(
                    s, me.expertise, me.storage, avail, projects
                ),
            )
            return "CONNECT {}".format(best.id)

        if diagnosed:
            return goto(MOLMOD, me)

        if undiagnosed:
            return goto(DIAG, me)

        return goto(SAMP, me)

    if me.target == MOLMOD:
        if not diagnosed:
            return goto(DIAG if undiagnosed else SAMP, me)

        mol = choose_molecule(loose_plan, me.storage, avail, me.expertise)

        if mol is not None:
            return "CONNECT {}".format(mol)

        if makeable:
            return goto(LAB, me)

        # Avoid waiting forever on impossible molecule availability.
        return goto(DIAG, me)

    if me.target == DIAG:
        if undiagnosed:
            s = max(undiagnosed, key=lambda x: x.rank)
            return "CONNECT {}".format(s.id)

        bad = choose_bad_sample_to_drop(
            carried,
            me.expertise,
            me.storage,
            avail,
            projects,
            loose_plan,
        )

        if bad is not None and (
            len(carried) >= 3
            or sample_value(bad, me.expertise, me.storage, avail, projects) < 3
        ):
            return "CONNECT {}".format(bad.id)

        if len(carried) < 3:
            s = choose_cloud_sample(
                cloud,
                carried,
                me.expertise,
                me.storage,
                avail,
                projects,
            )

            if s is not None:
                return "CONNECT {}".format(s.id)

        makeable = [
            s for s in diagnosed
            if can_make(s, me.storage, me.expertise)
        ]

        if makeable:
            return goto(LAB, me)

        if diagnosed:
            return goto(MOLMOD, me)

        if len(carried) < 3 and 200 - turn > 15:
            return goto(SAMP, me)

        return "WAIT"

    if me.target == SAMP:
        if len(carried) < 3 and 200 - turn > 15:
            return "CONNECT {}".format(choose_rank(me.expertise, turn))

        return goto(DIAG, me)

    # START_POS or unexpected target.
    if makeable:
        return goto(LAB, me)

    if undiagnosed:
        return goto(DIAG, me)

    if diagnosed:
        return goto(MOLMOD, me)

    return goto(SAMP, me)


# --- Main loop ---

project_count = int(input())
PROJECTS = [
    list(map(int, input().split()))
    for _ in range(project_count)
]

turn = 0

while True:
    try:
        robots = [Robot(input()), Robot(input())]
    except EOFError:
        break

    available = list(map(int, input().split()))

    sample_count = int(input())
    samples = [
        Sample(input())
        for _ in range(sample_count)
    ]

    act = action_for_turn(
        turn,
        PROJECTS,
        robots,
        available,
        samples,
    )

    me = robots[0]

    print(
        "T{} {} eta={} score={} store={} exp={} avail={} -> {}".format(
            turn,
            me.target,
            me.eta,
            me.score,
            me.storage,
            me.expertise,
            available,
            act,
        ),
        file=sys.stderr,
    )

    print(act)
    turn += 1