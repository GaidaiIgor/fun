import sys
from itertools import combinations, permutations

MOLS = "ABCDE"
SAMP = "SAMPLES"
DIAG = "DIAGNOSIS"
MOLMOD = "MOLECULES"
LAB = "LABORATORY"

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


def effective_cost(sample, exp):
    return [max(0, sample.cost[i] - exp[i]) for i in range(5)]


def can_make(sample, storage, exp):
    return sample.diagnosed and all(storage[i] + exp[i] >= sample.cost[i] for i in range(5))


def order_requirements(order, exp):
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

    if sum(storage) + sum(to_collect) > 10:
        return False

    # Not an absolute game rule, but a strong practical rule:
    # if one molecule type requirement is absurdly high, it is usually not
    # collectable in one trip unless we already hold some of it.
    if any(req[i] > 10 for i in range(5)):
        return False

    return True


def estimate_finish_turns(order, to_collect, target):
    mol_actions = sum(to_collect)
    lab_actions = len(order)

    if not order:
        return 999

    if mol_actions == 0:
        if target == LAB:
            return lab_actions
        if target == MOLMOD:
            return 3 + lab_actions
        if target == DIAG:
            return 4 + lab_actions
        if target == SAMP:
            return 3 + lab_actions
        return 2 + lab_actions

    if target == MOLMOD:
        return mol_actions + 3 + lab_actions
    if target == LAB:
        return 3 + mol_actions + 3 + lab_actions
    if target == DIAG:
        return 3 + mol_actions + 3 + lab_actions
    if target == SAMP:
        return 3 + mol_actions + 3 + lab_actions

    return 2 + mol_actions + 3 + lab_actions


def project_deficit(exp, project):
    return [max(0, project[i] - exp[i]) for i in range(5)]


def project_complete(exp, project):
    return all(exp[i] >= project[i] for i in range(5))


def max_project_need(projects):
    mx = [0] * 5
    for pr in projects:
        for i in range(5):
            mx[i] = max(mx[i], pr[i])
    return mx


def project_progress_bonus(exp, final_exp, projects):
    bonus = 0.0

    for pr in projects:
        before = project_deficit(exp, pr)
        after = project_deficit(final_exp, pr)

        before_sum = sum(before)
        after_sum = sum(after)

        if before_sum > 0 and after_sum == 0:
            # Science projects are worth 50. Slightly overweight them because
            # they are also denial against the opponent.
            bonus += 70.0

        bonus += 4.0 * max(0, before_sum - after_sum)

        before_max = max(before) if before else 0
        after_max = max(after) if after else 0
        bonus += 3.0 * max(0, before_max - after_max)

    return bonus


def gain_bonus(sample, exp, projects):
    gi = mi(sample.gain)
    if gi < 0:
        return 0.0

    mx_need = max_project_need(projects)
    b = 0.0

    # Diversity matters. The previous bot got murdered by one missing expertise.
    if exp[gi] == min(exp):
        b += 7.0

    if exp[gi] == 0:
        b += 8.0
    elif exp[gi] == 1:
        b += 5.0
    elif exp[gi] == 2:
        b += 2.5

    for pr in projects:
        deficit = project_deficit(exp, pr)
        gap = sum(deficit)

        if exp[gi] < pr[gi]:
            b += 7.0
            b += 2.0 * (pr[gi] - exp[gi])
            b += 12.0 / (1.0 + gap)

    # Stop worshipping already-saturated expertise types.
    if exp[gi] >= mx_need[gi] and exp[gi] >= 4:
        b -= 7.0

    return b


def time_and_supply_penalty(req, storage, avail, remaining):
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]
    shortage = [max(0, to_collect[i] - avail[i]) for i in range(5)]

    p = 0.0

    p += 0.20 * sum(req)
    p += 0.35 * sum(to_collect)

    # Shortages are survivable early, toxic late.
    if remaining > 80:
        p += 2.0 * sum(shortage)
    elif remaining > 45:
        p += 6.0 * sum(shortage)
    else:
        p += 25.0 * sum(shortage)

    # The v2 failure: chasing a huge single-type requirement late.
    for i in range(5):
        if req[i] >= 7 and storage[i] + avail[i] < req[i]:
            if remaining < 80:
                p += 40.0
            else:
                p += 12.0

    # Prefer not to deplete already-low terminal types unless it buys real value.
    for i in range(5):
        if to_collect[i] > 0 and avail[i] <= 1:
            p += 2.5 * to_collect[i]

    return p


def plan_value(order, req, final_exp, storage, avail, exp, projects, remaining, target):
    health = sum(s.health for s in order)
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]
    est = estimate_finish_turns(order, to_collect, target)

    value = health
    value += sum(gain_bonus(s, exp, projects) for s in order)
    value += project_progress_bonus(exp, final_exp, projects)
    value += 2.5 * (len(order) - 1)
    value -= time_and_supply_penalty(req, storage, avail, remaining)

    if est + 2 > remaining:
        value -= 100.0 + 5.0 * (est + 2 - remaining)

    # Rank-1 one-point samples are only acceptable early or when their gain is useful.
    for s in order:
        if s.health <= 1:
            if remaining < 130:
                value -= 5.0
            if remaining < 70:
                value -= 15.0

    return value


def late_supply_blocked(req, storage, avail, remaining):
    to_collect = [max(0, req[i] - storage[i]) for i in range(5)]

    if remaining < 50 and any(to_collect[i] > avail[i] for i in range(5)):
        return True

    if remaining < 80 and any(req[i] >= 7 and storage[i] + avail[i] < req[i] for i in range(5)):
        return True

    return False


def select_best_plan(samples, exp, storage, avail, projects, remaining, target, require_available=False):
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

                if late_supply_blocked(req, storage, avail, remaining):
                    continue

                est = estimate_finish_turns(order, to_collect, target)
                if est + 1 > remaining:
                    continue

                val = plan_value(order, req, final_exp, storage, avail, exp, projects, remaining, target)

                if best is None or val > best["value"]:
                    best = {
                        "order": list(order),
                        "ids": [s.id for s in order],
                        "req": req,
                        "to_collect": to_collect,
                        "final_exp": final_exp,
                        "value": val,
                        "est": est,
                    }

    return best


def best_single_value(s, exp, storage, avail, projects, remaining, target):
    p = select_best_plan([s], exp, storage, avail, projects, remaining, target, False)
    return p["value"] if p else -10**9


def choose_rank(exp, projects, remaining):
    et = sum(exp)

    # Early: build cheap expertise.
    if et < 3:
        return 1

    # Midgame: rank 2 is the workhorse. The previous bot reached rank 3,
    # but too eagerly chased awkward high-requirement leftovers.
    if et < 9:
        return 2

    # Rank 3 only when we have decent breadth and enough time.
    if remaining > 55 and min(exp) >= 1 and et >= 9:
        return 3

    return 2


def choose_completion_sample(makeable, exp, storage, projects):
    best = None

    for s in makeable:
        e2 = exp[:]
        gi = mi(s.gain)
        if gi >= 0:
            e2[gi] += 1

        score = s.health + gain_bonus(s, exp, projects) + project_progress_bonus(exp, e2, projects)

        if best is None or score > best[0]:
            best = (score, s)

    return best[1] if best else None


def choose_cloud_sample(cloud, carried, exp, storage, avail, projects, remaining, target):
    if len(carried) >= 3:
        return None

    current = select_best_plan(carried, exp, storage, avail, projects, remaining, target, False)
    current_val = current["value"] if current else 0.0

    best = None

    for s in cloud:
        if s.id in dropped_ids or not s.diagnosed:
            continue

        combined = select_best_plan(carried + [s], exp, storage, avail, projects, remaining, target, False)
        if combined is None:
            continue

        single_val = best_single_value(s, exp, storage, avail, projects, remaining, target)
        improvement = combined["value"] - current_val
        score = single_val + 0.8 * improvement

        if best is None or score > best[0]:
            best = (score, s)

    if best is not None and best[0] > 10:
        return best[1]

    return None


def choose_drop_sample(carried, exp, storage, avail, projects, remaining, target):
    diagnosed = [s for s in carried if s.diagnosed]

    if not diagnosed:
        return None

    possible = select_best_plan(diagnosed, exp, storage, avail, projects, remaining, target, False)

    if possible is not None:
        return None

    candidates = []

    for s in diagnosed:
        ec = effective_cost(s, exp)
        val = best_single_value(s, exp, storage, avail, projects, remaining, target)
        badness = -val

        if sum(ec) > 10:
            badness += 100

        if late_supply_blocked(ec, [0, 0, 0, 0, 0], avail, remaining):
            badness += 50

        if s.health <= 1:
            badness += 10

        candidates.append((badness, s))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def makeable_samples(carried, storage, exp):
    return [s for s in carried if s.diagnosed and can_make(s, storage, exp)]


def choose_molecule(plan, storage, avail, exp, projects):
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

        score = 0.0

        # Prioritize enabling the next medicine.
        score += 45.0 * min(1, first_need[i])

        # Then finish the whole plan.
        score += 10.0 * need[i]

        # Scarce molecules are worth grabbing before the opponent does.
        score += 3.0 * max(0, 4 - avail[i])

        # But if the science-project bottleneck needs this molecule expertise,
        # collecting molecules for samples that produce such expertise is already
        # handled in plan selection. Do not double-count too much here.

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
    global blocked_molecule_turns, turn, dropped_ids

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
        me.target,
        True,
    )

    loose_plan = strict_plan or select_best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        remaining,
        me.target,
        False,
    )

    makeable = makeable_samples(diagnosed, me.storage, me.expertise)

    if me.target == LAB:
        blocked_molecule_turns = 0

        if makeable:
            s = choose_completion_sample(makeable, me.expertise, me.storage, projects)
            return "CONNECT {}".format(s.id), loose_plan

        # Critical v3 rule: do not stand in Lab without a producible sample.
        if loose_plan is not None:
            return goto(MOLMOD, me), loose_plan

        if undiagnosed:
            return goto(DIAG, me), loose_plan

        if remaining > 15:
            return goto(SAMP, me), loose_plan

        return "WAIT", loose_plan

    if me.target == MOLMOD:
        if makeable:
            # If collecting more molecules would complete a much better batch and
            # there is enough time, continue. Otherwise bank points now.
            mol = choose_molecule(loose_plan, me.storage, avail, me.expertise, projects)

            if mol is not None and loose_plan is not None and loose_plan["est"] + 2 < remaining:
                return "CONNECT {}".format(mol), loose_plan

            blocked_molecule_turns = 0
            return goto(LAB, me), loose_plan

        if not diagnosed:
            blocked_molecule_turns = 0
            return goto(DIAG if undiagnosed else SAMP, me), loose_plan

        mol = choose_molecule(loose_plan, me.storage, avail, me.expertise, projects)

        if mol is not None:
            blocked_molecule_turns = 0
            return "CONNECT {}".format(mol), loose_plan

        # v2 bug fix: do not go to Lab unless something is makeable.
        # Waiting a little is okay early/midgame if the plan is blocked by scarcity.
        if loose_plan is not None and remaining > 45 and blocked_molecule_turns < 2:
            blocked_molecule_turns += 1
            return "WAIT", loose_plan

        blocked_molecule_turns = 0

        # If this plan is blocked, go back to Diagnosis/Samples to replace it.
        return goto(DIAG if carried else SAMP, me), loose_plan

    if me.target == DIAG:
        blocked_molecule_turns = 0

        if undiagnosed:
            # Diagnose higher-rank samples first, but among same rank prefer
            # newer/high-id only as tie-break.
            s = max(undiagnosed, key=lambda x: (x.rank, x.id))
            return "CONNECT {}".format(s.id), loose_plan

        if makeable:
            return goto(LAB, me), loose_plan

        if loose_plan is not None:
            return goto(MOLMOD, me), loose_plan

        # Try useful cloud samples before discarding ours.
        if len(carried) < 3:
            s = choose_cloud_sample(
                cloud,
                carried,
                me.expertise,
                me.storage,
                avail,
                projects,
                remaining,
                me.target,
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
            me.target,
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
            return "CONNECT {}".format(choose_rank(me.expertise, projects, remaining)), loose_plan

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
    opp = robots[1]
    carried = [s for s in samples if s.carried_by == 0]

    if plan is None:
        plan_s = "None"
    else:
        plan_s = "ids={} req={} val={:.1f} est={}".format(
            plan["ids"],
            plan["req"],
            plan["value"],
            plan["est"],
        )

    print(
        "T{} {} eta={} score={} opp={} store={} exp={} oppExp={} avail={} carried={} plan={} dropped={} -> {}".format(
            turn,
            me.target,
            me.eta,
            me.score,
            opp.score,
            me.storage,
            me.expertise,
            opp.expertise,
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