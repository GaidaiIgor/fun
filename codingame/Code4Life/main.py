import sys
from itertools import combinations, permutations

MOLS = "ABCDE"
SAMP = "SAMPLES"
DIAG = "DIAGNOSIS"
MOLMOD = "MOLECULES"
LAB = "LABORATORY"

DIST = {
    "START_POS": {SAMP: 2, DIAG: 2, MOLMOD: 2, LAB: 2},
    SAMP: {SAMP: 0, DIAG: 3, MOLMOD: 3, LAB: 3},
    DIAG: {SAMP: 3, DIAG: 0, MOLMOD: 3, LAB: 4},
    MOLMOD: {SAMP: 3, DIAG: 3, MOLMOD: 0, LAB: 3},
    LAB: {SAMP: 3, DIAG: 4, MOLMOD: 3, LAB: 0},
}

turn = 0
ban_until = {}
last_block_key = None
block_count = 0


def mi(g):
    return MOLS.find(g)


def mol_name(i):
    return MOLS[i]


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


def is_banned(sample_id):
    return turn < ban_until.get(sample_id, -1)


def ban_sample(sample_id, duration=22):
    ban_until[sample_id] = turn + duration


def can_make(sample, storage, exp):
    return sample.diagnosed and all(storage[i] + exp[i] >= sample.cost[i] for i in range(5))


def effective_cost(sample, exp):
    return [max(0, sample.cost[i] - exp[i]) for i in range(5)]


def apply_gain(exp, sample):
    e = exp[:]
    gi = mi(sample.gain)
    if gi >= 0:
        e[gi] += 1
    return e


def order_requirements(order, exp):
    e = exp[:]
    req = [0] * 5

    for s in order:
        ec = effective_cost(s, e)
        for i in range(5):
            req[i] += ec[i]
        e = apply_gain(e, s)

    return req, e


def need_from_req(req, storage):
    return [max(0, req[i] - storage[i]) for i in range(5)]


def shortage_from_req(req, storage, avail):
    need = need_from_req(req, storage)
    return [max(0, need[i] - avail[i]) for i in range(5)]


def zero_blocks(req, storage, avail):
    need = need_from_req(req, storage)
    return [i for i in range(5) if need[i] > 0 and avail[i] <= 0]


def feasible_capacity(req, storage):
    need = need_from_req(req, storage)
    if storage_total(storage) + sum(need) > 10:
        return False
    if any(req[i] > 10 for i in range(5)):
        return False
    return True


def project_deficit(exp, project):
    return [max(0, project[i] - exp[i]) for i in range(5)]


def project_gap(exp, project):
    return sum(project_deficit(exp, project))


def active_projects(projects, my_exp, opp_exp):
    out = []
    for p in projects:
        my_done = all(my_exp[i] >= p[i] for i in range(5))
        opp_done = all(opp_exp[i] >= p[i] for i in range(5))
        if not my_done and not opp_done:
            out.append(p)
    return out


def max_project_need(projects):
    mx = [0] * 5
    for p in projects:
        for i in range(5):
            mx[i] = max(mx[i], p[i])
    return mx


def urgent_gain(sample, exp, projects, opp_exp):
    gi = mi(sample.gain)
    if gi < 0:
        return False

    for p in active_projects(projects, exp, opp_exp):
        my_gap = project_gap(exp, p)
        opp_gap = project_gap(opp_exp, p)
        if exp[gi] < p[gi] and (my_gap <= 3 or opp_gap <= 2):
            return True

    return False


def trash_sample(sample, exp, projects, opp_exp, remaining):
    if not sample.diagnosed:
        return False

    if sample.rank == 1 and sample.health <= 1 and sum(exp) >= 2:
        if urgent_gain(sample, exp, projects, opp_exp):
            return False
        return True

    if remaining < 80 and sample.health <= 1:
        return True

    return False


def project_bonus(exp, final_exp, projects, opp_exp):
    active = active_projects(projects, exp, opp_exp)
    bonus = 0.0

    for p in active:
        before = project_deficit(exp, p)
        after = project_deficit(final_exp, p)
        opp_gap = project_gap(opp_exp, p)

        before_sum = sum(before)
        after_sum = sum(after)
        progress = before_sum - after_sum

        if progress > 0:
            bonus += 5.5 * progress
            bonus += 3.0 * max(0, max(before) - max(after))

        if before_sum > 0 and after_sum == 0:
            bonus += 80.0
            if opp_gap <= 2:
                bonus += 40.0

        if opp_gap <= 2 and progress > 0:
            bonus += 16.0 * progress

    return bonus


def expertise_gain_bonus(sample, exp, projects, opp_exp):
    gi = mi(sample.gain)
    if gi < 0:
        return 0.0

    active = active_projects(projects, exp, opp_exp)
    mx_need = max_project_need(active)
    b = 0.0

    if exp[gi] == min(exp):
        b += 10.0

    if exp[gi] == 0:
        b += 12.0
    elif exp[gi] == 1:
        b += 7.0
    elif exp[gi] == 2:
        b += 3.0

    for p in active:
        my_gap = project_gap(exp, p)
        opp_gap = project_gap(opp_exp, p)

        if exp[gi] < p[gi]:
            b += 7.0
            b += 2.0 * (p[gi] - exp[gi])
            b += 12.0 / (1 + my_gap)

            if opp_gap <= 2:
                b += 14.0

    if active and exp[gi] >= mx_need[gi] and exp[gi] >= 4:
        b -= 10.0

    return b


def dist_from(target, dest):
    return DIST.get(target, DIST["START_POS"]).get(dest, 3)


def estimate_finish_turns(order, req, storage, current_target):
    need = need_from_req(req, storage)
    mol_actions = sum(need)
    lab_actions = len(order)

    if not order:
        return 999

    if mol_actions == 0:
        if current_target == LAB:
            return lab_actions
        return dist_from(current_target, LAB) + lab_actions

    if current_target == MOLMOD:
        return mol_actions + dist_from(MOLMOD, LAB) + lab_actions

    return (
        dist_from(current_target, MOLMOD)
        + mol_actions
        + dist_from(MOLMOD, LAB)
        + lab_actions
    )


def opponent_plan_need(opp, samples, avail, projects):
    opp_carried = [s for s in samples if s.carried_by == 1 and s.diagnosed]
    if not opp_carried:
        return [0] * 5

    plan = select_best_plan(
        opp_carried,
        opp.expertise,
        opp.storage,
        avail,
        projects,
        200 - turn,
        opp.target,
        False,
        [0, 0, 0, 0, 0],
        opp_need=[0, 0, 0, 0, 0],
        ignore_ban=True,
        for_opponent=True,
    )

    if plan is None:
        return [0] * 5

    return need_from_req(plan["req"], opp.storage)


def contention_penalty(req, storage, avail, opp_need, remaining):
    need = need_from_req(req, storage)
    p = 0.0

    for i in range(5):
        if need[i] <= 0:
            continue

        overlap = min(need[i], opp_need[i])

        if overlap > 0:
            if avail[i] <= need[i] + opp_need[i]:
                p += 5.5 * overlap
            if avail[i] <= 2:
                p += 5.5 * overlap

        if avail[i] == 0:
            p += 32.0
            if remaining < 120:
                p += 35.0
        elif avail[i] == 1:
            p += 4.0 * need[i]

    return p


def plan_raw_value(order, req, final_exp, storage, avail, exp, projects,
                   remaining, current_target, opp_exp, opp_need, require_available):
    health = sum(s.health for s in order)
    need = need_from_req(req, storage)
    shortage = shortage_from_req(req, storage, avail)
    est = estimate_finish_turns(order, req, storage, current_target)

    if require_available and any(need[i] > avail[i] for i in range(5)):
        return -10**9

    value = health
    value += sum(expertise_gain_bonus(s, exp, projects, opp_exp) for s in order)
    value += project_bonus(exp, final_exp, projects, opp_exp)
    value += 1.2 * (len(order) - 1)

    value -= 0.18 * sum(req)
    value -= 0.35 * sum(need)
    value -= contention_penalty(req, storage, avail, opp_need, remaining)

    if remaining > 100:
        value -= 2.0 * sum(shortage)
    elif remaining > 55:
        value -= 8.0 * sum(shortage)
    else:
        value -= 35.0 * sum(shortage)

    z = zero_blocks(req, storage, avail)
    if z:
        value -= 28.0 + 18.0 * len(z)
        if storage_total(storage) >= 5:
            value -= 35.0

    if est + 1 > remaining:
        value -= 250.0 + 12.0 * (est + 1 - remaining)

    for s in order:
        if s.rank == 1 and s.health <= 1:
            if sum(exp) >= 2:
                value -= 25.0
            if remaining < 145:
                value -= 12.0
            if remaining < 90:
                value -= 35.0

        if s.health <= 1 and s.rank >= 2 and remaining < 100:
            value -= 15.0

    return value


def plan_metric(raw_value, est, order):
    if raw_value <= -10**8:
        return raw_value

    health = sum(s.health for s in order)
    # Better blend: do not let a tiny sample look brilliant just because it is short.
    return raw_value / max(1, est) + 0.055 * raw_value + 0.015 * health


def select_best_plan(samples, exp, storage, avail, projects, remaining, current_target,
                     require_available, opp_exp, opp_need=None, ignore_ban=False,
                     for_opponent=False, ignore_trash=True):
    diagnosed = []
    for s in samples:
        if not s.diagnosed:
            continue
        if not ignore_ban and is_banned(s.id):
            continue
        if ignore_trash and not for_opponent and trash_sample(s, exp, projects, opp_exp, remaining):
            continue
        diagnosed.append(s)

    if not diagnosed:
        return None

    if opp_need is None:
        opp_need = [0] * 5

    best = None

    for r in range(1, len(diagnosed) + 1):
        for subset in combinations(diagnosed, r):
            for order in permutations(subset):
                req, final_exp = order_requirements(order, exp)

                if not feasible_capacity(req, storage):
                    continue

                need = need_from_req(req, storage)

                if require_available and any(need[i] > avail[i] for i in range(5)):
                    continue

                est = estimate_finish_turns(order, req, storage, current_target)
                if est + 1 > remaining:
                    continue

                raw = plan_raw_value(
                    order, req, final_exp, storage, avail, exp,
                    projects, remaining, current_target, opp_exp, opp_need,
                    require_available,
                )

                metric = raw / max(1, est) if for_opponent else plan_metric(raw, est, order)

                if best is None or metric > best["metric"]:
                    best = {
                        "order": list(order),
                        "ids": [s.id for s in order],
                        "req": req,
                        "need": need,
                        "final_exp": final_exp,
                        "raw": raw,
                        "metric": metric,
                        "est": est,
                    }

    return best


def makeable_samples(carried, storage, exp):
    return [s for s in carried if s.diagnosed and can_make(s, storage, exp)]


def choose_completion_sample(makeable, plan, exp, projects, opp_exp):
    if plan is not None:
        for s in plan["order"]:
            if s in makeable:
                return s

    best = None

    for s in makeable:
        e2 = apply_gain(exp, s)
        score = s.health
        score += expertise_gain_bonus(s, exp, projects, opp_exp)
        score += project_bonus(exp, e2, projects, opp_exp)

        if best is None or score > best[0]:
            best = (score, s)

    return best[1]


def choose_rank(me, opp, projects, carried_count, remaining):
    exp = me.expertise
    et = sum(exp)

    if remaining < 32:
        return 1 if et < 5 else 2

    if et < 3:
        return 1

    # Controlled rank-3 aggression. The previous version stayed too civilized
    # and got mugged by higher-value work.
    if remaining > 95 and et >= 8:
        if carried_count == 0:
            return 3
        return 2

    if remaining > 70 and et >= 11:
        if carried_count == 0:
            return 3
        return 2

    if me.score + 35 < opp.score and remaining > 55 and et >= 8:
        if carried_count == 0:
            return 3
        return 2

    return 2


def choose_molecule(plan, storage, avail, exp, opp_need):
    if plan is None or storage_total(storage) >= 10:
        return None

    req = plan["req"]
    need = need_from_req(req, storage)

    if sum(need) == 0:
        return None

    first_need = [0] * 5
    if plan["order"]:
        first_req, _ = order_requirements([plan["order"][0]], exp)
        first_need = need_from_req(first_req, storage)

    best = None

    for i in range(5):
        if need[i] <= 0 or avail[i] <= 0:
            continue

        score = 52.0 * min(1, first_need[i])
        score += 10.0 * need[i]
        score += 5.0 * max(0, 4 - avail[i])

        if opp_need[i] > 0:
            score += 8.0
            if avail[i] <= need[i] + opp_need[i]:
                score += 12.0

        if best is None or score > best[0]:
            best = (score, i)

    return None if best is None else mol_name(best[1])


def recoverable_loose_plan(plan, storage, avail, remaining):
    if plan is None:
        return False

    need = need_from_req(plan["req"], storage)
    z = zero_blocks(plan["req"], storage, avail)
    shortage = shortage_from_req(plan["req"], storage, avail)

    if sum(need) == 0:
        return True

    useful_now = any(need[i] > 0 and avail[i] > 0 for i in range(5))
    if useful_now:
        return True

    if not z and sum(shortage) <= 1:
        return True

    if remaining > 110 and sum(shortage) <= 2:
        return True

    return False


def choose_drop_sample(carried, exp, storage, avail, projects, remaining,
                       current_target, loose_plan, opp_exp, opp_need):
    diagnosed = [s for s in carried if s.diagnosed]
    if not diagnosed:
        return None

    # Drop trash samples even if a "plan" exists, because otherwise they become
    # ceremonial anchors.
    trash = [s for s in diagnosed if trash_sample(s, exp, projects, opp_exp, remaining)]
    if trash:
        trash.sort(key=lambda s: (s.health, -sum(effective_cost(s, exp))))
        return trash[0]

    if recoverable_loose_plan(loose_plan, storage, avail, remaining):
        return None

    current_ids = set(loose_plan["ids"]) if loose_plan else set()
    blocked = zero_blocks(loose_plan["req"], storage, avail) if loose_plan else []
    best = None

    for s in diagnosed:
        ec = effective_cost(s, exp)

        without = [x for x in diagnosed if x.id != s.id]
        plan_without = select_best_plan(
            without, exp, storage, avail, projects, remaining, current_target,
            False, opp_exp, opp_need
        )

        badness = 0.0

        if plan_without is not None:
            badness += plan_without["metric"] * 6.0

        badness -= 1.5 * max(0, s.health)
        badness -= expertise_gain_bonus(s, exp, projects, opp_exp)

        if sum(ec) > 10:
            badness += 100

        for i in blocked:
            if ec[i] > 0:
                badness += 65 + 20 * ec[i]

        if any(ec[i] > 0 and avail[i] == 0 for i in range(5)):
            badness += 35

        if s.id not in current_ids:
            badness += 12

        if best is None or badness > best[0]:
            best = (badness, s)

    if best is None or best[0] < 35:
        return None

    return best[1]


def choose_cloud_sample(cloud, carried, exp, storage, avail, projects,
                        remaining, current_target, opp_exp, opp_need):
    if len(carried) >= 3:
        return None

    current = select_best_plan(
        carried, exp, storage, avail, projects, remaining, current_target,
        True, opp_exp, opp_need
    )
    current_metric = current["metric"] if current else 0.0

    best = None

    for s in cloud:
        if not s.diagnosed or is_banned(s.id):
            continue

        if trash_sample(s, exp, projects, opp_exp, remaining):
            continue

        combined = select_best_plan(
            carried + [s], exp, storage, avail, projects, remaining,
            current_target, True, opp_exp, opp_need
        )

        if combined is None:
            continue

        improvement = combined["metric"] - current_metric
        score = combined["metric"] + 0.7 * improvement

        if best is None or score > best[0]:
            best = (score, s)

    if best is not None and best[0] > 2.5:
        return best[1]

    return None


def block_key(plan, storage, avail):
    if plan is None:
        return None

    z = zero_blocks(plan["req"], storage, avail)
    if not z:
        return None

    return tuple(plan["ids"]), tuple(z), tuple(need_from_req(plan["req"], storage))


def update_block(plan, storage, avail):
    global last_block_key, block_count

    k = block_key(plan, storage, avail)
    if k is None:
        last_block_key = None
        block_count = 0
        return 0

    if k == last_block_key:
        block_count += 1
    else:
        last_block_key = k
        block_count = 1

    return block_count


def clear_block():
    global last_block_key, block_count
    last_block_key = None
    block_count = 0


def goto(module, me):
    if me.target == module:
        return "WAIT"
    return "GOTO " + module


def summarize(samples):
    out = []
    for s in samples:
        if s.diagnosed:
            out.append("{}:r{}:{}:{}:{}".format(
                s.id, s.rank, s.health, s.gain, "".join(map(str, s.cost))
            ))
        else:
            out.append("{}:r{}:?".format(s.id, s.rank))
    return "[" + ",".join(out) + "]"


def should_skip_low_value_plan(plan, carried, exp, projects, opp_exp, remaining):
    if plan is None:
        return False

    if remaining < 60:
        return False

    planned_ids = set(plan["ids"])
    planned = [s for s in carried if s.id in planned_ids]

    if not planned:
        return False

    if all(trash_sample(s, exp, projects, opp_exp, remaining) for s in planned):
        return True

    if plan["raw"] < 22 and len(carried) < 3:
        return True

    return False


def decide(projects, robots, avail, samples):
    global turn

    me = robots[0]
    opp = robots[1]
    remaining = 200 - turn

    carried = [s for s in samples if s.carried_by == 0]
    diagnosed = [s for s in carried if s.diagnosed]
    undiagnosed = [s for s in carried if not s.diagnosed]
    cloud = [s for s in samples if s.carried_by == -1 and s.diagnosed]

    if me.eta > 0:
        return "WAIT", None, "moving"

    opp_need = opponent_plan_need(opp, samples, avail, projects)

    strict_plan = select_best_plan(
        diagnosed, me.expertise, me.storage, avail, projects, remaining,
        me.target, True, opp.expertise, opp_need
    )

    loose_plan = strict_plan or select_best_plan(
        diagnosed, me.expertise, me.storage, avail, projects, remaining,
        me.target, False, opp.expertise, opp_need
    )

    makeable = makeable_samples(diagnosed, me.storage, me.expertise)

    if me.target == LAB:
        clear_block()

        if makeable:
            s = choose_completion_sample(makeable, loose_plan, me.expertise, projects, opp.expertise)
            return "CONNECT {}".format(s.id), loose_plan, "lab_make"

        if remaining <= 6:
            return "WAIT", loose_plan, "lab_late_wait"

        if should_skip_low_value_plan(strict_plan, carried, me.expertise, projects, opp.expertise, remaining):
            if len(carried) < 3:
                return goto(SAMP, me), strict_plan, "lab_skip_trash_to_samples"
            return goto(DIAG, me), strict_plan, "lab_skip_trash_to_diag"

        if strict_plan is not None:
            return goto(MOLMOD, me), strict_plan, "lab_to_mol"

        if undiagnosed:
            return goto(DIAG, me), loose_plan, "lab_to_diag"

        if len(carried) < 3 and remaining > 22:
            return goto(SAMP, me), loose_plan, "lab_to_samples"

        return "WAIT", loose_plan, "lab_wait"

    if me.target == MOLMOD:
        if makeable:
            mol = choose_molecule(strict_plan, me.storage, avail, me.expertise, opp_need)

            if (
                mol is not None
                and strict_plan is not None
                and strict_plan["est"] + 2 < remaining
                and strict_plan["raw"] >= 35
                and remaining > 35
            ):
                clear_block()
                return "CONNECT {}".format(mol), strict_plan, "mol_extend_batch"

            clear_block()
            return goto(LAB, me), strict_plan or loose_plan, "mol_bank"

        if not diagnosed:
            clear_block()
            return goto(DIAG if undiagnosed else SAMP, me), loose_plan, "mol_no_diag"

        if should_skip_low_value_plan(strict_plan, carried, me.expertise, projects, opp.expertise, remaining):
            clear_block()
            return goto(DIAG if len(carried) >= 3 else SAMP, me), strict_plan, "mol_skip_low"

        if strict_plan is not None:
            mol = choose_molecule(strict_plan, me.storage, avail, me.expertise, opp_need)
            if mol is not None:
                clear_block()
                return "CONNECT {}".format(mol), strict_plan, "mol_take_strict"
            clear_block()
            return goto(DIAG, me), strict_plan, "mol_strict_stuck"

        if loose_plan is not None:
            count = update_block(loose_plan, me.storage, avail)
            mol = choose_molecule(loose_plan, me.storage, avail, me.expertise, opp_need)

            if mol is not None:
                return "CONNECT {}".format(mol), loose_plan, "mol_take_loose"

            if zero_blocks(loose_plan["req"], me.storage, avail) and count <= 1 and remaining > 90:
                return "WAIT", loose_plan, "mol_wait_once"

            return goto(DIAG, me), loose_plan, "mol_replan"

        clear_block()
        return goto(DIAG if carried else SAMP, me), loose_plan, "mol_no_plan"

    if me.target == DIAG:
        clear_block()

        if undiagnosed:
            s = max(undiagnosed, key=lambda x: (x.rank, x.id))
            return "CONNECT {}".format(s.id), loose_plan, "diag_diagnose"

        if makeable:
            return goto(LAB, me), loose_plan, "diag_to_lab"

        if strict_plan is not None and not should_skip_low_value_plan(
            strict_plan, carried, me.expertise, projects, opp.expertise, remaining
        ):
            return goto(MOLMOD, me), strict_plan, "diag_to_mol_strict"

        bad = choose_drop_sample(
            carried, me.expertise, me.storage, avail, projects,
            remaining, me.target, loose_plan, opp.expertise, opp_need
        )

        if bad is not None:
            ban_sample(bad.id)
            return "CONNECT {}".format(bad.id), loose_plan, "diag_drop"

        if loose_plan is not None:
            mol = choose_molecule(loose_plan, me.storage, avail, me.expertise, opp_need)
            if mol is not None and recoverable_loose_plan(loose_plan, me.storage, avail, remaining):
                return goto(MOLMOD, me), loose_plan, "diag_to_mol_loose"

        if len(carried) < 3:
            s = choose_cloud_sample(
                cloud, carried, me.expertise, me.storage, avail, projects,
                remaining, me.target, opp.expertise, opp_need
            )
            if s is not None:
                return "CONNECT {}".format(s.id), loose_plan, "diag_take_cloud"

        if len(carried) < 3 and remaining > 22:
            return goto(SAMP, me), loose_plan, "diag_to_samples"

        return "WAIT", loose_plan, "diag_wait"

    if me.target == SAMP:
        clear_block()

        if len(carried) < 3 and remaining > 22:
            r = choose_rank(me, opp, projects, len(carried), remaining)
            return "CONNECT {}".format(r), loose_plan, "sample_take"

        return goto(DIAG, me), loose_plan, "sample_to_diag"

    clear_block()

    if makeable:
        return goto(LAB, me), loose_plan, "fallback_lab"
    if strict_plan is not None:
        return goto(MOLMOD, me), strict_plan, "fallback_mol"
    if undiagnosed:
        return goto(DIAG, me), loose_plan, "fallback_diag"
    return goto(SAMP, me), loose_plan, "fallback_samples"


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

    action, plan, reason = decide(projects, robots, available, samples)

    me = robots[0]
    opp = robots[1]
    carried = [s for s in samples if s.carried_by == 0]

    if plan is None:
        ps = "None"
    else:
        ps = "ids={} req={} need={} raw={:.1f} met={:.2f} est={}".format(
            plan["ids"], plan["req"], need_from_req(plan["req"], me.storage),
            plan["raw"], plan["metric"], plan["est"]
        )

    print(
        "T{} {} eta={} score={} opp={} store={} exp={} oppExp={} avail={} carried={} plan={} ban={} block={}x{} reason={} -> {}".format(
            turn,
            me.target,
            me.eta,
            me.score,
            opp.score,
            me.storage,
            me.expertise,
            opp.expertise,
            available,
            summarize(carried),
            ps,
            {k: v for k, v in ban_until.items() if v > turn},
            last_block_key,
            block_count,
            reason,
            action,
        ),
        file=sys.stderr,
    )

    print(action)
    turn += 1