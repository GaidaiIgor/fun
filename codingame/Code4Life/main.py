import sys
from itertools import combinations, permutations

MOLS = "ABCDE"
SAMP = "SAMPLES"
DIAG = "DIAGNOSIS"
MOLMOD = "MOLECULES"
LAB = "LABORATORY"

turn = 0
dropped_ids = set()

blocked_key = None
blocked_count = 0


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
            bonus += 65.0

        bonus += 3.2 * max(0, before_sum - after_sum)
        bonus += 2.5 * max(0, max(before) - max(after))

    return bonus


def gain_bonus(sample, exp, projects):
    gi = mi(sample.gain)
    if gi < 0:
        return 0.0

    mx_need = max_project_need(projects)
    b = 0.0

    if exp[gi] == min(exp):
        b += 7.0

    if exp[gi] == 0:
        b += 8.0
    elif exp[gi] == 1:
        b += 5.0
    elif exp[gi] == 2:
        b += 2.0

    for pr in projects:
        deficit = project_deficit(exp, pr)
        gap = sum(deficit)

        if exp[gi] < pr[gi]:
            b += 6.0
            b += 1.5 * (pr[gi] - exp[gi])
            b += 10.0 / (1.0 + gap)

    if exp[gi] >= mx_need[gi] and exp[gi] >= 4:
        b -= 8.0

    return b


def to_collect_for_req(req, storage):
    return [max(0, req[i] - storage[i]) for i in range(5)]


def shortage_for_req(req, storage, avail):
    need = to_collect_for_req(req, storage)
    return [max(0, need[i] - avail[i]) for i in range(5)]


def zero_block_types(req, storage, avail):
    need = to_collect_for_req(req, storage)
    return [i for i in range(5) if need[i] > 0 and avail[i] <= 0]


def plan_block_key(plan, storage, avail):
    if plan is None:
        return None

    z = zero_block_types(plan["req"], storage, avail)
    if not z:
        return None

    return tuple(plan["ids"]), tuple(z), tuple(to_collect_for_req(plan["req"], storage))


def time_and_supply_penalty(req, storage, avail, remaining, require_available):
    to_collect = to_collect_for_req(req, storage)
    shortage = shortage_for_req(req, storage, avail)

    p = 0.0
    p += 0.20 * sum(req)
    p += 0.35 * sum(to_collect)

    if require_available and sum(shortage) > 0:
        return 10**9

    if remaining > 90:
        p += 3.0 * sum(shortage)
    elif remaining > 55:
        p += 9.0 * sum(shortage)
    else:
        p += 35.0 * sum(shortage)

    z = zero_block_types(req, storage, avail)

    # This is the v4 fix. A plan blocked by a zero-availability molecule is not
    # merely "slightly worse"; it can burn 30 turns while the bot waits like an idiot.
    if z:
        p += 45.0 + 20.0 * len(z)

        if remaining < 120:
            p += 40.0

        if storage_total(storage) >= 5:
            p += 60.0

    for i in range(5):
        if to_collect[i] > 0 and avail[i] <= 1:
            p += 3.0 * to_collect[i]

        if req[i] >= 7 and storage[i] + avail[i] < req[i]:
            p += 30.0 if remaining < 90 else 12.0

    return p


def plan_value(order, req, final_exp, storage, avail, exp, projects, remaining, target, require_available):
    health = sum(s.health for s in order)
    to_collect = to_collect_for_req(req, storage)
    est = estimate_finish_turns(order, to_collect, target)

    value = health
    value += sum(gain_bonus(s, exp, projects) for s in order)
    value += project_progress_bonus(exp, final_exp, projects)
    value += 2.0 * (len(order) - 1)
    value -= time_and_supply_penalty(req, storage, avail, remaining, require_available)

    if est + 1 > remaining:
        value -= 120.0 + 6.0 * (est + 1 - remaining)

    for s in order:
        if s.health <= 1:
            if remaining < 130:
                value -= 5.0
            if remaining < 70:
                value -= 18.0

    return value


def select_best_plan(samples, exp, storage, avail, projects, remaining, target, require_available):
    diagnosed = [s for s in samples if s.diagnosed and s.id not in dropped_ids]
    best = None

    for r in range(1, len(diagnosed) + 1):
        for subset in combinations(diagnosed, r):
            for order in permutations(subset):
                req, final_exp = order_requirements(order, exp)

                if not feasible_capacity(req, storage):
                    continue

                to_collect = to_collect_for_req(req, storage)

                if require_available and any(to_collect[i] > avail[i] for i in range(5)):
                    continue

                est = estimate_finish_turns(order, to_collect, target)

                if est + 1 > remaining:
                    continue

                val = plan_value(
                    order,
                    req,
                    final_exp,
                    storage,
                    avail,
                    exp,
                    projects,
                    remaining,
                    target,
                    require_available,
                )

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


def makeable_samples(carried, storage, exp):
    return [s for s in carried if s.diagnosed and can_make(s, storage, exp)]


def choose_rank(exp, projects, remaining):
    et = sum(exp)

    if et < 3:
        return 1

    if et < 10:
        return 2

    # Rank 3 only when expertise is broad enough. The bot has already proven it
    # can find a glorious rank-3 tar pit and drown in it.
    if remaining > 60 and min(exp) >= 1 and et >= 11:
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

    current = select_best_plan(carried, exp, storage, avail, projects, remaining, target, True)
    current_val = current["value"] if current else 0.0

    best = None

    for s in cloud:
        if s.id in dropped_ids or not s.diagnosed:
            continue

        combined = select_best_plan(
            carried + [s],
            exp,
            storage,
            avail,
            projects,
            remaining,
            target,
            True,
        )

        if combined is None:
            continue

        single_val = best_single_value(s, exp, storage, avail, projects, remaining, target)
        improvement = combined["value"] - current_val
        score = single_val + 0.8 * improvement

        if best is None or score > best[0]:
            best = (score, s)

    if best is not None and best[0] > 6:
        return best[1]

    return None


def choose_drop_sample(carried, exp, storage, avail, projects, remaining, target, loose_plan):
    diagnosed = [s for s in carried if s.diagnosed]

    if not diagnosed:
        return None

    strict = select_best_plan(
        diagnosed,
        exp,
        storage,
        avail,
        projects,
        remaining,
        target,
        True,
    )

    if strict is not None:
        return None

    candidates = []

    plan_ids = set(loose_plan["ids"]) if loose_plan else set()
    plan_req = loose_plan["req"] if loose_plan else None
    z = zero_block_types(plan_req, storage, avail) if plan_req else []

    for s in diagnosed:
        ec = effective_cost(s, exp)
        val = best_single_value(s, exp, storage, avail, projects, remaining, target)
        badness = -val

        if sum(ec) > 10:
            badness += 100

        if s.id in plan_ids and z:
            for i in z:
                if ec[i] > 0:
                    badness += 90 + 20 * ec[i]

        if any(ec[i] > 0 and avail[i] == 0 for i in range(5)):
            badness += 70

        if s.health <= 1:
            badness += 10

        # Prefer dropping samples that need molecules we do not currently hold.
        missing_from_storage = sum(max(0, ec[i] - storage[i]) for i in range(5))
        badness += 3 * missing_from_storage

        candidates.append((badness, s))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def choose_molecule(plan, storage, avail, exp):
    if plan is None or storage_total(storage) >= 10:
        return None

    req = plan["req"]
    need = to_collect_for_req(req, storage)

    if sum(need) == 0:
        return None

    first_need = [0] * 5

    if plan["order"]:
        first_req, _ = order_requirements([plan["order"][0]], exp)
        first_need = to_collect_for_req(first_req, storage)

    best_i = None
    best_score = -10**9

    for i in range(5):
        if need[i] <= 0 or avail[i] <= 0:
            continue

        score = 0.0
        score += 45.0 * min(1, first_need[i])
        score += 10.0 * need[i]
        score += 4.0 * max(0, 4 - avail[i])

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


def update_block_memory(plan, storage, avail):
    global blocked_key, blocked_count

    k = plan_block_key(plan, storage, avail)

    if k is None:
        blocked_key = None
        blocked_count = 0
        return 0

    if k == blocked_key:
        blocked_count += 1
    else:
        blocked_key = k
        blocked_count = 1

    return blocked_count


def clear_block_memory():
    global blocked_key, blocked_count
    blocked_key = None
    blocked_count = 0


def decide(projects, robots, avail, samples):
    global turn, dropped_ids

    me = robots[0]
    remaining = 200 - turn

    carried = [s for s in samples if s.carried_by == 0]
    diagnosed = [s for s in carried if s.diagnosed]
    undiagnosed = [s for s in carried if not s.diagnosed]
    cloud = [s for s in samples if s.carried_by == -1 and s.diagnosed]

    if me.eta > 0:
        return "WAIT", None, "moving"

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
        clear_block_memory()

        if makeable:
            s = choose_completion_sample(makeable, me.expertise, me.storage, projects)
            return "CONNECT {}".format(s.id), loose_plan, "lab_make"

        # Never camp Lab without a producible sample. Humanity tried waiting in
        # rooms for solutions; it invented meetings.
        if strict_plan is not None:
            return goto(MOLMOD, me), strict_plan, "lab_to_mol_strict"

        if undiagnosed:
            return goto(DIAG, me), loose_plan, "lab_to_diag"

        if remaining > 15:
            return goto(SAMP, me), loose_plan, "lab_to_samples"

        return "WAIT", loose_plan, "lab_end_wait"

    if me.target == MOLMOD:
        if makeable:
            # Bank points unless a strict plan can be continued immediately.
            mol = choose_molecule(strict_plan, me.storage, avail, me.expertise)

            if mol is not None and strict_plan is not None and strict_plan["est"] + 2 < remaining:
                clear_block_memory()
                return "CONNECT {}".format(mol), strict_plan, "mol_extend_strict"

            clear_block_memory()
            return goto(LAB, me), strict_plan or loose_plan, "mol_bank_makeable"

        if not diagnosed:
            clear_block_memory()
            return goto(DIAG if undiagnosed else SAMP, me), loose_plan, "mol_no_diag"

        if strict_plan is not None:
            mol = choose_molecule(strict_plan, me.storage, avail, me.expertise)

            if mol is not None:
                clear_block_memory()
                return "CONNECT {}".format(mol), strict_plan, "mol_take_strict"

            clear_block_memory()
            return goto(DIAG, me), strict_plan, "mol_strict_no_mol"

        # No strict plan means the old code would start believing in molecule
        # fairies. We allow at most one wait on a blocked loose plan.
        if loose_plan is not None:
            count = update_block_memory(loose_plan, me.storage, avail)

            mol = choose_molecule(loose_plan, me.storage, avail, me.expertise)

            if mol is not None and count <= 1:
                return "CONNECT {}".format(mol), loose_plan, "mol_partial_once"

            if count <= 1 and remaining > 70:
                return "WAIT", loose_plan, "mol_wait_once_blocked"

            return goto(DIAG, me), loose_plan, "mol_escape_blocked"

        clear_block_memory()
        return goto(DIAG if carried else SAMP, me), loose_plan, "mol_no_plan"

    if me.target == DIAG:
        if undiagnosed:
            clear_block_memory()
            s = max(undiagnosed, key=lambda x: (x.rank, x.id))
            return "CONNECT {}".format(s.id), loose_plan, "diag_diagnose"

        if makeable:
            clear_block_memory()
            return goto(LAB, me), loose_plan, "diag_to_lab_makeable"

        if strict_plan is not None:
            clear_block_memory()
            return goto(MOLMOD, me), strict_plan, "diag_to_mol_strict"

        # We have a loose plan only because molecules are unavailable or timing is bad.
        # Drop the blocker instead of starting the Diagnosis <-> Molecules shuttle service.
        if loose_plan is not None:
            bad = choose_drop_sample(
                carried,
                me.expertise,
                me.storage,
                avail,
                projects,
                remaining,
                me.target,
                loose_plan,
            )

            if bad is not None:
                dropped_ids.add(bad.id)
                clear_block_memory()
                return "CONNECT {}".format(bad.id), loose_plan, "diag_drop_blocked"

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
                clear_block_memory()
                return "CONNECT {}".format(s.id), loose_plan, "diag_take_cloud"

        bad = choose_drop_sample(
            carried,
            me.expertise,
            me.storage,
            avail,
            projects,
            remaining,
            me.target,
            loose_plan,
        )

        if bad is not None:
            dropped_ids.add(bad.id)
            clear_block_memory()
            return "CONNECT {}".format(bad.id), loose_plan, "diag_drop_no_plan"

        if len(carried) < 3 and remaining > 18:
            clear_block_memory()
            return goto(SAMP, me), loose_plan, "diag_to_samples"

        clear_block_memory()
        return "WAIT", loose_plan, "diag_wait"

    if me.target == SAMP:
        clear_block_memory()

        if len(carried) < 3 and remaining > 18:
            return "CONNECT {}".format(choose_rank(me.expertise, projects, remaining)), loose_plan, "sample_take"

        return goto(DIAG, me), loose_plan, "sample_to_diag"

    clear_block_memory()

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
        plan_s = "None"
    else:
        plan_s = "ids={} req={} need={} val={:.1f} est={}".format(
            plan["ids"],
            plan["req"],
            to_collect_for_req(plan["req"], me.storage),
            plan["value"],
            plan["est"],
        )

    print(
        "T{} {} eta={} score={} opp={} store={} exp={} oppExp={} avail={} carried={} plan={} dropped={} block={}x{} reason={} -> {}".format(
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
            blocked_key,
            blocked_count,
            reason,
            action,
        ),
        file=sys.stderr,
    )

    print(action)
    turn += 1