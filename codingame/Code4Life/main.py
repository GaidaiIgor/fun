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


def dist(a, b):
    return DIST.get(a, DIST["START_POS"]).get(b, 3)


def can_make(s, storage, exp):
    return s.diagnosed and all(storage[i] + exp[i] >= s.cost[i] for i in range(5))


def eff_cost(s, exp):
    return [max(0, s.cost[i] - exp[i]) for i in range(5)]


def apply_gain(exp, s):
    e = exp[:]
    gi = mi(s.gain)
    if gi >= 0:
        e[gi] += 1
    return e


def order_req(order, exp):
    e = exp[:]
    req = [0] * 5

    for s in order:
        c = eff_cost(s, e)
        for i in range(5):
            req[i] += c[i]
        e = apply_gain(e, s)

    return req, e


def need(req, storage):
    return [max(0, req[i] - storage[i]) for i in range(5)]


def shortage(req, storage, avail):
    n = need(req, storage)
    return [max(0, n[i] - avail[i]) for i in range(5)]


def zero_block(req, storage, avail):
    n = need(req, storage)
    return [i for i in range(5) if n[i] > 0 and avail[i] <= 0]


def feasible_capacity(req, storage):
    n = need(req, storage)
    return storage_total(storage) + sum(n) <= 10 and all(r <= 10 for r in req)


def project_deficit(exp, pr):
    return [max(0, pr[i] - exp[i]) for i in range(5)]


def project_gap(exp, pr):
    return sum(project_deficit(exp, pr))


def active_projects(projects, my_exp, opp_exp):
    out = []

    for pr in projects:
        my_done = all(my_exp[i] >= pr[i] for i in range(5))
        opp_done = all(opp_exp[i] >= pr[i] for i in range(5))

        if not my_done and not opp_done:
            out.append(pr)

    return out


def project_bonus(exp, final_exp, projects, opp_exp):
    b = 0.0

    for pr in active_projects(projects, exp, opp_exp):
        before = project_deficit(exp, pr)
        after = project_deficit(final_exp, pr)

        before_sum = sum(before)
        after_sum = sum(after)
        progress = before_sum - after_sum
        opp_gap = project_gap(opp_exp, pr)

        if progress > 0:
            b += 4.5 * progress
            b += 2.5 * max(0, max(before) - max(after))

        if before_sum > 0 and after_sum == 0:
            b += 65.0
            if opp_gap <= 2:
                b += 35.0

        if opp_gap <= 2 and progress > 0:
            b += 12.0 * progress

    return b


def gain_bonus(s, exp, projects, opp_exp):
    gi = mi(s.gain)
    if gi < 0:
        return 0.0

    b = 0.0

    if exp[gi] == min(exp):
        b += 8.0

    if exp[gi] == 0:
        b += 10.0
    elif exp[gi] == 1:
        b += 6.0
    elif exp[gi] == 2:
        b += 2.0

    for pr in active_projects(projects, exp, opp_exp):
        if exp[gi] < pr[gi]:
            my_gap = project_gap(exp, pr)
            opp_gap = project_gap(opp_exp, pr)

            b += 5.0
            b += 1.5 * (pr[gi] - exp[gi])
            b += 8.0 / (1 + my_gap)

            if opp_gap <= 2:
                b += 10.0

    return b


def est_finish(order, req, storage, target):
    n = need(req, storage)
    mol_turns = sum(n)
    lab_turns = len(order)

    if not order:
        return 999

    if mol_turns == 0:
        if target == LAB:
            return lab_turns
        return dist(target, LAB) + lab_turns

    if target == MOLMOD:
        return mol_turns + dist(MOLMOD, LAB) + lab_turns

    return dist(target, MOLMOD) + mol_turns + dist(MOLMOD, LAB) + lab_turns


def opponent_need(opp, samples, avail, projects):
    carried = [s for s in samples if s.carried_by == 1 and s.diagnosed]

    if not carried:
        return [0] * 5

    p = best_plan(
        carried,
        opp.expertise,
        opp.storage,
        avail,
        projects,
        200 - turn,
        opp.target,
        False,
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        opponent=True,
    )

    if p is None:
        return [0] * 5

    return need(p["req"], opp.storage)


def raw_value(order, req, final_exp, storage, avail, exp, projects,
              remaining, target, opp_exp, opp_need, require_available):
    n = need(req, storage)
    sh = shortage(req, storage, avail)
    z = zero_block(req, storage, avail)
    est = est_finish(order, req, storage, target)

    if require_available and any(n[i] > avail[i] for i in range(5)):
        return -10**9

    value = sum(s.health for s in order)
    value += sum(gain_bonus(s, exp, projects, opp_exp) for s in order)
    value += project_bonus(exp, final_exp, projects, opp_exp)

    # Batch bonus, but deliberately small. Old versions got drunk on batch theory.
    value += 0.8 * (len(order) - 1)

    value -= 0.14 * sum(req)
    value -= 0.25 * sum(n)

    for i in range(5):
        if n[i] <= 0:
            continue

        overlap = min(n[i], opp_need[i])
        if overlap > 0 and avail[i] <= n[i] + opp_need[i]:
            value -= 6.0 * overlap

        if avail[i] == 0:
            value -= 35.0
            if remaining < 80:
                value -= 45.0
        elif avail[i] == 1:
            value -= 4.0 * n[i]

    if remaining > 80:
        value -= 2.0 * sum(sh)
    elif remaining > 45:
        value -= 8.0 * sum(sh)
    else:
        value -= 40.0 * sum(sh)

    if z and storage_total(storage) >= 4:
        value -= 35.0

    if est + 1 > remaining:
        value -= 300.0 + 15.0 * (est + 1 - remaining)

    for s in order:
        if s.rank == 1 and s.health <= 1 and remaining < 100:
            value -= 18.0
        if s.health <= 1 and remaining < 65:
            value -= 30.0

    return value


def metric(raw, est, order, remaining):
    if raw <= -10**8:
        return raw

    health = sum(s.health for s in order)

    # Late game: raw points dominate. Earlier: tempo matters more.
    if remaining < 55:
        return raw / max(1, est) + 0.075 * raw + 0.035 * health

    return raw / max(1, est) + 0.045 * raw + 0.015 * health


def best_plan(samples, exp, storage, avail, projects, remaining, target,
              require_available, opp_exp, opp_need, opponent=False):
    diagnosed = [s for s in samples if s.diagnosed]

    if not diagnosed:
        return None

    best = None

    for r in range(1, len(diagnosed) + 1):
        for sub in combinations(diagnosed, r):
            for order in permutations(sub):
                req, final_exp = order_req(order, exp)

                if not feasible_capacity(req, storage):
                    continue

                n = need(req, storage)

                if require_available and any(n[i] > avail[i] for i in range(5)):
                    continue

                est = est_finish(order, req, storage, target)

                if est + 1 > remaining:
                    continue

                raw = raw_value(
                    order,
                    req,
                    final_exp,
                    storage,
                    avail,
                    exp,
                    projects,
                    remaining,
                    target,
                    opp_exp,
                    opp_need,
                    require_available,
                )

                m = raw / max(1, est) if opponent else metric(raw, est, order, remaining)

                if best is None or m > best["metric"]:
                    best = {
                        "order": list(order),
                        "ids": [s.id for s in order],
                        "req": req,
                        "need": n,
                        "final_exp": final_exp,
                        "raw": raw,
                        "metric": m,
                        "est": est,
                    }

    return best


def makeable_samples(samples, storage, exp):
    return [s for s in samples if s.diagnosed and can_make(s, storage, exp)]


def choose_completion(makeable, plan, exp, projects, opp_exp):
    if plan is not None:
        for s in plan["order"]:
            if s in makeable:
                return s

    best = None

    for s in makeable:
        e2 = apply_gain(exp, s)
        score = s.health + gain_bonus(s, exp, projects, opp_exp)
        score += project_bonus(exp, e2, projects, opp_exp)

        if best is None or score > best[0]:
            best = (score, s)

    return best[1]


def desired_batch_size(remaining, exp_sum):
    if remaining < 38:
        return 1
    if remaining < 60:
        return 2
    return 3


def choose_rank(me, opp, carried_count, remaining):
    et = sum(me.expertise)

    # Only opening rank 1. Stop creating late paperwork goblins.
    if turn < 25 and et < 2:
        return 1

    # Endgame variance: one rank 3 can beat two mediocre rank 2s.
    if remaining > 32 and et >= 10 and carried_count == 0:
        return 3

    if remaining > 55 and et >= 8 and carried_count == 0:
        return 3

    if me.score + 35 < opp.score and remaining > 45 and et >= 8 and carried_count == 0:
        return 3

    return 2


def choose_molecule(plan, storage, avail, exp, opp_need):
    if plan is None or storage_total(storage) >= 10:
        return None

    n = need(plan["req"], storage)

    if sum(n) == 0:
        return None

    first_need = [0] * 5
    if plan["order"]:
        first_req, _ = order_req([plan["order"][0]], exp)
        first_need = need(first_req, storage)

    best = None

    for i in range(5):
        if n[i] <= 0 or avail[i] <= 0:
            continue

        score = 55.0 * min(1, first_need[i])
        score += 11.0 * n[i]
        score += 5.0 * max(0, 4 - avail[i])

        if opp_need[i] > 0:
            score += 8.0
            if avail[i] <= n[i] + opp_need[i]:
                score += 12.0

        if best is None or score > best[0]:
            best = (score, i)

    if best is None:
        return None

    return MOLS[best[1]]


def block_key(plan, storage, avail):
    if plan is None:
        return None

    z = zero_block(plan["req"], storage, avail)
    if not z:
        return None

    return tuple(plan["ids"]), tuple(z), tuple(need(plan["req"], storage))


def update_block(plan, storage, avail):
    global blocked_key, blocked_count

    k = block_key(plan, storage, avail)

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


def clear_block():
    global blocked_key, blocked_count
    blocked_key = None
    blocked_count = 0


def goto(module, me):
    if me.target == module:
        return "WAIT"
    return "GOTO " + module


def should_drop(s, exp, avail, remaining):
    c = eff_cost(s, exp)

    if sum(c) > 10:
        return True

    if remaining < 45 and s.health <= 1:
        return True

    if remaining < 55 and any(c[i] > 0 and avail[i] == 0 for i in range(5)):
        return True

    return False


def choose_drop(carried, exp, storage, avail, projects, remaining, target,
                loose_plan, opp_exp, opp_need):
    diagnosed = [s for s in carried if s.diagnosed]

    if not diagnosed:
        return None

    candidates = [s for s in diagnosed if should_drop(s, exp, avail, remaining)]

    if not candidates:
        z = zero_block(loose_plan["req"], storage, avail) if loose_plan else []
        if not z:
            return None

        # Drop only if we are genuinely blocked, not merely inconvenienced.
        if any(need(loose_plan["req"], storage)[i] > 0 and avail[i] > 0 for i in range(5)):
            return None

        candidates = diagnosed

    best = None

    for s in candidates:
        c = eff_cost(s, exp)
        bad = sum(c) - 0.7 * s.health - gain_bonus(s, exp, projects, opp_exp)

        if any(c[i] > 0 and avail[i] == 0 for i in range(5)):
            bad += 25

        if best is None or bad > best[0]:
            best = (bad, s)

    return None if best is None else best[1]


def choose_cloud(cloud, carried, exp, storage, avail, projects, remaining,
                 target, opp_exp, opp_need):
    if len(carried) >= 3:
        return None

    best = None
    current = best_plan(carried, exp, storage, avail, projects, remaining,
                        target, True, opp_exp, opp_need)
    cur_m = current["metric"] if current else 0.0

    for s in cloud:
        if not s.diagnosed:
            continue

        p = best_plan(carried + [s], exp, storage, avail, projects, remaining,
                      target, True, opp_exp, opp_need)

        if p is None:
            continue

        score = p["metric"] + 0.7 * (p["metric"] - cur_m)

        if best is None or score > best[0]:
            best = (score, s)

    if best and best[0] > 2.5:
        return best[1]

    return None


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

    opp_need = opponent_need(opp, samples, avail, projects)

    strict = best_plan(diagnosed, me.expertise, me.storage, avail, projects,
                       remaining, me.target, True, opp.expertise, opp_need)
    loose = strict or best_plan(diagnosed, me.expertise, me.storage, avail, projects,
                                remaining, me.target, False, opp.expertise, opp_need)

    makeable = makeable_samples(diagnosed, me.storage, me.expertise)

    if me.target == LAB:
        clear_block()

        if makeable:
            s = choose_completion(makeable, loose, me.expertise, projects, opp.expertise)
            return "CONNECT {}".format(s.id), loose, "lab_make"

        if remaining <= 6:
            return "WAIT", loose, "lab_late_wait"

        if strict is not None:
            if sum(strict["need"]) == 0:
                return "WAIT", strict, "lab_plan_noop"
            return goto(MOLMOD, me), strict, "lab_to_mol"

        if undiagnosed:
            return goto(DIAG, me), loose, "lab_to_diag"

        if len(carried) < desired_batch_size(remaining, sum(me.expertise)) and remaining > 24:
            return goto(SAMP, me), loose, "lab_to_samples"

        return "WAIT", loose, "lab_wait"

    if me.target == MOLMOD:
        if makeable:
            mol = choose_molecule(strict, me.storage, avail, me.expertise, opp_need)

            if (
                mol is not None
                and strict is not None
                and strict["raw"] >= 45
                and strict["est"] + 2 < remaining
                and remaining > 45
            ):
                clear_block()
                return "CONNECT {}".format(mol), strict, "mol_extend"

            clear_block()
            return goto(LAB, me), strict or loose, "mol_bank"

        if not diagnosed:
            clear_block()
            return goto(DIAG if undiagnosed else SAMP, me), loose, "mol_no_diag"

        if strict is not None:
            mol = choose_molecule(strict, me.storage, avail, me.expertise, opp_need)
            if mol is not None:
                clear_block()
                return "CONNECT {}".format(mol), strict, "mol_take_strict"
            clear_block()
            return goto(DIAG, me), strict, "mol_strict_stuck"

        if loose is not None:
            count = update_block(loose, me.storage, avail)
            mol = choose_molecule(loose, me.storage, avail, me.expertise, opp_need)

            if mol is not None:
                return "CONNECT {}".format(mol), loose, "mol_take_loose"

            if zero_block(loose["req"], me.storage, avail) and count <= 1 and remaining > 85:
                return "WAIT", loose, "mol_wait_once"

            return goto(DIAG, me), loose, "mol_replan"

        clear_block()
        return goto(DIAG if carried else SAMP, me), loose, "mol_no_plan"

    if me.target == DIAG:
        clear_block()

        # Critical v4 fix: if time is tight and we already have guaranteed points,
        # bank them. Do not keep diagnosing paperwork into the apocalypse.
        if makeable and (remaining < 65 or not undiagnosed):
            return goto(LAB, me), loose, "diag_bank_makeable"

        if strict is not None and remaining < 45 and strict["raw"] >= 25:
            if sum(strict["need"]) == 0:
                return goto(LAB, me), strict, "diag_late_lab"
            return goto(MOLMOD, me), strict, "diag_late_mol"

        if undiagnosed:
            s = max(undiagnosed, key=lambda x: (x.rank, x.id))
            return "CONNECT {}".format(s.id), loose, "diag_diagnose"

        if makeable:
            return goto(LAB, me), loose, "diag_to_lab"

        bad = choose_drop(carried, me.expertise, me.storage, avail, projects,
                          remaining, me.target, loose, opp.expertise, opp_need)
        if bad is not None:
            return "CONNECT {}".format(bad.id), loose, "diag_drop"

        if strict is not None:
            if sum(strict["need"]) == 0:
                return goto(LAB, me), strict, "diag_to_lab_zero"
            return goto(MOLMOD, me), strict, "diag_to_mol"

        if loose is not None:
            mol = choose_molecule(loose, me.storage, avail, me.expertise, opp_need)
            if mol is not None:
                return goto(MOLMOD, me), loose, "diag_to_mol_loose"

        if len(carried) < desired_batch_size(remaining, sum(me.expertise)):
            s = choose_cloud(cloud, carried, me.expertise, me.storage, avail,
                             projects, remaining, me.target, opp.expertise, opp_need)
            if s is not None:
                return "CONNECT {}".format(s.id), loose, "diag_take_cloud"

        if len(carried) == 0 and remaining > 24:
            return goto(SAMP, me), loose, "diag_to_samples_empty"

        return "WAIT", loose, "diag_wait"

    if me.target == SAMP:
        clear_block()

        # Do not keep sampling while holding diagnosed work.
        if diagnosed:
            return goto(DIAG, me), loose, "sample_leave_with_diag"

        target_count = desired_batch_size(remaining, sum(me.expertise))

        if len(carried) < target_count and remaining > 24:
            r = choose_rank(me, opp, len(carried), remaining)
            return "CONNECT {}".format(r), loose, "sample_take"

        return goto(DIAG, me), loose, "sample_to_diag"

    clear_block()

    if makeable:
        return goto(LAB, me), loose, "fallback_lab"
    if strict is not None:
        return goto(MOLMOD if sum(strict["need"]) else LAB, me), strict, "fallback_plan"
    if undiagnosed:
        return goto(DIAG, me), loose, "fallback_diag"
    return goto(SAMP, me), loose, "fallback_samples"


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
            plan["ids"],
            plan["req"],
            need(plan["req"], me.storage),
            plan["raw"],
            plan["metric"],
            plan["est"],
        )

    print(
        "T{} {} eta={} score={} opp={} store={} exp={} oppExp={} avail={} carried={} plan={} block={}x{} reason={} -> {}".format(
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
            blocked_key,
            blocked_count,
            reason,
            action,
        ),
        file=sys.stderr,
    )

    print(action)
    turn += 1