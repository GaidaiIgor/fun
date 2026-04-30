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
block_key_seen = None
block_count = 0
recently_dropped = {}


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


def completed(exp, pr):
    return all(exp[i] >= pr[i] for i in range(5))


def active_projects(projects, my_exp, opp_exp):
    return [
        pr for pr in projects
        if not completed(my_exp, pr) and not completed(opp_exp, pr)
    ]


def urgent_project(projects, my_exp, opp_exp):
    for pr in active_projects(projects, my_exp, opp_exp):
        if project_gap(opp_exp, pr) <= 2 or project_gap(my_exp, pr) <= 3:
            return True
    return False


def bottleneck_score(i, my_exp, opp_exp, projects, remaining):
    if i < 0:
        return 0.0

    s = 0.0
    mn = min(my_exp)

    if my_exp[i] == mn:
        s += 10.0
    if my_exp[i] == 0:
        s += 18.0
    elif my_exp[i] == 1:
        s += 9.0
    elif my_exp[i] == 2:
        s += 3.0

    if opp_exp[i] >= my_exp[i] + 2:
        s += 8.0

    for pr in active_projects(projects, my_exp, opp_exp):
        deficit = max(0, pr[i] - my_exp[i])
        if deficit <= 0:
            continue

        my_gap = project_gap(my_exp, pr)
        opp_gap = project_gap(opp_exp, pr)

        s += 13.0 * deficit
        s += 8.0 / (1.0 + my_gap)

        if my_gap <= 4:
            s += 14.0
        if opp_gap <= 3:
            s += 20.0
        if opp_gap <= 2:
            s += 22.0
        if opp_gap <= 1:
            s += 18.0

    if remaining < 55:
        s *= 0.55

    return s


def project_bonus(exp, final_exp, projects, opp_exp, remaining):
    b = 0.0

    for pr in active_projects(projects, exp, opp_exp):
        before = project_deficit(exp, pr)
        after = project_deficit(final_exp, pr)

        before_sum = sum(before)
        after_sum = sum(after)
        progress = before_sum - after_sum
        opp_gap = project_gap(opp_exp, pr)

        if progress > 0:
            b += 5.0 * progress
            b += 3.0 * max(0, max(before) - max(after))

            if opp_gap <= 3:
                b += 12.0 * progress
            if opp_gap <= 2:
                b += 12.0 * progress

        if before_sum > 0 and after_sum == 0:
            b += 95.0
            if opp_gap <= 3:
                b += 40.0
            if opp_gap <= 1:
                b += 25.0

    return b


def gain_bonus(s, exp, projects, opp_exp, remaining):
    return bottleneck_score(mi(s.gain), exp, opp_exp, projects, remaining)


def est_finish(order, req, storage, target):
    n = need(req, storage)
    mol_turns = sum(n)
    lab_turns = len(order)

    if not order:
        return 999

    if mol_turns == 0:
        return lab_turns if target == LAB else dist(target, LAB) + lab_turns

    if target == MOLMOD:
        return mol_turns + dist(MOLMOD, LAB) + lab_turns

    return dist(target, MOLMOD) + mol_turns + dist(MOLMOD, LAB) + lab_turns


def molecule_pressure_for(opp, samples, avail, projects):
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

    health = sum(s.health for s in order)

    value = float(health)
    value += sum(gain_bonus(s, exp, projects, opp_exp, remaining) for s in order)
    value += project_bonus(exp, final_exp, projects, opp_exp, remaining)

    value += 0.7 * (len(order) - 1)

    value -= 0.12 * sum(req)
    value -= 0.20 * sum(n)

    for i in range(5):
        if n[i] <= 0:
            continue

        overlap = min(n[i], opp_need[i])

        if overlap > 0 and avail[i] <= n[i] + opp_need[i]:
            value -= 7.0 * overlap

        if avail[i] == 0:
            value -= 45.0
            if remaining < 85:
                value -= 50.0
        elif avail[i] == 1:
            value -= 5.0 * n[i]

    if remaining > 85:
        value -= 2.2 * sum(sh)
    elif remaining > 45:
        value -= 9.0 * sum(sh)
    else:
        value -= 42.0 * sum(sh)

    if z and storage_total(storage) >= 4:
        value -= 45.0

    if est + 1 > remaining:
        value -= 350.0 + 18.0 * (est + 1 - remaining)

    for s in order:
        if s.rank == 1 and s.health <= 1 and remaining < 105:
            value -= 18.0
        if s.health <= 1 and remaining < 60:
            value -= 35.0

    return value


def metric(raw, est, order, remaining):
    if raw <= -10**8:
        return raw

    health = sum(s.health for s in order)

    if remaining < 55:
        return raw / max(1, est) + 0.085 * raw + 0.045 * health

    return raw / max(1, est) + 0.050 * raw + 0.018 * health


def best_plan(samples, exp, storage, avail, projects, remaining, target,
              require_available, opp_exp, opp_need, opponent=False):
    diagnosed = [s for s in samples if s.diagnosed]

    if not diagnosed:
        return None

    best = None

    for r in range(1, min(3, len(diagnosed)) + 1):
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


def choose_completion(makeable, plan, exp, projects, opp_exp, remaining):
    if plan is not None:
        for s in plan["order"]:
            if s in makeable:
                return s

    best = None

    for s in makeable:
        e2 = apply_gain(exp, s)
        score = s.health
        score += gain_bonus(s, exp, projects, opp_exp, remaining)
        score += project_bonus(exp, e2, projects, opp_exp, remaining)

        if best is None or score > best[0]:
            best = (score, s)

    return best[1]


def desired_batch_size(remaining, exp_sum):
    if remaining < 38:
        return 1
    if remaining < 65:
        return 2
    return 3


def choose_rank(me, opp, carried_count, remaining, projects):
    et = sum(me.expertise)

    if turn < 25 and et < 2:
        return 1

    # Final-attempt rule:
    # if science-project race is live, rank 2 is better than rank 3 because it
    # gives faster chances to find and complete the missing expertise letter.
    if urgent_project(projects, me.expertise, opp.expertise):
        return 2

    if remaining > 65 and et >= 11 and carried_count == 0:
        return 3

    if me.score + 45 < opp.score and remaining > 55 and et >= 9 and carried_count == 0:
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

        score = 58.0 * min(1, first_need[i])
        score += 11.0 * n[i]
        score += 5.0 * max(0, 4 - avail[i])

        if opp_need[i] > 0:
            score += 9.0
            if avail[i] <= n[i] + opp_need[i]:
                score += 13.0

        if best is None or score > best[0]:
            best = (score, i)

    return None if best is None else MOLS[best[1]]


def sample_promise(s, exp, storage, avail, projects, remaining, target, opp_exp, opp_need):
    p = best_plan(
        [s],
        exp,
        storage,
        avail,
        projects,
        remaining,
        target,
        False,
        opp_exp,
        opp_need,
    )

    return -10**9 if p is None else p["metric"]


def choose_cloud(cloud, carried, exp, storage, avail, projects, remaining,
                 target, opp_exp, opp_need):
    best = None

    for s in cloud:
        if not s.diagnosed:
            continue

        if turn < recently_dropped.get(s.id, -1):
            continue

        gi = mi(s.gain)
        focus = bottleneck_score(gi, exp, opp_exp, projects, remaining)

        if len(carried) < 3:
            p = best_plan(
                carried + [s],
                exp,
                storage,
                avail,
                projects,
                remaining,
                target,
                False,
                opp_exp,
                opp_need,
            )
        else:
            p = best_plan(
                [s],
                exp,
                storage,
                avail,
                projects,
                remaining,
                target,
                False,
                opp_exp,
                opp_need,
            )

        if p is None:
            continue

        score = p["metric"] + 0.25 * focus

        if best is None or score > best[0]:
            best = (score, s, p)

    return best


def choose_drop_for_cloud(carried, cloud_pick, exp, storage, avail, projects,
                          remaining, target, opp_exp, opp_need):
    if cloud_pick is None or len(carried) < 3:
        return None

    _, cloud_s, _ = cloud_pick

    cloud_score = sample_promise(
        cloud_s,
        exp,
        storage,
        avail,
        projects,
        remaining,
        target,
        opp_exp,
        opp_need,
    )
    cloud_gain = bottleneck_score(mi(cloud_s.gain), exp, opp_exp, projects, remaining)

    best_drop = None

    for s in carried:
        if not s.diagnosed:
            continue

        s_score = sample_promise(
            s,
            exp,
            storage,
            avail,
            projects,
            remaining,
            target,
            opp_exp,
            opp_need,
        )
        s_gain = bottleneck_score(mi(s.gain), exp, opp_exp, projects, remaining)
        keep_value = s_score + 0.20 * s_gain + 0.15 * s.health

        if best_drop is None or keep_value < best_drop[0]:
            best_drop = (keep_value, s)

    if best_drop is None:
        return None

    if cloud_score + 0.25 * cloud_gain > best_drop[0] + 12.0:
        return best_drop[1]

    return None


def should_drop_blocked(s, exp, avail, remaining):
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

    forced = [s for s in diagnosed if should_drop_blocked(s, exp, avail, remaining)]

    if forced:
        forced.sort(key=lambda s: (s.health, -sum(eff_cost(s, exp))))
        return forced[0]

    if loose_plan is None:
        return None

    z = zero_block(loose_plan["req"], storage, avail)

    if not z:
        return None

    n = need(loose_plan["req"], storage)

    if any(n[i] > 0 and avail[i] > 0 for i in range(5)):
        return None

    best = None

    for s in diagnosed:
        c = eff_cost(s, exp)

        bad = sum(c)
        bad -= 0.8 * s.health
        bad -= 0.2 * gain_bonus(s, exp, projects, opp_exp, remaining)

        if any(c[i] > 0 and avail[i] == 0 for i in range(5)):
            bad += 25

        if best is None or bad > best[0]:
            best = (bad, s)

    return None if best is None else best[1]


def update_block(plan, storage, avail):
    global block_key_seen, block_count

    if plan is None:
        block_key_seen = None
        block_count = 0
        return 0

    z = zero_block(plan["req"], storage, avail)

    if not z:
        block_key_seen = None
        block_count = 0
        return 0

    key = (tuple(plan["ids"]), tuple(z), tuple(need(plan["req"], storage)))

    if key == block_key_seen:
        block_count += 1
    else:
        block_key_seen = key
        block_count = 1

    return block_count


def clear_block():
    global block_key_seen, block_count
    block_key_seen = None
    block_count = 0


def goto(module, me):
    return "WAIT" if me.target == module else "GOTO " + module


def summarize(samples):
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
    global recently_dropped

    me = robots[0]
    opp = robots[1]
    remaining = 200 - turn

    recently_dropped = {k: v for k, v in recently_dropped.items() if v > turn}

    carried = [s for s in samples if s.carried_by == 0]
    diagnosed = [s for s in carried if s.diagnosed]
    undiagnosed = [s for s in carried if not s.diagnosed]
    cloud = [s for s in samples if s.carried_by == -1 and s.diagnosed]

    if me.eta > 0:
        return "WAIT", None, "moving"

    opp_need = molecule_pressure_for(opp, samples, avail, projects)

    strict = best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        remaining,
        me.target,
        True,
        opp.expertise,
        opp_need,
    )

    loose = strict or best_plan(
        diagnosed,
        me.expertise,
        me.storage,
        avail,
        projects,
        remaining,
        me.target,
        False,
        opp.expertise,
        opp_need,
    )

    makeable = makeable_samples(diagnosed, me.storage, me.expertise)
    target_batch = desired_batch_size(remaining, sum(me.expertise))

    if me.target == LAB:
        clear_block()

        if makeable:
            s = choose_completion(
                makeable,
                loose,
                me.expertise,
                projects,
                opp.expertise,
                remaining,
            )
            return "CONNECT {}".format(s.id), loose, "lab_make"

        if remaining <= 6:
            return "WAIT", loose, "lab_end"

        if strict is not None:
            return goto(MOLMOD if sum(strict["need"]) else LAB, me), strict, "lab_to_plan"

        if undiagnosed:
            return goto(DIAG, me), loose, "lab_to_diag"

        if len(carried) < target_batch and remaining > 24:
            return goto(SAMP, me), loose, "lab_to_samples"

        return "WAIT", loose, "lab_wait"

    if me.target == MOLMOD:
        if makeable:
            mol = choose_molecule(strict, me.storage, avail, me.expertise, opp_need)

            if (
                mol is not None
                and strict is not None
                and strict["raw"] >= 50
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
                return "CONNECT {}".format(mol), strict, "mol_take"

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

        if makeable and (remaining < 70 or not undiagnosed):
            return goto(LAB, me), loose, "diag_bank_makeable"

        if strict is not None and remaining < 45 and strict["raw"] >= 25:
            return goto(MOLMOD if sum(strict["need"]) else LAB, me), strict, "diag_late_plan"

        if undiagnosed:
            s = max(undiagnosed, key=lambda x: (x.rank, x.id))
            return "CONNECT {}".format(s.id), loose, "diag_diagnose"

        if makeable:
            return goto(LAB, me), loose, "diag_to_lab"

        cloud_pick = choose_cloud(
            cloud,
            carried,
            me.expertise,
            me.storage,
            avail,
            projects,
            remaining,
            me.target,
            opp.expertise,
            opp_need,
        )

        if cloud_pick is not None:
            _, cloud_s, _ = cloud_pick

            if len(carried) < 3:
                return "CONNECT {}".format(cloud_s.id), loose, "diag_take_cloud_focus"

            drop = choose_drop_for_cloud(
                carried,
                cloud_pick,
                me.expertise,
                me.storage,
                avail,
                projects,
                remaining,
                me.target,
                opp.expertise,
                opp_need,
            )

            if drop is not None:
                recently_dropped[drop.id] = turn + 12
                return "CONNECT {}".format(drop.id), loose, "diag_drop_for_cloud"

        bad = choose_drop(
            carried,
            me.expertise,
            me.storage,
            avail,
            projects,
            remaining,
            me.target,
            loose,
            opp.expertise,
            opp_need,
        )

        if bad is not None:
            recently_dropped[bad.id] = turn + 12
            return "CONNECT {}".format(bad.id), loose, "diag_drop_blocked"

        if strict is not None:
            return goto(MOLMOD if sum(strict["need"]) else LAB, me), strict, "diag_to_plan"

        if loose is not None:
            mol = choose_molecule(loose, me.storage, avail, me.expertise, opp_need)

            if mol is not None:
                return goto(MOLMOD, me), loose, "diag_to_loose"

        if len(carried) == 0 and remaining > 24:
            return goto(SAMP, me), loose, "diag_empty_samples"

        return "WAIT", loose, "diag_wait"

    if me.target == SAMP:
        clear_block()

        if diagnosed:
            return goto(DIAG, me), loose, "sample_leave_with_diag"

        if len(carried) < target_batch and remaining > 24:
            r = choose_rank(me, opp, len(carried), remaining, projects)
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
        "T{} {} eta={} score={} opp={} store={} exp={} oppExp={} avail={} carried={} plan={} reason={} -> {}".format(
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
            reason,
            action,
        ),
        file=sys.stderr,
    )

    print(action)
    turn += 1