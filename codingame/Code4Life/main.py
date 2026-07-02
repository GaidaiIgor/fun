import sys
import traceback
from itertools import permutations

TYPES = "ABCDE"
TIDX = {c: i for i, c in enumerate(TYPES)}
CARRY = 10  # molecule carry limit


def log(*args):
    print(*args, file=sys.stderr)


class Obj(object):
    pass


def parse_player(line):
    t = line.split()
    p = Obj()
    p.target = t[0]
    p.eta = int(t[1])
    p.score = int(t[2])
    p.storage = [int(x) for x in t[3:8]]
    p.expertise = [int(x) for x in t[8:13]]
    return p


def parse_sample(t):
    s = Obj()
    s.id = int(t[0])
    s.carried_by = int(t[1])
    s.rank = int(t[2])
    s.gain = t[3]
    s.gain_idx = TIDX.get(t[3], -1)
    s.health = int(t[4])
    s.cost = [int(x) for x in t[5:10]]
    s.diagnosed = s.health >= 0
    return s


def eff_cost(s, exp):
    return [max(0, s.cost[i] - exp[i]) for i in range(5)]


def feasible(s, exp):
    # can this sample ever be completed in one trip with current expertise?
    e = eff_cost(s, exp)
    return max(e) <= 5 and sum(e) <= CARRY


def buyable(s, exp, storage, avail):
    # can this sample be completed right now: missing molecules are in the
    # pool AND fit under the carry limit (held molecules can never be dropped)
    e = eff_cost(s, exp)
    take = [max(0, e[i] - storage[i]) for i in range(5)]
    return all(take[i] <= avail[i] for i in range(5)) and sum(storage) + sum(take) <= CARRY


def gift_risk(s, opp_exp, tl):
    # points the opponent likely gains if we dump this sample to the cloud
    if tl < 10:
        return 0.0
    e = [max(0, s.cost[i] - opp_exp[i]) for i in range(5)]
    if max(e) <= 5 and sum(e) <= 9:
        return float(s.health)
    return 0.0


def project_active(p, my_exp, opp_exp):
    if all(my_exp[i] >= p[i] for i in range(5)):
        return False
    if all(opp_exp[i] >= p[i] for i in range(5)):
        return False
    return True


def gain_bonus(gidx, my_exp, opp_exp, projects, tl):
    # value of gaining one expertise of type gidx
    if gidx < 0:
        return 0.0
    # generic value: cost savings pay back over the remaining game
    b = 1.0 + min(5.0, tl / 40.0)
    # balanced expertise unlocks more affordable samples and projects
    if my_exp[gidx] <= min(my_exp) + 1:
        b += 1.5
    for p in projects:
        if not project_active(p, my_exp, opp_exp):
            continue
        rem = [max(0, p[i] - my_exp[i]) for i in range(5)]
        tot = sum(rem)
        if tot > 0 and rem[gidx] > 0:
            # discount projects the opponent is closer to winning
            opp_rem = sum(max(0, p[i] - opp_exp[i]) for i in range(5))
            # nearly-complete projects stay worth chasing even when behind:
            # the opponent may never draw the finishing gain
            floor = 0.4 if tot <= 2 else 0.15
            w = 1.0 if tot <= opp_rem else max(floor, float(opp_rem) / (tot + opp_rem))
            b += w * 50.0 / tot
            if tot == 1 and opp_rem <= 2:
                b += 15.0  # sprint: land the decisive expertise first
    return b


def sample_value(s, my_exp, opp_exp, projects, tl):
    return s.health + gain_bonus(s.gain_idx, my_exp, opp_exp, projects, tl)


def best_plan(diag, my_exp, opp_exp, storage, available, projects, tl=999, overhead=3):
    """Pick the best ordered subset of carried diagnosed samples to produce in
    one molecules->lab trip. Expertise gained from producing earlier samples
    reduces the molecule needs of later ones, so order matters.
    A sequence must fit in the remaining turns: molecules to take + overhead
    (travel to lab) + one production turn per sample.
    Returns {"perm": tuple, "take": [5], "value": float} or None."""
    best = None
    stot = sum(storage)
    for r in range(1, len(diag) + 1):
        for perm in permutations(diag, r):
            exp_sim = list(my_exp)
            need_tot = [0] * 5
            val = 0.0
            for s in perm:
                for i in range(5):
                    need_tot[i] += max(0, s.cost[i] - exp_sim[i])
                val += s.health + gain_bonus(s.gain_idx, exp_sim, opp_exp, projects, tl)
                if s.gain_idx >= 0:
                    exp_sim[s.gain_idx] += 1
            take = [max(0, need_tot[i] - storage[i]) for i in range(5)]
            if any(take[i] > available[i] for i in range(5)):
                continue
            tk = sum(take)
            if stot + tk > CARRY:
                continue
            if tk + overhead + r > tl:
                continue
            key = (val - 0.35 * tk, r)
            if best is None or key > best["key"]:
                best = {"key": key, "perm": perm, "take": take, "value": val}
    return best


def choose_rank(E, k):
    # E = total expertise, k = samples already in hand
    k = min(k, 2)
    # early expertise rush: rank 1 costs exactly ~3 molecules and buys +1
    # expertise, which compounds into cheaper medicines and science projects
    if E <= 4:
        return (1, 1, 2)[k]
    if E <= 6:
        return 2
    if E <= 9:
        return (3, 2, 2)[k]
    if E <= 11:
        return (3, 3, 2)[k]
    return 3


def decide(me, opp, avail, samples, projects, turn, state):
    """Returns command. state persists across turns:
    state["wait"]: consecutive blocked-waits at MOLECULES
    state["blocked"]: ids of carried samples marked unbuyable (pool starved)
    state["age"]: turns each diagnosed sample has spent in hand"""
    wait_count = state["wait"]
    mine = [s for s in samples if s.carried_by == 0]
    mine_diag = [s for s in mine if s.diagnosed]
    mine_undiag = [s for s in mine if not s.diagnosed]
    # does this opponent mine the cloud? (samples that moved cloud -> their hand)
    cloud_ids = set(s.id for s in samples if s.carried_by == -1)
    opp_ids = set(s.id for s in samples if s.carried_by == 1)
    state["opp_mined"] += len(state["cloud_prev"] & opp_ids)
    state["cloud_prev"] = cloud_ids
    # withhold gifts only from proven repeat cloud-miners; a default-cautious
    # window clogged the hand against the majority who never mine
    opp_mines = state["opp_mined"] >= 2
    # how long each diagnosed sample has been stuck in hand
    ages = state["age"]
    held_ids = set(s.id for s in mine_diag)
    for i in list(ages.keys()):
        if i not in held_ids:
            del ages[i]
    for i in held_ids:
        ages[i] = ages.get(i, 0) + 1
    exp = me.expertise
    E = sum(exp)
    tl = 201 - turn  # turns remaining, including this one

    hand = " | ".join(
        "%d:r%d h%d g%s %s" % (s.id, s.rank, s.health, s.gain,
                               ("e%s=%d" % (eff_cost(s, exp), sum(eff_cost(s, exp)))) if s.diagnosed else "undiag")
        for s in mine) or "-"
    log("T%d tl=%d me@%s eta%d sc%d st%s=%d ex%s E%d" %
        (turn, tl, me.target, me.eta, me.score, me.storage, sum(me.storage), exp, E))
    log("  pool%s | opp@%s eta%d sc%d ex%s ost%d ohand%d" %
        (avail, opp.target, opp.eta, opp.score, opp.expertise, sum(opp.storage),
         len([s for s in samples if s.carried_by == 1])))
    log("  hand: %s" % hand)
    cloud = [s for s in samples if s.carried_by == -1 and s.diagnosed]
    if cloud:
        log("  cloud: %s" % " | ".join(
            "%d:h%d e%d o%d" % (s.id, s.health, sum(eff_cost(s, exp)),
                                sum(max(0, s.cost[i] - opp.expertise[i]) for i in range(5)))
            for s in sorted(cloud, key=lambda x: -x.health)[:6]))

    waited = False
    cmd = None
    why = ""

    if me.eta > 0:
        cmd, why = "WAIT", "moving"

    elif me.target not in ("SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY"):
        cmd, why = "GOTO SAMPLES", "leave start"

    elif me.target == "SAMPLES":
        if len(mine) < 3:
            r = choose_rank(E, len(mine))
            # near the end there is no time for expensive medicines
            # (a one-sample cycle from here is ~12 + molecule count turns)
            if tl < 25:
                r = min(r, 2)
            if tl < 17:
                # high expertise makes rank 2 samples often free
                r = 2 if E >= 12 else 1
            # storage clogged with mismatched molecules: cheap samples are the
            # likeliest to be producible from what we already hold
            if sum(me.storage) >= 9 and E < 6:
                r = 1
            cmd, why = "CONNECT %d" % r, "take rank%d" % r
        else:
            cmd, why = "GOTO DIAGNOSIS", "hand full"

    elif me.target == "DIAGNOSIS":
        if mine_undiag:
            cmd, why = "CONNECT %d" % mine_undiag[0].id, "diagnose"
        else:
            # dumping feeds cloud-mining opponents: withhold gifts from them
            # except in emergencies (weak opponents never touch the cloud)
            def safe(s):
                return not opp_mines or gift_risk(s, opp.expertise, tl) < 25
            junk = [s for s in mine_diag if not feasible(s, exp) and safe(s)]
            if not junk:
                # stale: stuck in hand for ages and still not buyable
                junk = [s for s in mine_diag
                        if ages.get(s.id, 0) > 40 and safe(s)
                        and not buyable(s, exp, me.storage, avail)]
            if not junk and tl < 25 and tl > 8:
                # no time left to finish this sample: free the slot - but a
                # dump costs a turn, so never strand a still-producible sample
                hopeless, min_slack = [], 999
                for s in mine_diag:
                    missing = sum(max(0, eff_cost(s, exp)[i] - me.storage[i]) for i in range(5))
                    needed = 5 if missing == 0 else missing + 7
                    if needed > tl and safe(s):
                        hopeless.append(s)
                    elif needed <= tl:
                        min_slack = min(min_slack, tl - needed)
                if hopeless and min_slack >= 2:
                    junk = hopeless
            if not junk and state["blocked"]:
                # samples we marked while starved at MOLECULES: dump those that
                # still cannot be bought (pool starved or storage clogged) and
                # are not part of a currently workable multi-sample plan.
                # This is the deadlock escape, so gifts are allowed here -
                # but dump the cheapest gift first
                p_now = best_plan(mine_diag, exp, opp.expertise, me.storage, avail,
                                  projects, tl=tl, overhead=6)
                keep = set(s.id for s in p_now["perm"]) if p_now else set()
                for s in mine_diag:
                    if s.id in state["blocked"] and s.id not in keep \
                            and not buyable(s, exp, me.storage, avail):
                        junk.append(s)
                if not junk:
                    state["blocked"].clear()
            if junk:
                junk.sort(key=lambda s: gift_risk(s, opp.expertise, tl))
                state["blocked"].discard(junk[0].id)
                cmd, why = "CONNECT %d" % junk[0].id, "dump %d (gift %d)" % (
                    junk[0].id, gift_risk(junk[0], opp.expertise, tl))
            else:
                if len(mine) < 3:
                    best_c, best_ratio = None, 0.0
                    thr = {0: 1.8, 1: 2.2}.get(len(mine), 2.8)
                    for s in samples:
                        if s.carried_by == -1 and s.diagnosed and feasible(s, exp):
                            # only samples we could actually buy right now
                            if not buyable(s, exp, me.storage, avail):
                                continue
                            take = sum(max(0, eff_cost(s, exp)[i] - me.storage[i])
                                       for i in range(5))
                            if 1 + 3 + take + 3 + 1 + len(mine_diag) > tl:
                                continue
                            v = sample_value(s, exp, opp.expertise, projects, tl)
                            ratio = v / (2.0 + take)
                            if ratio > best_ratio:
                                best_c, best_ratio = s, ratio
                    if best_c is not None and best_ratio >= thr:
                        cmd, why = "CONNECT %d" % best_c.id, "cloud pick %d (r=%.2f)" % (best_c.id, best_ratio)
                if cmd is None and len(mine) == 3 and not mine_undiag and tl > 30:
                    # hand full but the cloud holds a gem: swap out the worst
                    # held sample if the upgrade is clearly worth 2 turns
                    best_c, best_net = None, -999.0
                    for s in samples:
                        if s.carried_by == -1 and s.diagnosed and feasible(s, exp) \
                                and buyable(s, exp, me.storage, avail):
                            take = sum(max(0, eff_cost(s, exp)[i] - me.storage[i])
                                       for i in range(5))
                            if 2 + 3 + take + 3 + 1 + len(mine_diag) > tl:
                                continue
                            net = sample_value(s, exp, opp.expertise, projects, tl) - 0.35 * take
                            if net > best_net:
                                best_c, best_net = s, net
                    if best_c is not None and mine_diag:
                        worst = min(mine_diag, key=lambda s: sample_value(s, exp, opp.expertise, projects, tl)
                                    - 0.35 * sum(max(0, eff_cost(s, exp)[i] - me.storage[i]) for i in range(5)))
                        w_net = sample_value(worst, exp, opp.expertise, projects, tl) \
                            - 0.35 * sum(max(0, eff_cost(worst, exp)[i] - me.storage[i]) for i in range(5))
                        if best_net >= w_net + 18 and (not opp_mines
                                                       or gift_risk(worst, opp.expertise, tl) < 25):
                            cmd, why = "CONNECT %d" % worst.id, "swap out %d for cloud %d (+%.0f)" % (
                                worst.id, best_c.id, best_net - w_net)
                if cmd is None:
                    p_direct = best_plan(mine_diag, exp, opp.expertise, me.storage,
                                         [0] * 5, projects, tl=tl, overhead=4)
                    p_mol = best_plan(mine_diag, exp, opp.expertise, me.storage,
                                      avail, projects, tl=tl, overhead=6)
                    if len(mine) == 0:
                        cmd, why = "GOTO SAMPLES", "empty hand"
                    elif len(mine) <= 1 and tl > 40:
                        cmd, why = "GOTO SAMPLES", "top up hand"
                    elif p_direct is not None and (
                            p_mol is None or sum(p_mol["take"]) == 0
                            or p_direct["value"] >= p_mol["value"]):
                        # storage already covers the best plan: skip MOLECULES
                        cmd, why = "GOTO LABORATORY", "funded, direct %s" % [s.id for s in p_direct["perm"]]
                    else:
                        cmd, why = "GOTO MOLECULES", "go collect"

    elif me.target == "MOLECULES":
        # what the opponent is still missing for their diagnosed samples
        opp_need = [0] * 5
        for s in samples:
            if s.carried_by == 1 and s.diagnosed:
                for i in range(5):
                    opp_need[i] += max(0, s.cost[i] - opp.expertise[i])
        opp_need = [max(0, opp_need[i] - opp.storage[i]) for i in range(5)]

        p = best_plan(mine_diag, exp, opp.expertise, me.storage, avail, projects,
                      tl=tl, overhead=3)
        if p is not None:
            state["blocked"].clear()
            if sum(p["take"]) == 0:
                # done gathering: deny the opponent a scarce molecule on the
                # way out if it strands them and we have room and time
                deny = None
                if sum(me.storage) < CARRY and 3 + len(p["perm"]) + 1 <= tl:
                    # true molecule spend: expertise gained mid-sequence
                    # reduces later samples' needs
                    exp_sim = list(exp)
                    spend = 0
                    for s in p["perm"]:
                        for i in range(5):
                            spend += max(0, s.cost[i] - exp_sim[i])
                        if s.gain_idx >= 0:
                            exp_sim[s.gain_idx] += 1
                    junk_after = sum(me.storage) - min(spend, sum(me.storage)) + 1
                    if junk_after <= 5:
                        for i in range(5):
                            if opp_need[i] > 0 and 1 <= avail[i] <= 3 and avail[i] <= opp_need[i]:
                                deny = i if deny is None or avail[i] < avail[deny] else deny
                if deny is not None:
                    cmd, why = "CONNECT %s" % TYPES[deny], "deny %s (opp needs %d)" % (TYPES[deny], opp_need[deny])
                else:
                    cmd, why = "GOTO LABORATORY", "plan ready %s" % [s.id for s in p["perm"]]
            else:
                cands = [i for i in range(5) if p["take"][i] > 0]
                # grab the scarcest contested type first
                i = min(cands, key=lambda j: (avail[j] - p["take"][j], -p["take"][j]))
                cmd = "CONNECT %s" % TYPES[i]
                why = "gather for %s take%s" % ([s.id for s in p["perm"]], p["take"])
        else:
            feas = [s for s in mine_diag if feasible(s, exp)]
            # samples whose remainder fits the carry limit: pool refill can fix
            # these, waiting cannot fix a capacity clog
            cands_s = []
            starved = set()
            for s in feas:
                e = eff_cost(s, exp)
                take = [max(0, e[i] - me.storage[i]) for i in range(5)]
                if sum(me.storage) + sum(take) <= CARRY:
                    cands_s.append(s)
                    for i in range(5):
                        if take[i] > avail[i]:
                            starved.add(i)
            if cands_s and sum(me.storage) < CARRY:
                # pool is blocking a full plan: pre-gather what IS available
                # for the most valuable such sample that can still finish
                # in the remaining turns (missing + travel 3 + produce 1)
                timely = [s for s in cands_s
                          if sum(max(0, eff_cost(s, exp)[i] - me.storage[i])
                                 for i in range(5)) + 4 <= tl]
                if timely:
                    tgt = max(timely, key=lambda s: sample_value(s, exp, opp.expertise, projects, tl))
                    e = eff_cost(tgt, exp)
                    needs = [i for i in range(5) if e[i] - me.storage[i] > 0 and avail[i] > 0]
                    if needs:
                        i = min(needs, key=lambda j: avail[j])
                        cmd, why = "CONNECT %s" % TYPES[i], "partial gather for %d" % tgt.id
            if cmd is None:
                # will the opponent's future lab spend return a starved type?
                opp_spend = [0] * 5
                for s in samples:
                    if s.carried_by == 1 and s.diagnosed:
                        for i in range(5):
                            opp_spend[i] += max(0, s.cost[i] - opp.expertise[i])
                refill_helps = any(min(opp.storage[i], opp_spend[i]) > 0 for i in starved)
                if opp.target == "LABORATORY" and refill_helps:
                    cap = 8
                elif sum(opp.storage) >= 5 and refill_helps:
                    cap = 5
                else:
                    cap = 2
                if cands_s and wait_count < cap and tl > 8:
                    cmd, why, waited = "WAIT", "pool blocked (w%d/%d)" % (wait_count, cap), True
                elif len(mine) < 3 and tl > 25:
                    cmd, why = "GOTO SAMPLES", "stuck, refill hand"
                elif mine_diag and tl > 12:
                    # pool starved or storage clogged and waiting did not help:
                    # flush the hand
                    for s in mine_diag:
                        if not buyable(s, exp, me.storage, avail):
                            state["blocked"].add(s.id)
                    cmd, why = "GOTO DIAGNOSIS", "stuck, dump blocked %s" % sorted(state["blocked"])
                else:
                    # dead turns anyway: starve the opponent if possible
                    deny = None
                    if sum(me.storage) < CARRY - 1:
                        for i in range(5):
                            if opp_need[i] > 0 and 1 <= avail[i] <= 3 and avail[i] <= opp_need[i]:
                                deny = i if deny is None or avail[i] < avail[deny] else deny
                    if deny is not None:
                        cmd, why = "CONNECT %s" % TYPES[deny], "idle deny %s" % TYPES[deny]
                    else:
                        cmd, why, waited = "WAIT", "stuck, hold position", True

    elif me.target == "LABORATORY":
        p = best_plan(mine_diag, exp, opp.expertise, me.storage, [0] * 5, projects,
                      tl=tl, overhead=0)
        if p is not None:
            cmd, why = "CONNECT %d" % p["perm"][0].id, "produce, order %s" % [s.id for s in p["perm"]]
        else:
            p2 = best_plan(mine_diag, exp, opp.expertise, me.storage, avail, projects,
                           tl=tl, overhead=6)
            feas = [s for s in mine_diag if feasible(s, exp)]
            if p2 is not None and tl > 7:
                cmd, why = "GOTO MOLECULES", "fetch mols for %s" % [s.id for s in p2["perm"]]
            elif any(not feasible(s, exp) for s in mine_diag) and tl > 20:
                cmd, why = "GOTO DIAGNOSIS", "rework hand"
            elif len(mine) < 3 and (tl > 18 or (tl > 13 and E >= 10)):
                # with high expertise a free-sample cycle fits in ~13 turns
                cmd, why = "GOTO SAMPLES", "restock"
            elif feas and tl > 7:
                # pool may refill while we travel; partial-gather there
                cmd, why = "GOTO MOLECULES", "chase pool refill"
            else:
                cmd, why = "WAIT", "nothing useful left"

    if cmd is None:
        cmd, why = "WAIT", "fallthrough"

    state["wait"] = wait_count + 1 if waited else 0
    log("  -> %s (%s)" % (cmd, why))
    return cmd


def main():
    read = sys.stdin.readline
    project_count = int(read())
    projects = []
    for _ in range(project_count):
        projects.append([int(x) for x in read().split()])
    log("projects: %s" % projects)

    state = {"wait": 0, "blocked": set(), "age": {},
             "cloud_prev": set(), "opp_mined": 0}
    turn = 0
    while True:
        line = read()
        if not line:
            break
        turn += 1
        cmd = "WAIT"
        try:
            me = parse_player(line)
            opp = parse_player(read())
            avail = [int(x) for x in read().split()]
            n = int(read())
            samples = []
            for _ in range(n):
                samples.append(parse_sample(read().split()))
            cmd = decide(me, opp, avail, samples, projects, turn, state)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            cmd = "WAIT"
        print(cmd)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
