"""Plays Code vs Zombies: hill-climbs a small population of symbolic Ash plans with distinct outcomes (explicit targets, chasing
and striking, kill-avoiding gathering, waypoints and orbits, bait-human ambushes, edge herding, a parametrized evade & strike
policy; fresh plans often open with a threat kill or wait in ambush near the safest human) with an exact incremental game
simulation that evaluates the most killing strike move at every turn, and keeps the elite plans across turns. The simulation is
tuned for speed without changing any result: zombies cache a lower bound on their distance to the nearest human and that human
with a gap margin, the zombie move collects the zombies near Ash for the kill, strike and policy checks, mutated plans reuse the
turns they share with their elite, and random draws skip randrange's argument handling."""
import gc
import math
import random
import sys
import time

W, H = 16000, 9000
FIRST_BUDGET, TURN_BUDGET = 0.015, 0.070
MAX_TURNS = 150
CHAIN = 3
POP, P_BEST, P_CROSS = 2, 0.5, 0.15
# probability of a fresh plan per search iteration (P_FRESH_EARLY in the first EARLY_TURNS turns)
P_FRESH, P_FRESH_EARLY, EARLY_TURNS = 0.3, 0.5, 4
COMBO = [0, 1]
_f = [1, 2]
while len(COMBO) < 101:
    COMBO.append(COMBO[-1] + _f[len(COMBO) - 1])
    _f.append(_f[-1] + _f[-2])
LEFT = [k * math.pi / 8 for k in range(9)] + [-k * math.pi / 8 for k in range(1, 8)]
ROTS = [[(math.cos(o), math.sin(o)) for o in offs] for offs in (sorted(LEFT, key=abs), LEFT, [-o for o in LEFT])]
DIRS = [(math.cos(math.pi * k / 8), math.sin(math.pi * k / 8)) for k in range(16)]
CHASE, AVOID, ORBIT, POLICY = (-3, -3), 20000, -5, -6
P_POLICY = 0.4
LOOKS = [(0, 0), (2200, 1), (2400, 1), (2200, 3)]
DIRS12 = [(int(1000 * math.cos(k * math.tau / 12)), int(1000 * math.sin(k * math.tau / 12))) for k in range(12)] + [(0, 0)]
CENTERS, ORBITS = (0, 1), (0, 1, 2)
STAY = [(1, 0)]
P_THREAT, P_INTERP, P_AMB = 0.5, 0.25, 0.25
getrandbits = random.getrandbits
# a zombie moves at most 401.42 per turn, so one within 4400 (3010) of Ash after its move was within 4802 (3412) before it
NEAR, PRE_NEAR, PRE_KILL = 4400 ** 2, 4802 ** 2, 3412 ** 2
# per LOOKS entry: squared reach of the policy's zombie list, squared look distance, squared distance from a candidate within which
# zombies can block its neighbours, number of free neighbours needed
REACH = [((look + 2000 if look else 3000) ** 2, look ** 2, (look + 1001) ** 2, need) for look, need in LOOKS]
# DIRS12 orders that start with the direction k * 30 degrees
AWAY = [sorted(DIRS12[:12], key=lambda o: -(o[0] * math.cos(k * math.tau / 12) + o[1] * math.sin(k * math.tau / 12))) + [(0, 0)]
        for k in range(12)]
ARC = 6 / math.pi


def main():
    """Reads game turns from stdin and writes Ash targets to stdout."""
    gc.disable()
    plans, turn = [], 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        start = time.perf_counter()
        ax, ay = map(int, line.split())
        hpts = []
        for _ in range(int(sys.stdin.readline())):
            _, x, y = map(int, sys.stdin.readline().split())
            hpts.append((x, y))
        zombies = [tuple(map(int, sys.stdin.readline().split()))[1:] for _ in range(int(sys.stdin.readline()))]
        plans, (tx, ty) = step(make_root(ax, ay, hpts, zombies), plans, turn, start)
        print(tx, ty)
        sys.stdout.flush()
        turn += 1


def step(root: tuple, plans: list[list[tuple]], turn: int, start: float) -> tuple[list[list[tuple]], tuple[int, int]]:
    """Plays one turn: searches from the elite plans of the previous turn, each shifted to the current turn and, except the best
    one whose first move was played, also unshifted.
    :param root: current snapshot
    :param plans: elite plans of the previous turn, best first
    :param turn: turn index
    :param start: perf_counter time the turn started
    :return: elite plans, best first, and Ash's target"""
    plans = [p[1:] if len(p) > 1 or p and p[0][0] != POLICY else p for p in plans] + plans[1:]
    pop = search(root, plans, start + (FIRST_BUDGET if turn == 0 else TURN_BUDGET), P_FRESH_EARLY if turn < EARLY_TURNS else P_FRESH)
    return [e[1] for e in pop], pop[0][2][0]


def search(root: tuple, plans: list[list[tuple]], deadline: float, p_fresh: float) -> list[list]:
    """Improves a population of up to POP elite plans with distinct signatures (see signature) until the deadline by fresh plans (see
    restart) and by hill-climbing an elite (the best one with probability P_BEST, else another one) with a mutation or (with
    probability P_CROSS) a crossover with another elite; ties go to plans that end sooner.
    :param root: current state snapshot
    :param plans: symbolic plans to start from
    :param deadline: perf_counter deadline
    :param p_fresh: probability of a fresh plan per iteration
    :return: elites [score, symbolic plan, resolved Ash targets per turn, snapshots per turn, signature, rank key], best first"""
    pop = []
    for plan in plans or [[]]:
        insert(pop, *simulate([root], [], plan, 0, 0, 0))
    while time.perf_counter() < deadline:
        if random.random() < p_fresh:
            bar = pop[-1][0] if len(pop) == POP else 0
            score, plan, moves, snaps = simulate([root], [], restart(root), 0, 0, bar)
            if score >= bar:
                insert(pop, score, plan, moves, snaps)
            continue
        i = 0 if len(pop) == 1 or random.random() < P_BEST else 1 + below(len(pop) - 1)
        bs, best, res, snaps = pop[i][:4]
        if len(pop) > 1 and random.random() < P_CROSS:
            other, t = pop[(i + 1 + below(len(pop) - 1)) % len(pop)][1], below(len(res))
            plan = (best + [best[-1] if best and best[-1][0] == POLICY else CHASE] * (t - len(best)))[:t] \
                + (other[t:] if t < len(other) else [other[-1] if other and other[-1][0] == POLICY else CHASE])
        else:
            plan, t = mutate(best, len(res), snaps)
        score, plan, moves, new = simulate(snaps, res, plan, t, diverge(best, plan, t, len(res)), bs)
        if score > bs or score == bs and (t + len(moves) <= len(res) if score else t + len(moves) >= len(res)):
            insert(pop, score, plan, res[:t] + moves, snaps[:t] + new)
    return pop


def insert(pop: list[list], score: int, plan: list[tuple], moves: list[tuple[int, int]], snaps: list[tuple]):
    """Inserts a plan into the population: it replaces the elite with its signature unless that one ranks higher, or else takes a
    new place if the population is not full or replaces the lowest ranked elite if it does not rank lower. Elites rank by score,
    then by ending sooner (later if all humans die).
    :param pop: elites, best first (see search; modified)
    :param score: plan score
    :param plan: symbolic plan
    :param moves: resolved Ash targets per turn
    :param snaps: snapshots per turn"""
    sig = signature(score, moves, snaps)
    e = [score, plan, moves, snaps, sig, (-score, len(moves) if score else -len(moves))]
    for j, f in enumerate(pop):
        if f[4] == sig:
            if e[5] > f[5]:
                return
            pop[j] = e
            break
    else:
        if len(pop) < POP:
            pop.append(e)
        elif e[5] <= pop[-1][5]:
            pop[-1] = e
        else:
            return
    pop.sort(key=lambda f: f[5])


def signature(score: int, moves: list[tuple[int, int]], snaps: list[tuple]) -> tuple[tuple, int]:
    """Computes the strategy signature of a plan: the humans alive at its last turn (none if all die) and the direction octant of
    its first move (-1 for a move shorter than 200).
    :param score: plan score
    :param moves: resolved Ash targets per turn
    :param snaps: snapshots per turn
    :return: signature"""
    dx, dy = moves[0][0] - snaps[0][0], moves[0][1] - snaps[0][1]
    return tuple(snaps[-1][2]) if score else (), -1 if dx * dx + dy * dy < 40000 else int(math.atan2(dy, dx) * 4 / math.pi + 4.5) % 8


def simulate(snaps: list[tuple], res: list[tuple[int, int]], plan: list[tuple], t: int, t2: int, bar: int) \
        -> tuple[int, list[tuple], list[tuple[int, int]], list[tuple]]:
    """Simulates a plan from a base plan's snapshot at turn t; while a turn offered a strike (the move that kills the most zombies
    within reach, see strike) that looked better than the outcome, replays the plan with that strike followed by chasing. The turns
    before t2, where the plan's entries equal the base plan's, are taken from the base plan's simulation instead of replayed.
    :param snaps: snapshots per turn of the base plan's simulation (see run)
    :param res: resolved Ash targets per turn of the base plan
    :param plan: symbolic plan (see run)
    :param t: plan index of the start
    :param t2: plan index from which the plan's entries can differ from the base plan's (t if unknown, len(res) if they never do,
        in which case bar must be the base plan's score)
    :param bar: score to beat; strikes that cannot beat it are ignored and hopeless plans are cut short
    :return: final score (-1 if cut short), symbolic plan actually used, resolved Ash targets per turn from t on, snapshots per turn
        from t on"""
    bv, bt, bp = -1, 0, None
    for i in range(t, t2):
        ax, ay, hpts, _, zs, _, _, score, kl, _ = snaps[i]
        kn, n, hs = len(kl), len(zs), len(hpts) ** 2 * 10
        if kn and score + hs * (COMBO[kn] + n - kn) > bar and score + hs * (COMBO[kn] + n - kn) > bv:
            cx, cy, k = strike(ax, ay, kl)
            v = score + hs * (COMBO[k] + n - k)
            if v > bar and v > bv:
                bv, bt, bp = v, i, (cx, cy)
    if t < t2 == len(res):
        score, moves, snaps = bar, res[t:], snaps[t:]
    else:
        score, moves, sn, v, vt, vp = run(snaps[t2], plan, t2, bar)
        moves, snaps = res[t:t2] + moves, snaps[t:t2] + sn
        if v > bv:
            bv, bt, bp = v, vt, vp
    for _ in range(CHAIN):
        if bv <= score or bt - t < len(moves) and moves[bt - t] == bp:
            break
        plan2 = (plan + [plan[-1] if plan and plan[-1][0] == POLICY else CHASE] * (bt - len(plan)))[:bt] + [bp]
        s2, m2, sn2, bv, bt2, bp = run(snaps[bt - t], plan2, bt, max(bar, score))
        if s2 <= score:
            break
        score, plan, moves, snaps, bt = s2, plan2, moves[:bt - t] + m2, snaps[:bt - t] + sn2, bt2
    return score, plan, moves, snaps


def run(snap: tuple, plan: list[tuple], t: int, bar: int) -> tuple[int, list[tuple[int, int]], list[tuple], int, int, tuple[int, int]]:
    """Simulates the game from a snapshot, following the symbolic plan from index t and then chasing (or following the policy of
    a final policy entry), and records the most promising one-move strike.
    :param snap: state at turn t after the zombie move: Ash x, Ash y, alive human points, set of them, zombies, the same zombies before
        their move (their first two fields), points where zombies landed on their target this turn, score so far, (x, y, squared
        distance) of the zombies within strike reach (3010) of Ash and of those within 4400 of Ash (None unless the turn's entry is a
        policy with two-step evasion), both in zombie order. With one alive human a zombie is its point (x, y), with more it is
        (x, y, l, hx, hy, m): l is a lower bound on its distance to the nearest alive human and (hx, hy) is that human if m, a lower
        bound on the gap between the distances to the second nearest and the nearest alive human, is positive
    :param plan: per-turn entries: explicit target (x, y); CHASE (the strike move, see strike, if zombies are within reach, else a
        move towards the nearest zombie); (-1, 3 * c + o) gathering (kill-avoiding move towards center c, striking if all zombies
        can be killed at once); (-2, 3 * c) strike (move towards center c); (x + AVOID, y) (kill-avoiding move towards (x, y));
        (ORBIT, x, y, r) (kill-avoiding circling around (x, y) at radius |r| in the direction of the sign of r); (POLICY, prm)
        (evade & strike policy, see policy; as the last entry it persists until the end). Center c is 0 = zombie centroid,
        1 = centroid of the zombies that target Ash; orbit o selects the order of tried side steps (alternating, left first,
        right first)
    :param t: plan index of the snapshot
    :param bar: score to beat; strikes that cannot beat it are ignored and the simulation stops with score -1 once it cannot
        reach it
    :return: final score, resolved Ash targets per turn from t on, snapshots per turn from t on, best strike value, its turn and
        its target"""
    sqrt, floor = math.sqrt, math.floor
    ax, ay, hpts, pos, zs, pz, arrived, score, kl, nr = snap
    n, hs = len(zs), len(hpts) ** 2 * 10
    bv, bt, bp = -1, 0, None
    snaps, moves, n_plan = [], [], len(plan)
    tail = plan[-1] if n_plan and plan[-1][0] == POLICY else CHASE
    e = plan[t] if t < n_plan else tail
    while True:
        snaps.append((ax, ay, hpts, pos, zs, pz, arrived, score, kl, nr))
        kn = len(kl)
        if kn and score + hs * (COMBO[kn] + n - kn) > bar and score + hs * (COMBO[kn] + n - kn) > bv:
            cx, cy, k = strike(ax, ay, kl)
            v = score + hs * (COMBO[k] + n - k)
            if v > bar and v > bv:
                bv, bt, bp = v, t, (cx, cy)
        tx, ty = e[0], e[1]
        if tx == POLICY:
            tx, ty = policy(ax, ay, hpts, zs, kl, nr, arrived, ty)
        elif tx == -3:
            tx, ty = strike(ax, ay, kl)[:2] if kl else nearest(ax, ay, zs, nr)
        elif tx < 0 or tx >= AVOID:
            if tx == -1 or tx == -2:
                cx, cy = centroid(ax, ay, hpts, zs, pz, ty >= 3)
            if tx == -2:
                tx, ty = floor(cx), floor(cy)
            else:
                if tx == -1:
                    dx, dy = cx - ax, cy - ay
                    d = sqrt(dx * dx + dy * dy) or 1
                    nx, ny = (floor(cx), floor(cy)) if d <= 1000 else (floor(ax + dx * 1000 / d), floor(ay + dy * 1000 / d))
                    k = 0
                    for x, y, _ in kl:
                        if (x - nx) ** 2 + (y - ny) ** 2 <= 4000000:
                            k += 1
                    p = None if k == n else avoid(ax, ay, dx * 1000 / d, dy * 1000 / d, ROTS[ty % 3], kl)
                else:
                    if tx == ORBIT:
                        ox, oy, r = e[1], e[2], e[3]
                        a = math.atan2(ay - oy, ax - ox) + 1050 / r
                        dx, dy = ox + abs(r) * math.cos(a) - ax, oy + abs(r) * math.sin(a) - ay
                    else:
                        dx, dy = tx - AVOID - ax, ty - ay
                    d = sqrt(dx * dx + dy * dy)
                    p = avoid(ax, ay, dx, dy, STAY, kl) if d <= 1000 else None
                    if p is None:
                        if d < 1:
                            cx = cy = 0
                            for x, y, _ in kl:
                                cx, cy = cx + x, cy + y
                            dx, dy = ax - cx / kn, ay - cy / kn
                            d = sqrt(dx * dx + dy * dy) or 1
                        p = avoid(ax, ay, dx * 1000 / d, dy * 1000 / d, ROTS[0], kl)
                        if p is None:
                            cx, cy = centroid(ax, ay, hpts, zs, pz, False)
                tx, ty = p or (floor(cx), floor(cy))
        moves.append((tx, ty))
        dx, dy = tx - ax, ty - ay
        d = dx * dx + dy * dy
        if d <= 1000000:
            ax, ay = tx, ty
        else:
            d = 1000 / sqrt(d)
            ax, ay = floor(ax + dx * d), floor(ay + dy * d)
        if kn:
            k = 0
            for x, y, _ in kl:
                if (x - ax) ** 2 + (y - ay) ** 2 <= 4000000:
                    k += 1
            if k:
                score += hs * COMBO[k]
                n -= k
                if not n:
                    return score, moves, snaps, bv, bt, bp
                nzs = []
                for z in zs:
                    if (z[0] - ax) ** 2 + (z[1] - ay) ** 2 > 4000000:
                        nzs.append(z)
                zs = nzs
        if arrived:
            eaten = set()
            for p in arrived:
                if (p[0] - ax) ** 2 + (p[1] - ay) ** 2 > 4000000 and p in pos:
                    eaten.add(p)
            if eaten:
                hpts, zs = eat(hpts, zs, eaten)
                if not hpts:
                    return 0, moves, snaps, bv, bt, bp
                pos = set(hpts)
                hs = len(hpts) ** 2 * 10
        if score + hs * COMBO[n] < bar:
            return -1, moves, snaps, bv, bt, bp
        t += 1
        if t == MAX_TURNS:
            return score, moves, snaps, bv, bt, bp
        e = plan[t] if t < n_plan else tail
        if e[0] == POLICY and e[1][7]:
            pre, near, nr = PRE_NEAR, NEAR, []
        else:
            pre, near, nr = PRE_KILL, -1, None
        pz, nzs, arrived, kl = zs, [], [], []
        if len(hpts) == 1:
            (hx, hy), = hpts
            for x, y in zs:
                dx, dy = ax - x, ay - y
                da = dx * dx + dy * dy
                u, v = hx - x, hy - y
                bd = u * u + v * v
                if da <= bd:
                    if da > 161202:  # a 400-step from farther than 401.5 cannot land exactly on its target
                        d = 400 / sqrt(da)
                        x, y = floor(x + dx * d), floor(y + dy * d)
                    elif da <= 160000:
                        x, y = ax, ay
                        arrived.append((ax, ay))
                    else:
                        d = 400 / sqrt(da)
                        x, y = floor(x + dx * d), floor(y + dy * d)
                        if x == ax and y == ay:
                            arrived.append((ax, ay))
                elif bd > 161202:
                    d = 400 / sqrt(bd)
                    x, y = floor(x + u * d), floor(y + v * d)
                elif bd <= 160000:
                    x, y = hx, hy
                    arrived.append((hx, hy))
                else:
                    d = 400 / sqrt(bd)
                    x, y = floor(x + u * d), floor(y + v * d)
                    if x == hx and y == hy:
                        arrived.append((hx, hy))
                nzs.append((x, y))
                if da <= pre:
                    d = (x - ax) ** 2 + (y - ay) ** 2
                    if d <= 9060100:
                        kl.append((x, y, d))
                    if d <= near:
                        nr.append((x, y, d))
        else:
            for x, y, lb, hx, hy, m in zs:
                dx, dy = ax - x, ay - y
                da = dx * dx + dy * dy
                if da > lb * lb:
                    if m > 0:
                        u, v = hx - x, hy - y
                        bd = u * u + v * v
                    else:
                        bd = d2 = 1 << 62
                        for px, py in hpts:
                            d = (px - x) ** 2 + (py - y) ** 2
                            if d < bd:
                                d2, bd, hx, hy = bd, d, px, py
                            elif d < d2:
                                d2 = d
                        m = sqrt(d2) - sqrt(bd)
                        u, v = hx - x, hy - y
                    lb = sqrt(bd)
                    if da > bd:
                        if bd > 161202:
                            d = 400 / lb
                            x, y = floor(x + u * d), floor(y + v * d)
                        elif bd <= 160000:
                            x, y = hx, hy
                            arrived.append((hx, hy))
                        else:
                            d = 400 / lb
                            x, y = floor(x + u * d), floor(y + v * d)
                            if x == hx and y == hy:
                                arrived.append((hx, hy))
                        lb -= 402
                        nzs.append((x, y, lb if lb > 0 else 0, hx, hy, m - 3))
                        if da <= pre:
                            d = (x - ax) ** 2 + (y - ay) ** 2
                            if d <= 9060100:
                                kl.append((x, y, d))
                            if d <= near:
                                nr.append((x, y, d))
                        continue
                if da > 161202:
                    d = 400 / sqrt(da)
                    x, y = floor(x + dx * d), floor(y + dy * d)
                elif da <= 160000:
                    x, y = ax, ay
                    arrived.append((ax, ay))
                else:
                    d = 400 / sqrt(da)
                    x, y = floor(x + dx * d), floor(y + dy * d)
                    if x == ax and y == ay:
                        arrived.append((ax, ay))
                lb -= 402
                nzs.append((x, y, lb if lb > 0 else 0, hx, hy, m - 803))
                if da <= pre:
                    d = (x - ax) ** 2 + (y - ay) ** 2
                    if d <= 9060100:
                        kl.append((x, y, d))
                    if d <= near:
                        nr.append((x, y, d))
        zs = nzs


def strike(ax: int, ay: int, kl: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Finds the strike move that kills the most zombies within reach among Ash stepping towards their centroid, staying, and
    stepping onto one end (the same one for every zombie) of the arc of the circle of radius 999 around Ash that lies within 1999 of
    a zombie, as the circle's points that kill the most include such an end; stops at the first move that kills them all.
    :param ax: Ash x
    :param ay: Ash y
    :param kl: (x, y, squared distance) of the zombies within 3010 of Ash (not empty)
    :return: target x, target y and the number of zombies the move kills"""
    n, cx, cy = len(kl), 0, 0
    for x, y, _ in kl:
        cx, cy = cx + x, cy + y
    cands = [(cx // n, cy // n), (ax, ay)]
    for x, y, d in kl:
        if 1000000 < d < 8988004:
            a = (d - 2998000) / (2 * d)
            h = math.sqrt(998001 / d - a * a)
            tx, ty = round(ax + a * (x - ax) + h * (y - ay)), round(ay + a * (y - ay) - h * (x - ax))
            if 0 <= tx < W and 0 <= ty < H:
                cands.append((tx, ty))
    bk = -1
    for tx, ty in cands:
        dx, dy = tx - ax, ty - ay
        d = dx * dx + dy * dy
        if d <= 1000000:
            px, py = tx, ty
        else:
            d = 1000 / math.sqrt(d)
            px, py = math.floor(ax + dx * d), math.floor(ay + dy * d)
        k = 0
        for x, y, _ in kl:
            if (x - px) ** 2 + (y - py) ** 2 <= 4000000:
                k += 1
        if k > bk:
            bk, bx, by = k, tx, ty
            if k == n:
                break
    return bx, by, bk


def policy(ax: int, ay: int, hpts: list[tuple[int, int]], zs: list[tuple], kl: list[tuple[int, int, int]],
           nr: list[tuple[int, int, int]] | None, arrived: list[tuple[int, int]], prm: tuple[int, ...]) -> tuple[int, int]:
    """Chooses Ash's target by the evade & strike policy: strike when a zombie in reach stands on a protected human or when the
    best move kills at least the threshold, otherwise take the kill-free move closest to the anchor. Candidate moves are handled
    as offsets from Ash and zombies as offsets from Ash or from the candidate.
    :param ax: Ash x
    :param ay: Ash y
    :param hpts: alive human points
    :param zs: zombies after their move (see run)
    :param kl: (x, y, squared distance) of the zombies within 3010 of Ash
    :param nr: (x, y, squared distance) of the zombies within 4400 of Ash, or None if not computed
    :param arrived: points where zombies landed on their target this turn
    :param prm: strike threshold, anchor mode (0 point, 1 orbit around the point, 2 nearest zombie, 3 human nearest to the point,
        4 orbit around the zombie centroid), anchor x, anchor y, signed orbit radius, number of protected humans nearest to the
        point, forced-kill mode (1 fewest kills, 0 most), two-step evasion mode (index in LOOKS)
    :return: target point"""
    sqrt, floor = math.sqrt, math.floor
    thr, mode, px, py, rad, prot, forced, look = prm
    reach, look, pre, need = REACH[look]
    if look:
        cl = within(ax, ay, zs, reach) if nr is None else nr if reach == NEAR else [q for q in nr if q[2] <= reach]
        kc, n3 = kl, 0
        for q in kl:
            if q[2] <= 9000000:
                n3 += 1
    else:
        cl = kc = [q for q in kl if q[2] <= 9000000]
        n3 = len(cl)
    if mode == 0:
        gx, gy = px, py
    elif mode == 1:
        a = math.atan2(ay - py, ax - px) + 1000 / rad
        gx, gy = min(max(int(px + abs(rad) * math.cos(a)), 0), W - 1), min(max(int(py + abs(rad) * math.sin(a)), 0), H - 1)
    elif mode == 2:
        gx, gy = nearest(ax, ay, zs, kl or nr)
    elif mode == 3:
        bd = 1 << 62
        for x, y in hpts:
            d = (x - px) ** 2 + (y - py) ** 2
            if d < bd:
                bd, gx, gy = d, x, y
    else:
        cx, cy = centroid(ax, ay, hpts, zs, zs, False)
        a = math.atan2(ay - cy, ax - cx) + 1000 / rad
        gx, gy = min(max(int(cx + abs(rad) * math.cos(a)), 0), W - 1), min(max(int(cy + abs(rad) * math.sin(a)), 0), H - 1)
    if not cl:
        return gx, gy
    ex, ey = gx - ax, gy - ay
    d = ex * ex + ey * ey
    og = (ex, ey) if d <= 1000000 else (floor(ax + ex * 1000 / sqrt(d)) - ax, floor(ay + ey * 1000 / sqrt(d)) - ay)
    cx = cy = 0
    for x, y, _ in cl:
        cx, cy = cx + x, cy + y
    cx, cy = cx // len(cl) - ax, cy // len(cl) - ay
    d = cx * cx + cy * cy
    oc = (cx, cy) if d <= 1000000 else (floor(ax + cx * 1000 / sqrt(d)) - ax, floor(ay + cy * 1000 / sqrt(d)) - ay)
    offs = (DIRS12 if 1000 <= ax < W - 1000 and 1000 <= ay < H - 1000 else ring(ax, ay)) + [og, oc]
    kr = []
    for x, y, _ in kc:
        kr.append((x - ax, y - ay))
    threat = guards(ax, ay, hpts, arrived, px, py, prot) if arrived and (prot or len(hpts) == 1) else ()
    thr = min(thr, len(zs))
    if threat or n3 >= thr:
        bkey = None
        for ox, oy in offs:
            k = j = 0
            for x, y in kr:
                if (x - ox) ** 2 + (y - oy) ** 2 <= 4000000:
                    k += 1
            for x, y in threat:
                if (x - ox) ** 2 + (y - oy) ** 2 <= 4000000:
                    j += 1
            key = j, k, -(ex - ox) ** 2 - (ey - oy) ** 2
            if bkey is None or key > bkey:
                bkey, tx, ty = key, ox, oy
        if bkey[0] or bkey[1] >= thr:
            return ax + tx, ay + ty
    keys = [(ex - ox) ** 2 + (ey - oy) ** 2 for ox, oy in offs]
    order = sorted(range(15), key=keys.__getitem__)
    bc, lx, ly = -1, 1 << 30, 0  # (lx, ly): the zombie that blocked the previous candidate, tried first
    for i in order:
        ox, oy = offs[i]
        if (lx - ox) ** 2 + (ly - oy) ** 2 <= 4000000:
            continue
        for lx, ly in kr:
            if (lx - ox) ** 2 + (ly - oy) ** 2 <= 4000000:
                break
        else:
            nx, ny = ax + ox, ay + oy
            if not look:
                return nx, ny
            lc, sx, sy = [], 0, 0
            for x, y, _ in cl:
                x, y = x - nx, y - ny
                if x * x + y * y <= pre:
                    lc.append((x, y))
                    sx, sy = sx + x, sy + y
            if not lc:
                return nx, ny
            c = 0  # free neighbours, counted away from the zombies first (the outcome does not depend on the order)
            for dx, dy in AWAY[int(math.atan2(-sy, -sx) * ARC + 12.5) % 12]:
                if 0 <= nx + dx < W and 0 <= ny + dy < H:
                    for x, y in lc:
                        if (x - dx) ** 2 + (y - dy) ** 2 <= look:
                            break
                    else:
                        c += 1
                        if c >= need:
                            return nx, ny
            if c > bc:
                bc, tx, ty = c, nx, ny
    if bc >= 0:
        return tx, ty
    bk = -1
    for i in order:
        ox, oy = offs[i]
        k = 0
        for x, y in kr:
            if (x - ox) ** 2 + (y - oy) ** 2 <= 4000000:
                k += 1
        if bk < 0 or (k < bk if forced else k > bk):
            bk, tx, ty = k, ox, oy
    return ax + tx, ay + ty


def within(ax: int, ay: int, zs: list[tuple], reach: int) -> list[tuple[int, int, int]]:
    """Lists the zombies within a distance of Ash.
    :param ax: Ash x
    :param ay: Ash y
    :param zs: zombies (see run)
    :param reach: squared distance
    :return: (x, y, squared distance) of the zombies within reach in zombie order"""
    near = []
    for z in zs:
        d = (z[0] - ax) ** 2 + (z[1] - ay) ** 2
        if d <= reach:
            near.append((z[0], z[1], d))
    return near


def nearest(ax: int, ay: int, zs: list[tuple], near: list[tuple[int, int, int]] | None) -> tuple[int, int]:
    """Finds the zombie nearest to Ash (the first one on ties).
    :param ax: Ash x
    :param ay: Ash y
    :param zs: zombies (see run)
    :param near: (x, y, squared distance) of the zombies within some distance of Ash in zombie order, or None
    :return: zombie point"""
    bd = 1 << 62
    if near:
        for x, y, d in near:
            if d < bd:
                bd, tx, ty = d, x, y
    else:
        for z in zs:
            d = (z[0] - ax) ** 2 + (z[1] - ay) ** 2
            if d < bd:
                bd, tx, ty = d, z[0], z[1]
    return tx, ty


def centroid(ax: int, ay: int, hpts: list[tuple[int, int]], zs: list[tuple], pz: list[tuple], chasers: bool) -> tuple[float, float]:
    """Computes the centroid of the zombies or, if asked and there are any, of the zombies that targeted Ash in their last move.
    :param ax: Ash x
    :param ay: Ash y
    :param hpts: alive human points
    :param zs: zombies (see run)
    :param pz: zombies before their last move
    :param chasers: whether to use only the zombies that targeted Ash
    :return: centroid"""
    sx = sy = sn = 0
    if chasers:
        for p, z in zip(pz, zs):
            x, y = p[0], p[1]
            d = (ax - x) ** 2 + (ay - y) ** 2
            for px, py in hpts:
                if (px - x) ** 2 + (py - y) ** 2 < d:
                    break
            else:
                sx, sy, sn = sx + z[0], sy + z[1], sn + 1
        if sn:
            return sx / sn, sy / sn
    for z in zs:
        sx, sy = sx + z[0], sy + z[1]
    return sx / len(zs), sy / len(zs)


def ring(ax: int, ay: int) -> list[tuple[int, int]]:
    """Lists the offsets of Ash's 1000-steps in 12 directions and staying, clamped to the map.
    :param ax: Ash x
    :param ay: Ash y
    :return: candidate offsets"""
    lx, ly, hx, hy = -ax, -ay, W - 1 - ax, H - 1 - ay
    return [(lx if dx < lx else hx if dx > hx else dx, ly if dy < ly else hy if dy > hy else dy) for dx, dy in DIRS12]


def guards(ax: int, ay: int, hpts: list[tuple[int, int]], arrived: list[tuple[int, int]], px: int, py: int, prot: int) \
        -> list[tuple[int, int]]:
    """Lists the zombie arrivals within 3000 of Ash on the protected humans (the max(prot, 1) humans nearest to (px, py)) as
    offsets from Ash.
    :param ax: Ash x
    :param ay: Ash y
    :param hpts: alive human points
    :param arrived: points where zombies landed on their target this turn
    :param px: protection point x
    :param py: protection point y
    :param prot: number of protected humans
    :return: threatened arrival offsets (one per zombie)"""
    guard = set(sorted(hpts, key=lambda h: (h[0] - px) ** 2 + (h[1] - py) ** 2)[:max(prot, 1)])
    return [(p[0] - ax, p[1] - ay) for p in arrived if p in guard and (p[0] - ax) ** 2 + (p[1] - ay) ** 2 <= 9000000]


def avoid(ax: int, ay: int, cb: float, sb: float, rot: list[tuple[float, float]], near: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    """Finds the first Ash step among rotations of a step vector after which no zombie is in kill range.
    :param ax: Ash x
    :param ay: Ash y
    :param cb: step vector x
    :param sb: step vector y
    :param rot: rotations (cos, sin) to try in order
    :param near: (x, y, squared distance to Ash) of the zombies that could be in kill range
    :return: target point (clamped to the map) or None if every rotation kills"""
    floor = math.floor
    for co, so in rot:
        nx, ny = floor(ax + cb * co - sb * so), floor(ay + sb * co + cb * so)
        nx, ny = 0 if nx < 0 else W - 1 if nx >= W else nx, 0 if ny < 0 else H - 1 if ny >= H else ny
        for x, y, _ in near:
            if (x - nx) ** 2 + (y - ny) ** 2 <= 4000000:
                break
        else:
            return nx, ny
    return None


def eat(hpts: list[tuple[int, int]], zs: list[tuple], eaten: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple]]:
    """Removes eaten humans and updates the zombies' cached nearest humans (see run).
    :param hpts: alive human points
    :param zs: zombies
    :param eaten: points of the eaten humans
    :return: alive human points and zombies"""
    hpts = [p for p in hpts if p not in eaten]
    if len(hpts) > 1:
        return hpts, [(z[0], z[1], z[2], 0, 0, 0) if (z[3], z[4]) in eaten else z for z in zs]
    return hpts, [(z[0], z[1]) for z in zs]


def restart(snap: tuple) -> list[tuple]:
    """Draws a fresh plan from a snapshot: with probability P_AMB an ambush, the evade & strike policy that waits at a spot near the
    safest human (see spot), protects the human nearest to the spot and strikes only when it kills all zombies; otherwise random
    explicit moves or (with probability P_THREAT) a threat kill, then a random tail. The threat kill holds as explicit target the
    position of the zombie nearest to a random human (the first one on ties; or, with probability P_INTERP, a random point between
    them) for about as many turns as Ash needs to get it within kill range.
    :param snap: state where the plan starts (see run)
    :return: symbolic plan"""
    ax, ay, hpts, _, zs = snap[:5]
    if random.random() < P_AMB:
        return [(POLICY, (len(zs), 0, *spot(hpts, zs), 2000, 1, below(2), below(4)))]
    if random.random() >= P_THREAT:
        return random_plan(ax, ay, below(10)) + random_tail(snap)
    hx, hy = hpts[below(len(hpts))]
    zx, zy = min(zs, key=lambda z: (z[0] - hx) ** 2 + (z[1] - hy) ** 2)[:2]
    f = random.random() if random.random() < P_INTERP else 0
    tx, ty = int(zx + (hx - zx) * f), int(zy + (hy - zy) * f)
    return [(tx, ty)] * max(1, int((math.hypot(tx - ax, ty - ay) - 2000) / 1000) + 1 + below(2)) + random_tail(snap)


def spot(hpts: list[tuple[int, int]], zs: list[tuple]) -> tuple[int, int]:
    """Picks the waiting spot of an ambush: 2000-3200 from the human farthest from its nearest zombie, on its side away from the
    zombies (their centroid or, with probability 1/2, their inverse-square weighted centroid around the human), turned in 15 degree
    steps alternately to both sides (starting with a random side, with gaussian angle noise) until it lies on the map.
    :param hpts: alive human points
    :param zs: zombies (see run)
    :return: waiting spot"""
    hx, hy = max(hpts, key=lambda h: min((z[0] - h[0]) ** 2 + (z[1] - h[1]) ** 2 for z in zs))
    sw, sx, sy = len(zs), sum(z[0] for z in zs), sum(z[1] for z in zs)
    if random.random() < 0.5:
        sw = sx = sy = 0
        for z in zs:
            w = 1 / ((z[0] - hx) ** 2 + (z[1] - hy) ** 2 + 1)
            sw, sx, sy = sw + w, sx + w * z[0], sy + w * z[1]
    a, d, s = math.atan2(hy - sy / sw, hx - sx / sw), 2000 + below(1201), 2 * below(2) - 1
    for k in range(13):
        b = a + s * ((k + 1) // 2) * (1 if k % 2 else -1) * math.pi / 12 + random.gauss(0, 0.2)
        x, y = hx + d * math.cos(b), hy + d * math.sin(b)
        if 0 <= x < W and 0 <= y < H:
            break
    return min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1)


def random_plan(ax: int, ay: int, k: int) -> list[tuple]:
    """Generates k random explicit Ash targets starting from Ash's position (stay/direction moves are resolved along the path).
    :param ax: Ash x
    :param ay: Ash y
    :param k: number of moves
    :return: plan"""
    plan = []
    for _ in range(k):
        tx, ty = random_move(ax, ay)
        plan.append((tx, ty))
        dx, dy = tx - ax, ty - ay
        d = dx * dx + dy * dy
        ax, ay = (tx, ty) if d <= 1000000 else (int(ax + dx * 1000 / math.sqrt(d)), int(ay + dy * 1000 / math.sqrt(d)))
    return plan


def mutate(plan: list[tuple], n: int, snaps: list[tuple]) -> tuple[list[tuple], int]:
    """Applies a random mutation to the symbolic plan: replace one entry by a random move (optionally cutting the rest), replace
    the rest by a random tail, perturb one explicit target, waypoint, orbit or policy, duplicate or delete one entry.
    :param plan: symbolic plan (not modified)
    :param n: number of simulated turns of the plan
    :param snaps: snapshots of the plan's simulation
    :return: mutated plan and the first changed index"""
    t = below(n)
    plan = plan + [plan[-1] if plan and plan[-1][0] == POLICY else CHASE] * (t + 1 - len(plan))
    r = random.random()
    if r < 0.35:
        plan[t] = random_move(snaps[t][0], snaps[t][1])
        if random.random() < 0.3:
            del plan[t + 1:]
    elif r < 0.7:
        plan[t:] = random_tail(snaps[t])
    elif r < 0.85:
        e = plan[t]
        if e[0] == POLICY:
            j = t
            while j < len(plan) and plan[j] == e:
                j += 1
            plan[t:j] = [(POLICY, mutate_prm(e[1], len(snaps[t][4]), len(snaps[t][2])))] * (j - t)
        elif e[0] == ORBIT:
            j = t
            while j < len(plan) and plan[j] == e:
                j += 1
            r = abs(e[3]) + int(random.gauss(0, 300))
            plan[t:j] = [(ORBIT, min(max(int(random.gauss(e[1], 600)), 0), W - 1), min(max(int(random.gauss(e[2], 600)), 0), H - 1),
                          max(r, 2100) * (-1 if (e[3] < 0) != (random.random() < 0.2) else 1))] * (j - t)
        else:
            x, y = e
            if x < 0:
                x, y = snaps[t][0], snaps[t][1]
            a = AVOID if x >= AVOID else 0
            plan[t] = min(max(int(random.gauss(x - a, 600)), 0), W - 1) + a, min(max(int(random.gauss(y, 600)), 0), H - 1)
    elif random.random() < 0.5:
        plan.insert(t, plan[t])
    elif t < len(plan):
        del plan[t]
    return plan, t


def random_move(ax: int, ay: int) -> tuple[int, int]:
    """Draws a random explicit Ash target: stay, a 1000-step in one of 16 directions or a uniform point.
    :param ax: Ash x
    :param ay: Ash y
    :return: target point"""
    r = random.random()
    if r < 0.1:
        return ax, ay
    if r < 0.4:
        c, s = DIRS[below(16)]
        return min(max(int(ax + 1000 * c), 0), W - 1), min(max(int(ay + 1000 * s), 0), H - 1)
    return below(W), below(H)


def random_tail(snap: tuple) -> list[tuple]:
    """Draws a random symbolic tail: the evade & strike policy, nothing (nearest-zombie chase), an optional chase with
    kill-avoiding waypoints and gathering, a bait-human ambush, edge herding along corners and edge midpoints or an orbit.
    :param snap: state where the tail starts (see run)
    :return: symbolic plan entries"""
    if random.random() < P_POLICY:
        return [(POLICY, random_prm(len(snap[4]), len(snap[2])))]
    r = random.random()
    if r < 0.35:
        return []
    ax, ay, hpts, _, zs = snap[:5]
    if r < 0.65:
        tail = [CHASE] * below(16) if random.random() < 0.2 else []
        for _ in range(below(3)):
            tail += [(below(W) + AVOID, below(H))] * (1 + below(15))
        return tail + [(-1, CENTERS[below(2)] * 3 + ORBITS[below(3)])] * (1 + below(40))
    cx, cy = sum(z[0] for z in zs) / len(zs), sum(z[1] for z in zs) / len(zs)
    if r < 0.8:
        hx, hy = hpts[below(len(hpts))]
        a, d = math.atan2(hy - cy, hx - cx) + random.gauss(0, 0.5), 1800 + below(1401)
        return [(min(max(int(hx + d * math.cos(a)), 0), W - 1) + AVOID, min(max(int(hy + d * math.sin(a)), 0), H - 1))] \
            * (5 + below(76))
    if r < 0.92:
        i = below(301)
        cyc = [(i, i), (W // 2, i), (W - 1 - i, i), (W - 1 - i, H // 2), (W - 1 - i, H - 1 - i), (W // 2, H - 1 - i), (i, H - 1 - i),
               (i, H // 2)]
        k = min(range(8), key=lambda j: (cyc[j][0] - ax) ** 2 + (cyc[j][1] - ay) ** 2)
        s, tail = 2 * below(2) - 1, []
        for _ in range(1 + below(5)):
            k = (k + s * (1 + below(2))) % 8
            tail += [(cyc[k][0] + AVOID, cyc[k][1])] * (2 + below(14))
        return tail
    r = 2400 + below(1601)
    ox = min(max(int(cx) + below(6001) - 3000, r), W - r)
    oy = min(max(int(cy) + below(4001) - 2000, r // 2), H - r // 2)
    return [(ORBIT, ox, oy, (2 * below(2) - 1) * r)] * (5 + below(76))


def mutate_prm(prm: tuple[int, ...], nz: int, nh: int) -> tuple[int, ...]:
    """Changes one policy parameter: moves the anchor point, changes the orbit radius or the threshold, or redraws one.
    :param prm: policy parameters
    :param nz: number of zombies alive
    :param nh: number of humans alive
    :return: mutated parameters"""
    prm, k = list(prm), below(8)
    if k == 2 and random.random() < 0.7:
        prm[2], prm[3] = min(max(int(random.gauss(prm[2], 1000)), 0), W - 1), min(max(int(random.gauss(prm[3], 1000)), 0), H - 1)
    elif k == 4 and random.random() < 0.7:
        prm[4] = max(int(random.gauss(abs(prm[4]), 400)), 1500) * (1 if prm[4] > 0 else -1)
    elif k == 0 and random.random() < 0.7:
        prm[0] = max(1, prm[0] + (-2, -1, 1, 2)[below(4)])
    else:
        prm[k] = random_prm(nz, nh)[k]
    return tuple(prm)


def random_prm(nz: int, nh: int) -> tuple[int, ...]:
    """Samples random evade & strike policy parameters (see policy).
    :param nz: number of zombies alive
    :param nh: number of humans alive
    :return: policy parameters"""
    return (nz, nz, 1 + below(nz))[below(3)], below(5), below(W), below(H), (2 * below(2) - 1) * (2000 + below(2501)), below(nh + 1), \
        below(2), below(4)


def below(n: int) -> int:
    """Draws a uniform random integer below n exactly like random.randrange(n) (and randint and choice, which reduce to it) does
    in CPython 3.11, consuming the same random bits, but without randrange's argument handling.
    :param n: positive bound
    :return: random integer in [0, n)"""
    k = n.bit_length()
    r = getrandbits(k)
    while r >= n:
        r = getrandbits(k)
    return r


def diverge(best: list[tuple], plan: list[tuple], t: int, n: int) -> int:
    """Finds the first plan index from t on where two symbolic plans (with their persisting tails) give different entries.
    :param best: first plan
    :param plan: second plan
    :param t: start index
    :param n: index limit
    :return: first differing index, or n if there is none before it"""
    tb, tp = best[-1] if best and best[-1][0] == POLICY else CHASE, plan[-1] if plan and plan[-1][0] == POLICY else CHASE
    while t < n and (plan[t] if t < len(plan) else tp) == (best[t] if t < len(best) else tb):
        t += 1
    return t


def make_root(ax: int, ay: int, hpts: list[tuple[int, int]], zombies: list[tuple[int, int, int, int]]) -> tuple:
    """Builds the simulation snapshot of the current turn.
    :param ax: Ash x
    :param ay: Ash y
    :param hpts: alive human points
    :param zombies: zombie current and next positions
    :return: snapshot (see run)"""
    pos, kl, nr = set(hpts), [], []
    zs = [(z[2], z[3]) if len(hpts) == 1 else (z[2], z[3], 0, 0, 0, 0) for z in zombies]
    for x, y, *_ in zs:
        d = (x - ax) ** 2 + (y - ay) ** 2
        if d <= NEAR:
            nr.append((x, y, d))
            if d <= 9060100:
                kl.append((x, y, d))
    return ax, ay, hpts, pos, zs, zombies, [(z[2], z[3]) for z in zombies if (z[2], z[3]) in pos], 0, kl, nr


if __name__ == "__main__":
    main()
