import sys

# ---------------------------------------------------------------------------
# Code4Life style bot.
#
# Pipeline per cycle:
#   1. COLLECT  : grab up to 3 undiagnosed samples at SAMPLES (ranks chosen by
#                 a schedule that ramps up with accumulated expertise).
#   2. DIAGNOSE : reveal molecule costs at DIAGNOSIS.
#   3. (DUMP)   : if a diagnosed sample needs > 10 molecules even after
#                 expertise it can never be carried -> send it to the cloud.
#   4. GATHER   : at MOLECULES, buy molecules for the best affordable subset of
#                 carried diagnosed samples (max total health, fits in 10 slots,
#                 respects per-type availability).
#   5. PRODUCE  : at LABORATORY, research the affordable samples for points +
#                 expertise.
#
# All decisions are derived from the observable state each turn (only a tiny
# "stuck" counter persists) so the bot self-heals if the world changes
# (opponent grabs molecules, samples move, etc.).
# ---------------------------------------------------------------------------

TYPES = "ABCDE"
N = 5
CAP_MOL = 10
MAX_SAMPLES = 3


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Sample:
    __slots__ = ("id", "carried", "rank", "gain", "health", "cost")

    def __init__(self, sid, carried, rank, gain, health, cost):
        self.id = sid
        self.carried = carried      # 0 = me, 1 = opponent, -1 = cloud
        self.rank = rank
        self.gain = gain            # letter A-E (or '0'/'-1' if undiagnosed)
        self.health = health
        self.cost = cost            # list[5]; all -1 while undiagnosed

    @property
    def diagnosed(self):
        return self.cost[0] >= 0


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def collection_ranks(expertise):
    """Which sample ranks to grab this collection trip (3 of them).

    Be patient with rank 3: early rank-3 samples often need > 10 molecules
    (infeasible without expertise) and get wasted.  Instead, farm cheap rank-1
    first to build balanced expertise fast (also helps science projects), then
    rank 2, and only commit to rank 3 once expertise makes them cheap."""
    et = sum(expertise)
    if et >= 12:
        return [3, 3, 3]
    if et >= 8:
        return [3, 3, 2]
    if et >= 4:
        return [2, 2, 2]
    return [1, 1, 1]


def total_need(s, expertise):
    """Total molecules still required for sample s, accounting for expertise."""
    tot = 0
    c = s.cost
    for t in range(N):
        d = c[t] - expertise[t]
        if d > 0:
            tot += d
    return tot


def required_for_subset(subset, expertise):
    req = [0] * N
    for s in subset:
        c = s.cost
        for t in range(N):
            d = c[t] - expertise[t]
            if d > 0:
                req[t] += d
    return req


def affordable(s, storage, expertise):
    c = s.cost
    for t in range(N):
        if storage[t] < c[t] - expertise[t]:
            return False
    return True


def project_bonus(gain, expertise, projects):
    """Extra value of completing a sample, from the expertise it grants toward
    the still-open science projects (50 pts each, need expertise in every type).

    Completing this sample gives +1 expertise of type `gain`.  For every project
    that still needs that type, we credit a fraction of its 50 points scaled by
    how close the project is (closer -> larger credit; the last point -> +50)."""
    if not projects or gain not in TYPES:
        return 0.0
    g = TYPES.index(gain)
    b = 0.0
    for p in projects:
        rem_g = p[g] - expertise[g]
        if rem_g <= 0:
            continue  # this project no longer needs type g (met or over)
        total_rem = 0
        for t in range(N):
            d = p[t] - expertise[t]
            if d > 0:
                total_rem += d
        if total_rem > 0:
            b += 50.0 / total_rem
    return b


def choose_targets(diag, storage, expertise, available, projects=None):
    """Pick the subset of carried diagnosed samples to complete this trip.

    Maximises total value (health + science-project progress) subject to:
      - final molecules held (max(storage, required)) fits in 10 slots,
      - molecules we still need to buy are actually available.
    Tie-breaks toward fewer molecules to buy, then more samples.
    """
    n = len(diag)
    bonus = {s.id: project_bonus(s.gain, expertise, projects) for s in diag}
    best = None
    best_key = None
    for mask in range(1, 1 << n):
        subset = [diag[i] for i in range(n) if mask & (1 << i)]
        req = required_for_subset(subset, expertise)
        held_sum = 0
        buy_total = 0
        ok = True
        for t in range(N):
            fh = storage[t] if storage[t] > req[t] else req[t]
            held_sum += fh
            buy = req[t] - storage[t]
            if buy > 0:
                if buy > available[t]:
                    ok = False
                    break
                buy_total += buy
        if not ok or held_sum > CAP_MOL:
            continue
        value = 0.0
        for s in subset:
            value += s.health + bonus[s.id]
        key = (value, -buy_total, len(subset))
        if best_key is None or key > best_key:
            best_key = key
            best = subset
    return best if best else []


# ---------------------------------------------------------------------------
# Per-turn decision
# ---------------------------------------------------------------------------

def decide(loc, eta, storage, expertise, available, samples, stuck, projects=None):
    """Return (command_string, new_stuck)."""
    if eta > 0:
        return "WAIT", stuck

    carried = [s for s in samples if s.carried == 0]
    undiag = [s for s in carried if not s.diagnosed]
    diag = [s for s in carried if s.diagnosed]

    # ---- COLLECT: still building the batch (no diagnosed yet, room for more)
    if not diag and len(carried) < MAX_SAMPLES:
        if loc == "SAMPLES":
            ranks = collection_ranks(expertise)
            r = ranks[len(carried)]
            return "CONNECT %d" % r, 0
        return "GOTO SAMPLES", 0

    # ---- DIAGNOSE undiagnosed samples
    if undiag:
        if loc == "DIAGNOSIS":
            return "CONNECT %d" % undiag[0].id, 0
        return "GOTO DIAGNOSIS", 0

    # ---- everything carried is diagnosed now ----

    # Dump samples that can never be carried (need > capacity even w/ expertise)
    infeasible = [s for s in diag if total_need(s, expertise) > CAP_MOL]
    if infeasible:
        if loc == "DIAGNOSIS":
            return "CONNECT %d" % infeasible[0].id, 0
        return "GOTO DIAGNOSIS", 0

    subset = choose_targets(diag, storage, expertise, available, projects)

    if not subset:
        # Can't complete anything right now (availability). Wait for refills;
        # if blocked too long, dump the most expensive sample to change plans.
        stuck += 1
        if stuck >= 8 and diag:
            worst = max(diag, key=lambda s: total_need(s, expertise))
            if loc == "DIAGNOSIS":
                return "CONNECT %d" % worst.id, 0
            return "GOTO DIAGNOSIS", stuck
        return "WAIT", stuck

    req = required_for_subset(subset, expertise)
    need_more = any(storage[t] < req[t] for t in range(N))

    if need_more:
        if loc == "MOLECULES":
            cands = [t for t in range(N) if storage[t] < req[t] and available[t] > 0]
            if cands:
                # grab the scarcest needed molecule first (beat the opponent to it)
                t = min(cands, key=lambda t: available[t])
                return "CONNECT %s" % TYPES[t], 0
            # nothing buyable right now: produce what we can, else wait
            if any(affordable(s, storage, expertise) for s in subset):
                return "GOTO LABORATORY", 0
            return "WAIT", stuck + 1
        return "GOTO MOLECULES", 0

    # ---- have all molecules: PRODUCE
    if loc == "LABORATORY":
        aff = [s for s in subset if affordable(s, storage, expertise)]
        if aff:
            target = max(aff, key=lambda s: s.health)
            return "CONNECT %d" % target.id, 0
        return "WAIT", stuck
    return "GOTO LABORATORY", 0


# ---------------------------------------------------------------------------
# I/O loop
# ---------------------------------------------------------------------------

def read_robot(inp):
    p = inp().split()
    target = p[0]
    eta = int(p[1])
    score = int(p[2])
    storage = [int(x) for x in p[3:8]]
    expertise = [int(x) for x in p[8:13]]
    return target, eta, score, storage, expertise


def main():
    inp = input
    project_count = int(inp())
    projects = [list(map(int, inp().split())) for _ in range(project_count)]
    log("projects:", projects)

    turn = 0
    stuck = 0
    while True:
        try:
            my_target, my_eta, my_score, storage, expertise = read_robot(inp)
        except EOFError:
            break
        _opp_target, _opp_eta, opp_score, _ostore, _oexp = read_robot(inp)
        available = [int(x) for x in inp().split()]
        sample_count = int(inp())
        samples = []
        for _ in range(sample_count):
            q = inp().split()
            samples.append(Sample(
                int(q[0]), int(q[1]), int(q[2]), q[3], int(q[4]),
                [int(x) for x in q[5:10]],
            ))

        turn += 1
        try:
            cmd, stuck = decide(my_target, my_eta, storage, expertise,
                                available, samples, stuck, projects)
        except Exception as e:  # never crash -> never produce invalid output
            log("ERR", repr(e))
            cmd = "WAIT"

        log("T%d me=%d opp=%d loc=%s eta=%d exp=%s avail=%s -> %s" % (
            turn, my_score, opp_score, my_target, my_eta,
            expertise, available, cmd))
        print(cmd, flush=True)


if __name__ == "__main__":
    main()
