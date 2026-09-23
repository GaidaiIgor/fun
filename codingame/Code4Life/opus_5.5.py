"""Code4Life bot: opponent-aware molecule play on a set/order-aware sample planner (merged round-2 improvements)."""
import os
import sys
import time
from itertools import combinations, permutations

TY = "ABCDE"
R5 = range(5)
SAM, DIA, MOL, LAB, START = "SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY", "START_POS"
_DM = {(SAM, DIA): 3, (SAM, MOL): 3, (SAM, LAB): 3, (DIA, MOL): 3, (DIA, LAB): 4, (MOL, LAB): 3}
MAXT = 200


def dist(a, b):
    if a == b:
        return 0
    if a == START or b == START:
        return 2
    return _DM.get((a, b)) or _DM[(b, a)]


def prm(name, default):
    v = os.environ.get("WF_" + name)
    return float(v) if v is not None else default


RATE = prm("RATE", 2.0)          # points per turn used to price time
MOLC = prm("MOLC", 1.0)          # cost of one molecule take inside plan comparisons
EXPW = prm("EXPW", 12.0)          # base value of one expertise point early in the game
LEFTF = prm("LEFTF", 0.7)        # credit for a kept sample completable on a later trip
SLOTPEN = prm("SLOTPEN", 5.0)    # penalty for holding a sample that cannot be completed
DENYF = prm("DENYF", 0.6)        # fraction of blocked opponent value credited to denial
JUNK = prm("JUNK", 1.5)          # cost of holding one useless molecule
WAITMAX = prm("WAITMAX", 8.0)    # max turns to wait at MOLECULES for released molecules
T1 = prm("T1", 6.0)              # rank 1 below T1 total expertise
T2 = prm("T2", 10.0)             # rank 2 below T2 total expertise, rank 3 above
GIFT = prm("GIFT", 0.3)          # penalty weight for dumping a sample the opponent can use
RACE = prm("RACE", 1.0)          # 1: race-aware plan choice at MOLECULES
ENDH = prm("ENDH", 185.0)         # horizon (turns left) of the endgame denial evaluator (0 disables)
ENDT = prm("ENDT", 30.0)         # turns left below which denial is valued at full opponent value
REFSLACK = prm("REFSLACK", 2.0)  # capacity slack allowed when refilling with unfinished samples held
RKMODE = prm("RKMODE", 0.0)      # 1: capacity-budget rank choice (U1/U2 cap the rank, BUD = need budget)
U1 = prm("U1", 3.0)
U2 = prm("U2", 8.0)
BUD = prm("BUD", 11.0)
SLOTB = prm("SLOTB", 0.0)        # effective expertise bonus per sample already held when choosing a rank
REFT = prm("REFT", 50.0)         # turns left below which refills use the endgame batch check
XLEFT = prm("XLEFT", 40.0)       # turns left at which the generic expertise value reaches 0
PF1 = prm("PF1", 1.0)            # project credit factor when I am closer than the opponent
PF2 = prm("PF2", 0.7)            # ... when equally close
PF3 = prm("PF3", 0.4)            # ... when farther
PFEAS = prm("PFEAS", 9.0)        # turns per missing expertise for a project to count as reachable
PROJW = prm("PROJW", 1.0)        # multiplier of project value in expertise value
DENYKE = prm("DENYKE", 5.0)      # max molecules for one late denial
ENDKC = prm("ENDKC", 0.5)        # per-molecule cost of a late denial
ENDMIN = prm("ENDMIN", 5.0)      # min blocked value for the endgame denial evaluator
ENDOWN = prm("ENDOWN", 10.0)     # bonus kept for own refill potential in endgame denial
KEEPM = prm("KEEPM", 0.5)        # hysteresis of keep_set
LEFTC = prm("LEFTC", 2.0)        # fixed cost of a kept later-trip sample
FRUIT = prm("FRUIT", 2.0)        # fruitless MOLECULES visits before giving up
CLOUDH = prm("CLOUDH", 10.0)     # min health for a cloud sample to be worth a DIAGNOSIS detour
EXSL = prm("EXSL", 8.0)          # slack turns needed to pre-take molecules for later samples
PFMIN = prm("PFMIN", 0.5)        # min feasibility of a rank in endgame batches
GIFT2 = prm("GIFT2", 2.0)        # gift multiplier when the opponent needs <=2 molecules for a dumped sample
BLKS = prm("BLKS", 25.0)         # turns left above which blocked samples lead to SAMPLES
XB = prm("XB", 1.5)              # expertise balance: scales the per-level deviation XD of an expertise value
XD = [prm("XD0", 0.4), prm("XD1", 0.2), 0.0, prm("XD3", -0.5), prm("XD4", -0.6), prm("XD5", -0.7)]
HOLDH = prm("HOLDH", 40.0)       # research holding only when turns left <= HOLDH
DLAB = prm("DLAB", 0.0)          # 1: mid-game denial also while the opponent is at LAB with leftovers
DK = prm("DK", 3.0)              # max molecules of one mid-game denial
DPRE = prm("DPRE", 0.0)          # 1: denial is considered before pre-taking for later samples
DFULL = prm("DFULL", 0.0)        # extra credit when a denial leaves the opponent nothing completable
PICK = prm("PICK", 0.0)          # 2: take order chosen by simulating the race vs. the modelled opponent
PKEY = prm("PKEY", 0.0)          # take-order key: 0 base, 1 contested class + small rem, 2 base + small rem, 3 class + big rem
HOLD = prm("HOLD", 1.0)          # 1: endgame - delay research whose released molecules would unblock the opponent
UNDIAG = prm("UNDIAG", 1.0)      # 1: undiagnosed samples in hand go to DIAGNOSIS (not SAMPLES) and block deny trips
OSTART = prm("OSTART", 0.0)      # 1: an opponent at LAB with a molecule plan counts as coming to MOL

# deck prior for undiagnosed samples (costs only)
_D1 = ["11110", "12110", "22010", "00131", "00210", "20200", "00300", "04000", "10111", "10121", "10220", "01310",
       "10002", "00202", "00003", "00040", "01111", "01211", "02201", "31001", "00021", "02002", "03000", "00400",
       "11011", "11012", "01022", "13100", "21000", "02020", "00030", "00004", "11101", "21101", "20102", "10013",
       "02100", "20020", "30000", "40000"]
_D2 = ["32200", "30302", "01420", "00530", "50000", "00006", "02230", "02303", "53000", "20014", "05000", "06000",
       "00322", "23030", "00142", "00053", "00050", "60000", "30230", "23002", "42001", "05300", "00500", "00600",
       "20023", "03023", "14200", "30005", "00005", "00060"]
_D3 = ["33530", "00070", "00363", "00073", "30335", "70000", "63003", "73000", "03353", "00007", "30036", "30007",
       "53033", "07000", "36300", "07300", "35303", "00700", "03630", "00730"]
DECK = {r: [[int(ch) for ch in c] for c in d] for r, d in ((1, _D1), (2, _D2), (3, _D3))}
HEALTH = {1: 2.1, 2: 18.3, 3: 40.0}


def rank_stats(rank, ex, cap=10):
    """Expected need (over feasible samples) and feasibility probability of an undiagnosed sample."""
    tot, nf = 0, 0
    for c in DECK[rank]:
        n = 0
        ok = True
        for t in R5:
            d = c[t] - ex[t]
            if d > 0:
                n += d
                if d > 5:
                    ok = False
        if ok and n <= cap:
            nf += 1
            tot += n
    if nf == 0:
        return 99.0, 0.0
    return tot / nf, nf / len(DECK[rank])


class Robot:
    __slots__ = ("target", "eta", "score", "st", "ex")

    def __init__(self, p):
        self.target, self.eta, self.score = p[0], int(p[1]), int(p[2])
        self.st = [int(x) for x in p[3:8]]
        self.ex = [int(x) for x in p[8:13]]


class Sample:
    __slots__ = ("id", "carrier", "rank", "gain", "health", "cost", "diag")

    def __init__(self, p):
        self.id, self.carrier, self.rank = int(p[0]), int(p[1]), int(p[2])
        self.gain = TY.index(p[3]) if p[3] in TY else -1
        self.health = int(p[4])
        self.cost = [int(x) for x in p[5:10]]
        self.diag = self.health >= 0

    def __repr__(self):
        return str(self.id)


def need_of(s, ex):
    return [s.cost[t] - ex[t] if s.cost[t] > ex[t] else 0 for t in R5]


def order_need(order, ex):
    e = list(ex)
    tot = [0] * 5
    for s in order:
        c = s.cost
        for t in R5:
            d = c[t] - e[t]
            if d > 0:
                tot[t] += d
        e[s.gain] += 1
    return tot


def pkey(on, margin, rem, av):
    if PKEY == 1:
        return (on > 0 and margin < 0 and av >= rem, on > 0, -margin, -rem)
    if PKEY == 2:
        return (margin < 0 and on > 0, -margin, -rem)
    if PKEY == 3:
        return (on > 0 and margin < 0 and av >= rem, on > 0, -margin, rem)
    return (margin < 0 and on > 0, -margin, rem)


def vec(v):
    return "".join(str(x) if 0 <= x < 10 else f"({x})" for x in v)


class Plan:
    __slots__ = ("order", "extra", "m", "score", "time", "need")

    def __init__(self, order, extra, need, score, tm):
        self.order, self.extra, self.need, self.m, self.score, self.time = order, extra, need, sum(extra), score, tm


def all_plans(samples, st, ex, av, max_time, loc, val, cap=10):
    """All subsets (best order each) completable from storage st plus availability av in one molecule trip,
    sorted by score (best first)."""
    out = []
    held = sum(st)
    for k in range(len(samples), 0, -1):
        for comb in combinations(samples, k):
            bo = None
            for perm in permutations(comb):
                need = order_need(perm, ex)
                extra = [need[t] - st[t] if need[t] > st[t] else 0 for t in R5]
                m = sum(extra)
                if held + m > cap or any(extra[t] > av[t] for t in R5):
                    continue
                if bo is None or m < bo[0]:
                    bo = (m, perm, extra, need)
            if bo is None:
                continue
            m, perm, extra, need = bo
            tm = (dist(loc, MOL) + m + dist(MOL, LAB) if m else dist(loc, LAB)) + k
            if tm > max_time:
                continue
            out.append(Plan(list(perm), extra, need, sum(val(s) for s in comb) - MOLC * m, tm))
    out.sort(key=lambda p: -p.score)
    return out


def best_plan(samples, st, ex, av, max_time, loc, val, cap=10):
    ps = all_plans(samples, st, ex, av, max_time, loc, val, cap)
    return ps[0] if ps else None


class Bot:
    def __init__(self):
        n = int(input())
        self.projects = [[int(x) for x in input().split()] for _ in range(n)]
        self.turn = 0
        self.diag_by = {}
        self.dumped = {}
        self.total = None
        self.waits = 0
        self.fruitless = 0      # consecutive MOLECULES visits that ended without taking anything
        self.took_here = False
        self.o_prev = None
        self.o_deny = 0         # observed opponent molecule takes not needed by its own samples
        self.why = ""

    # ------------------------------------------------------------------ input
    def read(self):
        self.me = me = Robot(input().split())
        self.op = op = Robot(input().split())
        self.av = [max(0, int(x)) for x in input().split()]
        n = int(input())
        self.samples = [Sample(input().split()) for _ in range(n)]
        if self.total is None:
            self.total = [self.av[t] + me.st[t] + op.st[t] for t in R5]
        for s in self.samples:
            if s.diag and s.id not in self.diag_by:
                self.diag_by[s.id] = s.carrier
        self.mine = [s for s in self.samples if s.carrier == 0]
        self.opp = [s for s in self.samples if s.carrier == 1]
        self.cloud = [s for s in self.samples if s.carrier == -1]
        self.left = MAXT - self.turn
        self.open_proj = [p for p in self.projects if not self.done_proj(p, me.ex) and not self.done_proj(p, op.ex)]
        self.xv = [self.xval(g, me.ex, op.ex) for g in R5]
        self.oxv = [self.xval(g, op.ex, me.ex) for g in R5]
        self.track_opp()
        self.opp_model()

    @staticmethod
    def done_proj(p, ex):
        return all(ex[t] >= p[t] for t in R5)

    def track_opp(self):
        """Detects opponent molecule takes that none of its diagnosed samples need (denial behaviour)."""
        op = self.op
        if self.o_prev is not None and op.target == MOL:
            pst, _ = self.o_prev
            for t in R5:
                if op.st[t] > pst[t]:
                    want = sum(need_of(s, op.ex)[t] for s in self.opp if s.diag)
                    if op.st[t] > want:
                        self.o_deny += 1
        self.o_prev = (list(op.st), op.target)

    # ------------------------------------------------------------------ values
    def xval(self, g, ex, oex):
        """Value of one more expertise of type g for the player with expertise ex."""
        left = self.left
        v = EXPW * max(0, left - XLEFT) / (200 - XLEFT) * (1 + XB * XD[min(ex[g], 5)])
        for p in self.open_proj:
            if ex[g] >= p[g]:
                continue
            mm = sum(max(0, p[t] - ex[t]) for t in R5)
            mo = sum(max(0, p[t] - oex[t]) for t in R5)
            if mm * PFEAS > left + 10:
                continue
            f = PF1 if mm < mo else PF2 if mm == mo else PF3
            v += 50 * PROJW * f / mm
        return v

    def sval(self, s):
        return s.health + self.xv[s.gain]

    def oval(self, s):
        return s.health + self.oxv[s.gain]

    # ------------------------------------------------------------------ opponent model
    def opp_model(self):
        op = self.op
        od = [s for s in self.opp if s.diag]
        self.o_diag = od
        ready, e, st = [], list(op.ex), list(op.st)
        progress = True
        while progress:
            progress = False
            for s in od:
                if s in ready:
                    continue
                nd = need_of(s, e)
                if all(nd[t] <= st[t] for t in R5):
                    ready.append(s)
                    for t in R5:
                        st[t] -= nd[t]
                    e[s.gain] += 1
                    progress = True
        self.o_ready = ready
        oloc = op.target if op.target != START else SAM
        self.o_plan = best_plan(od, op.st, op.ex, self.av, self.left - op.eta, oloc, self.oval) if od else None
        self.o_extra = self.o_plan.extra if self.o_plan else [0] * 5
        # molecules of the opponent that its current plan will consume (the rest is locked for now)
        cons = [min(op.st[t], self.o_plan.need[t]) for t in R5] if self.o_plan else [0] * 5
        if op.target == LAB and ready:
            cons = [op.st[t] - st[t] for t in R5]
        self.o_locked = [op.st[t] - cons[t] for t in R5]
        # predicted release events: (turn offset when visible, molecule vector)
        self.o_rel = []
        if op.target == LAB and ready:
            e = list(op.ex)
            for i, s in enumerate(ready):
                self.o_rel.append((op.eta + i + 1, need_of(s, e)))
                e[s.gain] += 1
        elif self.o_plan and op.target in (MOL, DIA):
            base = op.eta + (0 if op.target == MOL else 3 + sum(1 for s in self.opp if not s.diag)) + self.o_plan.m + 3
            e = list(op.ex)
            for i, s in enumerate(self.o_plan.order):
                self.o_rel.append((base + i + 1, need_of(s, e)))
                e[s.gain] += 1

    def opp_start(self):
        """Turns until the opponent can take molecules (99 if not heading there)."""
        op = self.op
        if op.target == MOL:
            return op.eta
        if op.target == DIA and op.eta == 0 and self.o_plan and self.o_plan.m > 0:
            return 3 + sum(1 for s in self.opp if not s.diag)
        if OSTART > 0 and op.target == LAB and self.o_plan and self.o_plan.m > 0:
            return op.eta + len(self.o_ready) + 3
        return 99

    def opp_takes_before(self, delay):
        """Molecules the opponent is predicted to take before I can start taking in `delay` turns."""
        got = [0] * 5
        if not self.o_plan or self.o_plan.m == 0:
            return got
        n = max(0, delay - self.opp_start())
        rem = list(self.o_extra)
        order = sorted(R5, key=lambda t: self.av[t])
        while n > 0 and sum(rem) > 0:
            for t in order:
                if rem[t] > 0 and n > 0:
                    rem[t] -= 1
                    got[t] += 1
                    n -= 1
        return got

    def pred_av(self, delay, releases=True):
        """Predicted availability when I can start taking molecules `delay` turns from now."""
        tk = self.opp_takes_before(delay)
        av = [max(0, self.av[t] - tk[t]) for t in R5]
        if releases:
            for when, v in self.o_rel:
                if when <= delay:
                    for t in R5:
                        av[t] += v[t]
        return av

    def supply(self):
        """Molecules I could eventually hold of each type (everything not locked by the opponent)."""
        return [self.total[t] - self.o_locked[t] for t in R5]

    def race_ok(self, extra):
        """Simulates simultaneous taking at MOLECULES vs. the opponent; True if I collect `extra`."""
        start = self.opp_start()
        m = sum(extra)
        if start >= m:
            return True
        av = [self.total[t] - self.me.st[t] - self.op.st[t] for t in R5]
        rem, orem = list(extra), list(self.o_extra)
        blocker = self.o_deny > 0
        ocap = 10 - sum(self.op.st)
        for i in range(m):
            t = self.race_pick(rem, orem, av, i >= start)
            if t is None:
                return False
            u = None
            if i >= start and ocap > 0:
                cand = [x for x in R5 if orem[x] > 0 and av[x] > 0]
                if cand:
                    u = min(cand, key=lambda x: av[x])
                elif blocker:
                    cand = [x for x in R5 if rem[x] > 0 and av[x] > 0 and av[x] - rem[x] + 1 <= min(3, ocap)]
                    if cand:
                        u = min(cand, key=lambda x: av[x] - rem[x])
            if u == t:
                av[t] -= 2
                rem[t] -= 1
                orem[t] = max(0, orem[t] - 1)
                ocap -= 1
            else:
                av[t] -= 1
                rem[t] -= 1
                if u is not None:
                    av[u] -= 1
                    orem[u] = max(0, orem[u] - 1)
                    ocap -= 1
        return True

    def race_ok3(self, extra):
        start = self.opp_start()
        if start >= sum(extra):
            return True
        return any(self.race_sim(extra, t, start) == 0 for t in R5
                   if extra[t] > 0 and self.total[t] - self.me.st[t] - self.op.st[t] > 0)

    @staticmethod
    def race_pick(rem, orem, av, active):
        best, bk = None, None
        for t in R5:
            if rem[t] <= 0 or av[t] <= 0:
                continue
            on = orem[t] if active else 0
            margin = av[t] - rem[t] - on
            k = pkey(on, margin, rem[t], av[t])
            if bk is None or k > bk:
                best, bk = t, k
        return best

    # ------------------------------------------------------------------ decisions
    def act(self):
        me = self.me
        if me.eta > 0:
            self.why = "moving"
            return "WAIT"
        loc = me.target
        undiag = any(not s.diag for s in self.mine)
        ready = loc == LAB and any(s.diag and all(me.st[t] >= s.cost[t] - me.ex[t] for t in R5) for s in self.mine)
        if self.left <= ENDH and loc != START and not (loc in (SAM, DIA) and undiag and self.left > 8) \
                and not ready:
            d = self.endgame_deny()
            if d is not None:
                return d
        if loc == START:
            self.why = "start"
            return "GOTO " + SAM
        if loc == SAM:
            return self.at_samples()
        if loc == DIA:
            return self.at_diagnosis()
        if loc == MOL:
            return self.at_molecules()
        return self.at_lab()

    def opp_mol_start(self):
        """Turns until the opponent is expected to start taking molecules for its diagnosed samples."""
        op = self.op
        if op.target == MOL:
            return op.eta
        undiag = sum(1 for s in self.opp if not s.diag)
        if op.target == DIA:
            return op.eta + undiag + 3
        if op.target == LAB:
            return op.eta + len(self.o_ready) + 3
        return op.eta + 3 + undiag + 3 + 3

    def opp_end_value(self, st, av, t0):
        """Value of the opponent's best plan from storage st and pool av when it starts taking in t0 turns."""
        op = self.op
        p = best_plan(self.o_diag, st, op.ex, av, self.left - t0, MOL, self.oval)
        return (sum(self.oval(s) for s in p.order), p) if p else (0.0, None)

    def sim_deny(self, types, arr, ost):
        """Simulates me taking `types` from turn `arr` while the opponent collects its plan from turn `ost`.
        Returns the opponent's achievable value afterwards."""
        op = self.op
        av = [self.total[t] - self.me.st[t] - op.st[t] for t in R5]
        orem = list(self.e_plan.extra)
        ogot = [0] * 5
        mine = list(types)
        for step in range(arr + len(types) + sum(orem) + ost + 1):
            mt = mine[0] if step >= arr and mine else None
            ot = None
            if step >= ost and sum(orem) > 0:
                cand = [x for x in R5 if orem[x] > 0 and av[x] > 0]
                if cand:
                    ot = min(cand, key=lambda x: av[x])
            if mt is not None and mt == ot:
                if av[mt] >= 1:
                    av[mt] -= 2
                    orem[mt] -= 1
                    ogot[mt] += 1
                mine.pop(0)
                continue
            if mt is not None:
                if av[mt] >= 1:
                    av[mt] -= 1
                mine.pop(0)
            if ot is not None and av[ot] >= 1:
                av[ot] -= 1
                orem[ot] -= 1
                ogot[ot] += 1
            if not mine and sum(orem) == 0:
                break
        st2 = [op.st[t] + ogot[t] for t in R5]
        return self.opp_end_value(st2, [max(0, a) for a in av], ost)[0]

    def endgame_deny(self):
        """Endgame: blocks the opponent's last completions when that beats my own remaining scoring."""
        me, op = self.me, self.op
        if not self.o_diag or me.eta > 0:
            return None
        ost = self.opp_mol_start()
        vo, ep = self.opp_end_value(op.st, self.av, ost)
        if ep is None or ep.m == 0:
            return None
        self.e_plan = ep
        arr = dist(me.target, MOL)
        room = 10 - sum(me.st)
        if room <= 0 or arr >= self.left:
            return None
        if UNDIAG > 0 and arr > 0 and any(not s.diag for s in self.mine):
            return None
        best, bg = None, 0.0
        for t in R5:
            if ep.extra[t] <= 0 or self.av[t] <= 0:
                continue
            k = self.av[t] - ep.extra[t] + 1
            if k > room or arr + k > self.left:
                continue
            v2 = self.sim_deny([t] * k, arr, ost)
            g = vo - v2
            if g > bg + 1e-9:
                best, bg = (t, k), g
        if best is None or bg < ENDMIN:
            return None
        t, k = best
        # my own scoring potential with and without spending the denial turns
        diag = [s for s in self.mine if s.diag]
        own = best_plan(diag, me.st, me.ex, self.av, self.left, me.target, self.sval) if diag else None
        v_own = sum(s.health for s in own.order) if own else 0.0
        st2 = list(me.st)
        st2[t] += k
        av2 = list(self.av)
        av2[t] -= k
        own2 = best_plan(diag, st2, me.ex, av2, self.left - arr - k, MOL, self.sval) if diag else None
        v_own2 = sum(s.health for s in own2.order) if own2 else 0.0
        if v_own2 < v_own and self.left > 25 and self.endgame_batch(me.target) is not None:
            v_own += ENDOWN
        net = bg - (v_own - v_own2)
        lead = me.score + v_own - op.score - vo
        if net <= 0 or lead > 0 and me.score + v_own2 - op.score - vo + bg < lead:
            return None
        if me.target == MOL:
            self.took_here = True
            self.why = f"END deny {TY[t]}x{k} blocks {bg:.0f} (own -{v_own - v_own2:.0f}) ost{ost}"
            return f"CONNECT {TY[t]}"
        return self.goto(MOL, f"END go deny {TY[t]}x{k} blocks {bg:.0f} ost{ost}")

    def goto(self, mod, why):
        self.why = why
        if mod == self.me.target:
            return "WAIT"
        if self.me.target == MOL:
            self.fruitless = 0 if self.took_here else self.fruitless + 1
            self.took_here = False
        return "GOTO " + mod

    # ---------------- SAMPLES
    def choose_rank(self):
        if RKMODE > 0:
            me = self.me
            sx = sum(me.ex)
            top = 1 if sx < U1 else 2 if sx < U2 else 3
            held = sum(sum(need_of(s, me.ex)) if s.diag else rank_stats(s.rank, me.ex)[0] for s in self.mine)
            f = 3 - len(self.mine)
            for r in range(top, 0, -1):
                en, pf = rank_stats(r, me.ex)
                if pf >= 0.5 and held + f * en <= BUD:
                    return r
            return 1
        sx = sum(self.me.ex) + SLOTB * len(self.mine)
        return 1 if sx < T1 else 2 if sx < T2 else 3

    def batch_time(self, loc, k, r):
        """Estimated turns to fetch k new rank-r samples starting at loc and complete them with the held ones."""
        me = self.me
        en, pf = rank_stats(r, me.ex)
        held_need = sum(sum(need_of(s, me.ex)) if s.diag else rank_stats(s.rank, me.ex)[0] for s in self.mine)
        undiag = sum(1 for s in self.mine if not s.diag)
        m = max(0.0, held_need + k * en - 0.5 * sum(me.st))
        trips = 1 if m + sum(me.st) <= 10 else 2
        return dist(loc, SAM) + k + 3 + undiag + k + 3 + m + 3 * (2 * trips - 1) + len(self.mine) + k

    def endgame_batch(self, loc):
        """Best (k, rank) of new samples to fetch that can still be completed in time, or None."""
        free = 3 - len(self.mine)
        best, bv = None, 0.0
        for k in range(1, free + 1):
            for r in (1, 2, 3):
                en, pf = rank_stats(r, self.me.ex)
                if pf < PFMIN or self.batch_time(loc, k, r) + 1 > self.left:
                    continue
                v = k * (HEALTH[r] + 1) * pf
                if v > bv:
                    best, bv = (k, r), v
        return best

    def at_samples(self):
        n = len(self.mine)
        if n < 3:
            if self.left >= REFT:
                r = self.choose_rank()
                self.why = f"take r{r}"
                return f"CONNECT {r}"
            eb = self.endgame_batch(SAM)
            if eb:
                self.why = f"take r{eb[1]} (end k{eb[0]})"
                return f"CONNECT {eb[1]}"
        if any(not s.diag for s in self.mine):
            return self.goto(DIA, "diagnose")
        if self.mine:
            return self.goto(MOL, "have diag")
        return self.endgame_move("samples idle")

    # ---------------- DIAGNOSIS
    def at_diagnosis(self):
        me = self.me
        left = self.left
        undiag = [s for s in self.mine if not s.diag]
        if undiag and left > 3 + 3 + 2:
            self.why = "diag"
            return f"CONNECT {undiag[0].id}"
        choice = self.keep_set()
        if choice is not None:
            return choice
        diag = [s for s in self.mine if s.diag]
        if not diag:
            if left > 3 + 1 + 3 + 1 + 3 + 3 + 3:
                return self.goto(SAM, "nothing kept")
            return self.endgame_move("diag idle")
        av = self.pred_av(3)
        plan = best_plan(diag, me.st, me.ex, av, left, DIA, self.sval)
        if plan and plan.m == 0:
            return self.goto(LAB, "all ready")
        if plan and self.fruitless < FRUIT:
            return self.goto(MOL, f"collect {plan.order}")
        for delay in range(4, int(WAITMAX) + 4):
            p2 = best_plan(diag, me.st, me.ex, self.pred_av(delay), left - delay, DIA, self.sval)
            if p2 and self.fruitless < FRUIT:
                return self.goto(MOL, f"collect after release {p2.order}")
        if len(self.mine) < 3 and left > BLKS:
            return self.goto(SAM, "blocked->samples")
        # all three held samples blocked: dump the least valuable one
        worst = min(diag, key=lambda s: self.sval(s))
        self.dumped[worst.id] = sum(me.ex)
        self.why = f"dump blocked {worst.id}"
        return f"CONNECT {worst.id}"

    def set_value(self, sub, av, left):
        me = self.me
        p = best_plan(list(sub), me.st, me.ex, av, left, DIA, self.sval)
        v = 0.0
        e2, st2 = list(me.ex), list(me.st)
        rest = list(sub)
        if p:
            v = p.score
            for s in p.order:
                e2[s.gain] += 1
            st2 = [me.st[t] - p.need[t] for t in R5]
            rest = [s for s in sub if s not in p.order]
            left -= p.time
        sup = self.supply()
        for s in rest:
            nd = need_of(s, e2)
            ext = [max(0, nd[t] - st2[t]) for t in R5]
            if all(nd[t] <= sup[t] for t in R5) and sum(st2) + sum(ext) <= 10 and left > sum(ext) + 8:
                v += LEFTF * self.sval(s) - MOLC * sum(ext) - LEFTC
            else:
                v -= SLOTPEN
        return v

    def keep_set(self):
        """Chooses which carried/cloud samples to hold; returns a CONNECT action or None if the set is final."""
        me, left = self.me, self.left
        carried = [s for s in self.mine if s.diag]
        sx = sum(me.ex)
        sup = self.supply()
        cl = []
        for s in self.cloud:
            if self.dumped.get(s.id) == sx:
                continue
            nd = need_of(s, me.ex)
            if any(nd[t] > sup[t] for t in R5) or sum(nd) > 10:
                continue
            cl.append((self.sval(s) / (sum(nd) + 3), s))
        cl.sort(key=lambda x: -x[0])
        cloud = [s for _, s in cl[:4]]
        cands = carried + cloud
        av = self.pred_av(3)
        base = None
        best, bset = None, None
        for k in range(0, min(3, len(cands)) + 1):
            for sub in combinations(cands, k):
                dumps = [s for s in carried if s not in sub]
                takes = [s for s in sub if s.carrier == -1]
                cost = len(dumps) + len(takes)
                v = self.set_value(sub, av, left - cost) - RATE * cost
                for s in dumps:
                    v -= GIFT * self.opp_usable(s)
                if not takes and not dumps:
                    base = v
                if best is None or v > best + 1e-9:
                    best, bset = v, sub
        if base is not None and best <= base + KEEPM:
            return None
        for s in carried:
            if s not in bset:
                self.dumped[s.id] = sx
                self.why = f"dump {s.id} ({best:.0f} vs {base:.0f})"
                return f"CONNECT {s.id}"
        takes = [s for s in bset if s.carrier == -1]
        takes.sort(key=lambda s: (self.cloud_risky(s), -self.sval(s)))
        for s in takes:
            self.why = f"cloud {s.id} ({best:.0f} vs {base:.0f})"
            return f"CONNECT {s.id}"
        return None

    def cloud_risky(self, s):
        op = self.op
        return op.target == DIA and op.eta == 0 and len(self.opp) < 3 and self.diag_by.get(s.id) == 1

    def opp_usable(self, s):
        """Value of a sample for the opponent if it picked it up from the cloud (0 if it cannot use it)."""
        op = self.op
        nd = need_of(s, op.ex)
        if not all(nd[t] <= self.total[t] - self.me.st[t] for t in R5) or sum(nd) > 10 or self.left < 15:
            return 0
        ext = sum(max(0, nd[t] - op.st[t]) for t in R5)
        return s.health * (GIFT2 if ext <= 2 else 1.0)

    # ---------------- MOLECULES
    def pick_type(self, extra):
        start = self.opp_start()
        av = [self.total[t] - self.me.st[t] - self.op.st[t] for t in R5]
        if PICK >= 2 and start < sum(extra):
            best, bk = None, None
            for t in R5:
                if extra[t] <= 0 or av[t] <= 0:
                    continue
                short = self.race_sim(extra, t, start)
                on = self.o_extra[t]
                k = (-short,) + pkey(on, av[t] - extra[t] - on, extra[t], av[t])
                if bk is None or k > bk:
                    best, bk = t, k
            return best
        return self.race_pick(extra, self.o_extra, av, start < sum(extra))

    def race_sim(self, extra, first, start):
        """Simulates simultaneous taking vs. the modelled opponent (its plan first, then denial if it has
        denied before); returns how many of my `extra` molecules I fail to collect."""
        av = [self.total[t] - self.me.st[t] - self.op.st[t] for t in R5]
        rem, orem = list(extra), list(self.o_extra)
        ocap = 10 - sum(self.op.st)
        deny = self.o_deny > 0
        room = 10 - sum(self.me.st)
        for i in range(sum(extra) + 6):
            if sum(rem) == 0 or room <= 0:
                break
            if i == 0:
                t = first
            else:
                t, bk = None, None
                act = i >= start
                for x in R5:
                    if rem[x] <= 0 or av[x] <= 0:
                        continue
                    on = orem[x] if act else 0
                    k = pkey(on, av[x] - rem[x] - on, rem[x], av[x])
                    if bk is None or k > bk:
                        t, bk = x, k
            u = None
            if i >= start and ocap > 0:
                cand = [x for x in R5 if orem[x] > 0 and av[x] > 0]
                if cand:
                    u = min(cand, key=lambda x: (av[x] - orem[x], av[x]))
                elif deny:
                    cand = [x for x in R5 if rem[x] > 0 and av[x] > 0 and av[x] - rem[x] + 1 <= min(3, ocap)]
                    if cand:
                        u = min(cand, key=lambda x: av[x] - rem[x])
            if t is None and u is None:
                break
            if t is not None and t == u:
                av[t] -= 2
                rem[t] -= 1
                room -= 1
                orem[t] = max(0, orem[t] - 1)
                ocap -= 1
                continue
            if t is not None:
                av[t] -= 1
                rem[t] -= 1
                room -= 1
            if u is not None:
                av[u] -= 1
                orem[u] = max(0, orem[u] - 1)
                ocap -= 1
        return sum(rem)

    def mol_plan(self, diag):
        me, left, av = self.me, self.left, self.av
        plans = all_plans(diag, me.st, me.ex, av, left, MOL, self.sval)
        if not plans:
            return None
        if RACE > 0 and self.opp_start() < plans[0].m:
            for p in plans[:6]:
                if self.race_ok(p.extra) if RACE < 2 else self.race_ok3(p.extra):
                    if p is not plans[0]:
                        self.why_race = f" race-alt(best {plans[0].order} fails)"
                    return p
        return plans[0]

    def at_molecules(self):
        me, left, av = self.me, self.left, self.av
        diag = [s for s in self.mine if s.diag]
        self.why_race = ""
        plan = self.mol_plan(diag) if diag else None
        if plan and plan.m > 0:
            t = self.pick_type(plan.extra)
            self.waits = 0
            self.took_here = True
            self.why = f"plan{plan.order} x{vec(plan.extra)}{self.why_race}"
            return f"CONNECT {TY[t]}"
        if plan:
            self.waits = 0
            extra = self.extra_take(plan, diag)
            if extra is not None:
                self.took_here = True
                return extra
            return self.goto(LAB, f"plan{plan.order} ready")
        # nothing completable now: wait for predicted releases?
        if diag and self.waits < 15:
            for delay in range(1, int(WAITMAX) + 1):
                av2 = self.pred_av(delay)
                p2 = best_plan(diag, me.st, me.ex, av2, left - delay, MOL, self.sval)
                if p2:
                    self.waits += 1
                    for t in sorted(R5, key=lambda x: av[x]):
                        if p2.extra[t] > 0 and av[t] > 0 and sum(me.st) < 10:
                            self.took_here = True
                            self.why = f"prewait{p2.order} d{delay}"
                            return f"CONNECT {TY[t]}"
                    self.why = f"wait{p2.order} d{delay}"
                    return "WAIT"
        self.waits = 0
        if not diag:
            if UNDIAG > 0 and self.mine and left > 8:
                return self.goto(DIA, "undiagnosed")
            if left > 20:
                return self.goto(SAM, "no samples")
            return self.endgame_move("mol idle")
        if left < 12:
            return self.endgame_move("stuck end")
        if len(self.mine) < 3 and not self.good_cloud() and (left >= REFT or self.endgame_batch(MOL) is not None):
            return self.goto(SAM, "stuck->samples")
        return self.goto(DIA, "stuck->swap")

    def extra_take(self, plan, diag):
        """After the plan is covered: pre-take molecules for remaining samples, or deny the opponent."""
        me, av, left = self.me, self.av, self.left
        room = 10 - sum(me.st)
        if room <= 0:
            return None
        st2 = [me.st[t] - plan.need[t] for t in R5]
        e2 = list(me.ex)
        for s in plan.order:
            e2[s.gain] += 1
        rest = [s for s in diag if s not in plan.order]
        slack = left - plan.time
        useful = [0] * 5
        sup = self.supply()
        if rest and slack > EXSL:
            best, bv = None, -1e9
            for s in rest:
                nd = need_of(s, e2)
                ext = [max(0, nd[t] - st2[t]) for t in R5]
                if any(nd[t] > sup[t] for t in R5) or sum(st2) + sum(ext) > 10:
                    continue
                v = self.sval(s)
                if v > bv:
                    best, bv = ext, v
            if best is not None:
                useful = best
                if DPRE > 0:
                    d = self.deny_type(room, slack, useful)
                    if d is not None:
                        return f"CONNECT {TY[d]}"
                for t in sorted(R5, key=lambda t: av[t]):
                    if best[t] > 0 and av[t] > 0:
                        self.why = f"pretake {TY[t]} for later"
                        return f"CONNECT {TY[t]}"
        d = self.deny_type(room, slack, useful)
        if d is not None:
            return f"CONNECT {TY[d]}"
        return None

    def opp_value(self, av):
        op = self.op
        p = best_plan(self.o_diag, op.st, op.ex, av, self.left - op.eta, op.target if op.target != START else SAM,
                      self.oval)
        return sum(self.oval(s) for s in p.order) if p else 0.0

    def deny_type(self, room, slack, useful):
        """Returns a molecule type to take purely to block the opponent's best plan, or None."""
        op, av = self.op, self.av
        if slack < 2 or room <= 0 or not self.o_plan or self.o_plan.m == 0:
            return None
        if op.target not in (MOL, DIA) and not (DLAB > 0 and op.target == LAB):
            return None
        base = self.opp_value(av)
        best, bv = None, 0.0
        end = self.left < ENDT
        for t in R5:
            if self.o_extra[t] <= 0 or av[t] <= 0:
                continue
            k = av[t] - self.o_extra[t] + 1
            if k > room or k > (DENYKE if end else DK) or k > slack:
                continue
            av2 = list(av)
            av2[t] -= k
            v2 = self.opp_value(av2)
            junk = max(0, k - useful[t])
            if end:
                gain = (base - v2) - k * ENDKC
            else:
                gain = DENYF * (base - v2) - k * RATE - junk * JUNK + (DFULL if v2 <= 0 else 0.0)
            if gain > bv:
                best, bv = t, gain
        if best is not None:
            self.why = f"deny {TY[best]} ox{vec(self.o_extra)} g{bv:.0f}"
        return best

    def good_cloud(self):
        sup = self.supply()
        for s in self.cloud:
            if self.dumped.get(s.id) == sum(self.me.ex):
                continue
            nd = need_of(s, self.me.ex)
            if all(nd[t] <= min(sup[t], self.av[t] + self.me.st[t]) for t in R5) and sum(nd) <= 10 and s.health >= CLOUDH:
                return True
        return False

    # ---------------- LABORATORY
    def at_lab(self):
        me, left = self.me, self.left
        diag = [s for s in self.mine if s.diag]
        plan = best_plan(diag, me.st, me.ex, [0] * 5, left, LAB, self.sval) if diag else None
        if plan:
            if HOLD > 0 and left <= min(ENDH, HOLDH):
                h = self.hold_research(plan)
                if h is not None:
                    return h
            s = plan.order[0]
            self.why = f"research {s.id} order{plan.order}"
            return f"CONNECT {s.id}"
        n = len(self.mine)
        und = n - len(diag)
        if und and left > 4 + und + 3 + 3 + 3 + 2:
            return self.goto(DIA, "diagnose held")
        av = self.pred_av(3)
        rp = best_plan(diag, me.st, me.ex, av, left, LAB, self.sval) if diag else None
        free = 3 - n
        refill = free > 0 and (left >= REFT or self.endgame_batch(LAB) is not None)
        if rp:
            if not refill:
                return self.goto(MOL, f"finish {rp.order}")
            joint = len(rp.order) == len(diag)
            r = self.choose_rank() if left >= REFT else self.endgame_batch(LAB)[1]
            en = rank_stats(r, me.ex)[0]
            if joint and sum(me.st) + rp.m + free * en <= 10 + REFSLACK:
                if self.good_cloud() and n >= 1:
                    return self.goto(DIA, f"cloud (held {rp.order} m{rp.m})")
                return self.goto(SAM, f"refill (held {rp.order} m{rp.m})")
            return self.goto(MOL, f"finish {rp.order} first")
        if refill:
            if self.good_cloud() and n >= 1:
                return self.goto(DIA, "cloud")
            return self.goto(SAM, "refill")
        if diag and left > 12:
            for delay in range(3, int(WAITMAX) + 3):
                if best_plan(diag, me.st, me.ex, self.pred_av(delay), left - delay, LAB, self.sval):
                    return self.goto(MOL, "wait release")
            return self.goto(DIA, "blocked->swap")
        return self.endgame_move("lab idle")

    def hold_research(self, plan):
        """Endgame: waits at LAB instead of researching when the released molecules would let the opponent
        complete samples it is blocked on; researches late enough that it cannot use them. Returns WAIT or None."""
        me, op, left = self.me, self.op, self.left
        n = len(plan.order)
        if left <= max(n, 4) or not self.o_diag:
            return None
        rel = [0] * 5
        e = list(me.ex)
        for s in plan.order:
            nd = need_of(s, e)
            for t in R5:
                rel[t] += nd[t]
            e[s.gain] += 1
        if any(self.done_proj(p, e) and not self.done_proj(p, me.ex) for p in self.open_proj):
            return None
        t0 = self.opp_mol_start()
        v0 = self.opp_end_value(op.st, self.av, t0)[0]
        v1 = self.opp_end_value(op.st, [self.av[t] + rel[t] for t in R5], t0)[0]
        if v1 - v0 < 5:
            return None
        # my own use of the remaining time after researching now
        rest = [s for s in self.mine if s not in plan.order]
        if any(not s.diag for s in rest):
            return None
        mine_v = 0.0
        if rest:
            st2 = [me.st[t] - rel[t] for t in R5]
            rp = best_plan(rest, st2, e, [self.av[t] + rel[t] for t in R5], left - n, LAB, self.sval)
            mine_v = sum(s.health for s in rp.order) if rp else 0.0
        if mine_v == 0 and len(rest) < 3:
            eb = self.endgame_batch(LAB)
            if eb is not None:
                k, r = eb
                mine_v = k * (HEALTH[r] + 1) * rank_stats(r, e)[1]
        if mine_v >= v1 - v0:
            return None
        self.why = f"hold research {plan.order}: release unblocks opp {v1 - v0:.0f} (mine {mine_v:.0f})"
        return "WAIT"

    # ---------------- endgame
    def endgame_move(self, why):
        """Nothing productive left: try to block the opponent's remaining completions."""
        me = self.me
        if self.o_plan is not None and self.o_plan.m > 0 and sum(me.st) < 10:
            if me.target == MOL:
                for t in sorted(R5, key=lambda x: self.av[x]):
                    if self.o_extra[t] > 0 and self.av[t] > 0:
                        self.took_here = True
                        self.why = f"{why}: deny {TY[t]}"
                        return f"CONNECT {TY[t]}"
            elif self.left > 3:
                return self.goto(MOL, f"{why}: go deny")
        self.why = why
        return "WAIT"

    # ------------------------------------------------------------------ safety
    def valid(self, a):
        me = self.me
        parts = a.split()
        if parts[0] == "WAIT":
            return True
        if parts[0] == "GOTO":
            return me.eta == 0 and parts[1] in (SAM, DIA, MOL, LAB)
        if me.eta > 0 or len(parts) < 2:
            return False
        arg = parts[1]
        loc = me.target
        if loc == SAM:
            return arg in ("1", "2", "3") and len(self.mine) < 3
        if loc == DIA:
            sid = int(arg)
            s = next((x for x in self.samples if x.id == sid), None)
            if s is None:
                return False
            return s.carrier == 0 or (s.carrier == -1 and len(self.mine) < 3)
        if loc == MOL:
            return arg in TY and self.av[TY.index(arg)] > 0 and sum(me.st) < 10
        if loc == LAB:
            sid = int(arg)
            s = next((x for x in self.mine if x.id == sid), None)
            return s is not None and s.diag and all(me.st[t] >= s.cost[t] - me.ex[t] for t in R5)
        return False

    def log(self, a, ms):
        me, op = self.me, self.op
        ms_ = " ".join(f"{s.id}:{s.rank}{TY[s.gain] if s.gain >= 0 else '?'}"
                       f"{str(s.health) + '/' + vec(need_of(s, me.ex)) if s.diag else ''}" for s in self.mine)
        os_ = " ".join(f"{s.id}:{s.rank}{'/' + vec(need_of(s, op.ex)) if s.diag else '?'}" for s in self.opp)
        print(f"T{self.turn} {me.target[:3]}{me.eta} {me.score} st{vec(me.st)} ex{vec(me.ex)} [{ms_}] | "
              f"op {op.target[:3]}{op.eta} {op.score} st{vec(op.st)} ex{vec(op.ex)} [{os_}] | av{vec(self.av)} "
              f"c{len(self.cloud)}{' D' + str(self.o_deny) if self.o_deny else ''}", file=sys.stderr)
        print(f"  -> {a} ({self.why}) {ms:.1f}ms", file=sys.stderr, flush=True)


def main():
    bot = Bot()
    while True:
        try:
            bot.read()
        except (EOFError, ValueError):
            break
        t0 = time.perf_counter()
        try:
            a = bot.act()
            if not bot.valid(a):
                bot.why += " !INVALID " + a
                a = "WAIT"
        except Exception as e:  # never crash: fall back to WAIT
            bot.why = f"EXC {type(e).__name__}: {e}"
            a = "WAIT"
        bot.log(a, (time.perf_counter() - t0) * 1000)
        print(a, flush=True)
        bot.turn += 1


main()
