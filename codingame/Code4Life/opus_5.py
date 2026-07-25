"""Code4Life bot: collects samples, diagnoses them, gathers molecules and produces medicine."""
import os
import sys
import traceback
from collections import namedtuple
from itertools import permutations

TYPES = "ABCDE"
SAMPLES = "SAMPLES"
DIAGNOSIS = "DIAGNOSIS"
MOLECULES = "MOLECULES"
LABORATORY = "LABORATORY"
START = "START_POS"
DIST = {
    START: {SAMPLES: 2, DIAGNOSIS: 2, MOLECULES: 2, LABORATORY: 2},
    SAMPLES: {SAMPLES: 0, DIAGNOSIS: 3, MOLECULES: 3, LABORATORY: 3},
    DIAGNOSIS: {SAMPLES: 3, DIAGNOSIS: 0, MOLECULES: 3, LABORATORY: 4},
    MOLECULES: {SAMPLES: 3, DIAGNOSIS: 3, MOLECULES: 0, LABORATORY: 3},
    LABORATORY: {SAMPLES: 3, DIAGNOSIS: 4, MOLECULES: 3, LABORATORY: 0},
}
MAX_MOL = 10
MAX_SAMP = 3
MAX_TURNS = 200
STOCK = 5
R = range(5)
PRIOR3 = [[0, 0, 4, 3, 0], [3, 0, 0, 4, 3], [0, 4, 4, 3, 3], [2, 3, 3, 0, 3],
          [0, 0, 0, 7, 3], [5, 3, 0, 3, 3]]  # stand-in rank 3 costs until the game shows us real ones


def cfg(key, default):
    return float(os.environ.get(key, default))


TURN_COST = cfg("C4L_TURN_COST", 2.0)          # health-equivalent value of one turn
EXP_VALUE = cfg("C4L_EXP_VALUE", 4.0)          # value of one point of molecule expertise
PROJ_VALUE = cfg("C4L_PROJ_VALUE", 50.0)       # health awarded by a science project
PREFETCH_TURNS = cfg("C4L_PREFETCH", 25)       # only pre-gather molecules while this much time is left
RANKS = os.environ.get("C4L_RANKS", "111,111,111,331,333").split(",")  # rank per (expertise // 3, held)
FETCH_AT = cfg("C4L_FETCH_AT", 0)              # most samples in hand that still justifies a sample run
RIDE = cfg("C4L_RIDE", 1)                      # leftovers worth less than this share of a molecule
                                               # round trip ride along with the next batch instead
LAST_FETCH = cfg("C4L_LAST_FETCH", 15)         # stop collecting new samples below this many turns
STALL = cfg("C4L_STALL", 10)                   # turns stuck at MOLECULES before giving up on a sample
DRY = cfg("C4L_DRY", 25)                       # turns without producing before ditching blocked samples
HOSTILE = cfg("C4L_HOSTILE", 30)               # turns without a rival medicine before writing off their molecules
DENY = cfg("C4L_DENY", 1)                      # take scarce molecules the rival still needs
DENY_LEFT = cfg("C4L_DENY_LEFT", 4)            # ... while at most this many of the type are on offer
DENY_TURNS = cfg("C4L_DENY_TURNS", 10)         # ... and at least this many turns remain
SPARE = cfg("C4L_SPARE", 1)                    # molecule slots kept free when denying
HARVEST = cfg("C4L_HARVEST", 90)               # below this many turns left, draw rank 3 for the health
READY = cfg("C4L_READY", 0)                    # share of seen rank 3 samples our expertise must cover
END_TURNS = cfg("C4L_END_TURNS", 18)           # below this, request the cheapest samples
END_RANK = cfg("C4L_END_RANK", 1)
KEEP_GAP = cfg("C4L_KEEP_GAP", 2)              # expertise gap at which an unbuildable sample is ditched
R3_MIN = cfg("C4L_R3_MIN", 1)                  # expertise in every type before rank 3 is worth drawing
R3_FALL = cfg("C4L_R3_FALL", 2)                # rank drawn instead when rank 3 is not worth it yet

Plan = namedtuple("Plan", "seq short score")
EMPTY = Plan((), (0, 0, 0, 0, 0), 0.0)
STUCK = "WAIT stuck"


class Sample:
    def __init__(self, parts):
        self.id = int(parts[0])
        self.carried = int(parts[1])
        self.rank = int(parts[2])
        self.health = int(parts[4])
        self.cost = [int(x) for x in parts[5:10]]
        self.diagnosed = self.health >= 0
        self.gain = TYPES.index(parts[3]) if parts[3] in TYPES else 0

    def __repr__(self):
        return "#%d/r%d/h%d/%s/%s" % (self.id, self.rank, self.health, TYPES[self.gain],
                                      "".join(str(c) for c in self.cost))


class Robot:
    def __init__(self, parts):
        self.target = parts[0]
        self.eta = int(parts[1])
        self.score = int(parts[2])
        self.storage = [int(x) for x in parts[3:8]]
        self.expertise = [int(x) for x in parts[8:13]]


class Bot:
    def __init__(self, projects):
        self.projects = projects
        self.turn = 0
        self.blocked = 0
        self.dumped = set()
        self.seen = {}
        self.opp_score = 0
        self.opp_moved = 0
        self.my_score = 0
        self.my_moved = 0

    # ------------------------------------------------------------------ state
    def update(self, me, opp, avail, samples):
        self.turn += 1
        self.turns_left = MAX_TURNS - self.turn + 1
        self.me = me
        self.opp = opp
        self.avail = [max(0, a) for a in avail]
        self.mine = [s for s in samples if s.carried == 0]
        self.cloud = [s for s in samples if s.carried == -1 and s.id not in self.dumped]
        self.theirs = [s for s in samples if s.carried == 1 and s.diagnosed]
        self.diag = [s for s in self.mine if s.diagnosed]
        self.undiag = [s for s in self.mine if not s.diagnosed]
        self.pos = me.target if me.target in DIST else START
        for s in samples:
            if s.diagnosed:
                self.seen[s.id] = s
        self.turn_cost = TURN_COST if self.turns_left > 30 else 0.3
        self.live = [p for p in self.projects if not all(opp.expertise[t] >= p[t] for t in R)]
        if opp.score > self.opp_score:
            self.opp_score, self.opp_moved = opp.score, self.turn
        if me.score > self.my_score:
            self.my_score, self.my_moved = me.score, self.turn
        self.dry = self.turn - self.my_moved
        active = self.turn - self.opp_moved < HOSTILE
        self.reach = [me.storage[t] + self.avail[t] + (opp.storage[t] if active else 0) for t in R]
        self.plan = self.best_plan(True)
        self.ideal = self.best_plan(False)

    # ------------------------------------------------------------- evaluation
    def seq_req(self, seq):
        """Molecules needed to produce seq in order, accounting for expertise gained along the way."""
        exp = list(self.me.expertise)
        tot = [0] * 5
        for s in seq:
            for t in R:
                tot[t] += max(0, s.cost[t] - exp[t])
            exp[s.gain] += 1
        return tot

    def gain_value(self, gain, exp):
        """Health-equivalent worth of gaining one point of expertise of type gain."""
        v = EXP_VALUE * min(1.0, self.turns_left / 60.0)
        for p in self.live:
            rem = sum(max(0, p[t] - exp[t]) for t in R)
            if rem and p[gain] > exp[gain]:
                v += PROJ_VALUE / rem ** 1.6
        return v

    def plan_value(self, seq):
        exp = list(self.me.expertise)
        v = 0.0
        for s in seq:
            v += s.health + self.gain_value(s.gain, exp)
            exp[s.gain] += 1
        return v

    def net_val(self, s):
        req = sum(max(0, s.cost[t] - self.me.expertise[t]) for t in R)
        return s.health + self.gain_value(s.gain, self.me.expertise) - self.turn_cost * req

    def best_plan(self, respect_avail):
        """Best subset+order of carried diagnosed samples deliverable in one trip to the lab."""
        best = EMPTY
        st = self.me.storage
        for r in range(1, len(self.diag) + 1):
            for seq in permutations(self.diag, r):
                req = self.seq_req(seq)
                if sum(max(st[t], req[t]) for t in R) > MAX_MOL or any(req[t] > self.reach[t] for t in R):
                    continue
                short = [max(0, req[t] - st[t]) for t in R]
                if respect_avail and any(short[t] > self.avail[t] for t in R):
                    continue
                tot = sum(short)
                if tot:
                    turns = DIST[self.pos][MOLECULES] + tot + DIST[MOLECULES][LABORATORY]
                else:
                    turns = DIST[self.pos][LABORATORY]
                if turns + r > self.turns_left:
                    continue
                score = self.plan_value(seq) - self.turn_cost * (tot + r)
                if score > best.score:
                    best = Plan(seq, short, score)
        return best

    def producible(self):
        """Longest/best sequence of carried samples that can be produced with the molecules in hand."""
        best, key = (), None
        limit = max(1, min(len(self.diag), self.turns_left))
        for r in range(1, limit + 1):
            for seq in permutations(self.diag, r):
                req = self.seq_req(seq)
                if any(req[t] > self.me.storage[t] for t in R):
                    continue
                k = (self.plan_value(seq[:self.turns_left]), -sum(req), seq[0].health)
                if key is None or k > key:
                    best, key = seq, k
        return best

    # ---------------------------------------------------------------- choices
    def cloud_good(self):
        """Cloud samples we could still finish, best first, as (value, sample) pairs."""
        out = [(self.net_val(s), s) for s in self.cloud if not self.hopeless(s)]
        out.sort(key=lambda x: -x[0])
        return [x for x in out if x[0] > 0]

    def want_samples(self):
        """True when a run to the samples module pays off.

        Finishing leftovers needs a dedicated molecule round trip, so cheap leftovers are better
        carried along with the next batch than delivered on their own."""
        if len(self.mine) >= MAX_SAMP or self.turns_left < LAST_FETCH:
            return False
        if len(self.mine) <= FETCH_AT or not self.plan.seq:
            return True
        detour = DIST[LABORATORY][MOLECULES] + DIST[MOLECULES][LABORATORY]
        return self.plan.score < RIDE * detour * self.turn_cost

    def pick_rank(self):
        """Rank to request: expertise first while it still has time to pay off, then health.

        Expertise only earns its keep through the samples it discounts later, so once too little
        of the game is left for that, we switch to drawing the richest samples we can afford -
        but only while our expertise can actually build them."""
        if self.turns_left < END_TURNS:
            return int(END_RANK)
        if self.turns_left < HARVEST:
            return 3 if self.r3_ready() else int(R3_FALL)
        rank = int(RANKS[min(len(RANKS) - 1, sum(self.me.expertise) // 3)][min(2, len(self.mine))])
        if rank < 3:
            return rank
        return 3 if min(self.me.expertise) >= R3_MIN and self.r3_ready() else int(R3_FALL)

    def r3_ready(self):
        """True when enough of the rank 3 samples seen this game are within our expertise.

        Rank 3 costs reach 7 of a single type against a stock of 5, so a draw is dead unless the
        matching expertise is banked. Both robots' diagnosed samples are visible, which gives a
        live sample of the deck to judge that against instead of guessing."""
        if not READY:
            return True
        costs = [s.cost for s in self.seen.values() if s.rank == 3]
        if len(costs) < 4:
            costs = costs + PRIOR3
        ok = sum(1 for c in costs if self.buildable(c))
        return ok >= READY * len(costs)

    def buildable(self, cost):
        req = [max(0, cost[t] - self.me.expertise[t]) for t in R]
        return sum(req) <= MAX_MOL and all(req[t] <= STOCK for t in R)

    def gap(self, s):
        """Expertise points still missing before this sample could ever be produced.

        Costs run up to 7 of a single type while only 5 exist, so some samples are dead weight
        until enough expertise of that type is banked - each point cuts the requirement by one.
        Measured against what we can actually reach, which also shrinks when a rival sits on
        molecules and stops producing: that is what lets us ditch samples we can never finish."""
        req = [max(0, s.cost[t] - self.me.expertise[t]) for t in R]
        return max(sum(req) - MAX_MOL, max(req[t] - self.reach[t] for t in R), 0)

    def hopeless(self, s):
        return self.gap(s) > 0

    def pick_dump(self):
        """Diagnosed sample worth pushing back to the cloud, if any."""
        if self.turns_left < 22 or not self.diag:
            return None
        for s in self.diag:
            gap = self.gap(s)
            if gap >= KEEP_GAP or (gap and (s.health <= 30 or self.turns_left < 60)):
                return s
        if len(self.mine) < MAX_SAMP:
            return None
        worst = min(self.diag, key=self.net_val)
        good = self.cloud_good()
        if good and good[0][0] > self.net_val(worst) + 8:
            return worst
        if (self.blocked >= STALL or self.dry >= DRY) and not self.plan.seq:
            stuck = max(self.diag, key=self.unreachable)
            return stuck if self.unreachable(stuck) else None
        return None

    def unreachable(self, s):
        """Molecules of a sample that are neither carried nor available right now."""
        return sum(max(0, s.cost[t] - self.me.expertise[t] - self.me.storage[t] - self.avail[t]) for t in R)

    def deny(self):
        """Scarce molecule the rival still needs, when we can afford to sit on it.

        Points only decide a game through who has more of them, so starving a rival of a type
        they are short of is worth a turn once our own batch is already covered."""
        held = sum(self.me.storage)
        if not DENY or held >= MAX_MOL or held > MAX_MOL - SPARE or self.turns_left < DENY_TURNS:
            return None
        need = [0] * 5
        for s in self.theirs:
            for t in R:
                need[t] += max(0, s.cost[t] - self.opp.expertise[t])
        cand = [t for t in R if 0 < self.avail[t] <= DENY_LEFT and need[t] > self.opp.storage[t]]
        return max(cand, key=lambda t: need[t]) if cand else None

    def pick_cloud(self):
        """Diagnosed sample worth collecting from the cloud - a turn cheaper than a fresh draw."""
        if len(self.mine) >= MAX_SAMP or self.turns_left < 15:
            return None
        good = self.cloud_good()
        return good[0][1] if good and good[0][0] >= 6 else None

    def gather_order(self):
        """Carried samples ordered by how urgently their molecules should be collected."""
        order = list(self.plan.seq)
        order += [s for s in self.ideal.seq if s not in order]
        order += sorted((s for s in self.diag if s not in order), key=self.net_val, reverse=True)
        return order

    def pick_molecule(self):
        """Molecule type to fetch next, or None when there is nothing useful to take."""
        st = self.me.storage
        if sum(st) >= MAX_MOL:
            return None
        pool = list(st)
        exp = list(self.me.expertise)
        cum = 0
        for i, s in enumerate(self.gather_order()):
            if i >= len(self.plan.seq) and self.plan.seq and self.turns_left < PREFETCH_TURNS:
                break
            need = [max(0, s.cost[t] - exp[t]) for t in R]
            if cum + sum(need) > MAX_MOL or any(need[t] > self.reach[t] for t in R):
                break
            cum += sum(need)
            for t in R:
                d = min(pool[t], need[t])
                pool[t] -= d
                need[t] -= d
            cand = [t for t in R if need[t] > 0 and self.avail[t] > 0]
            if cand:
                return min(cand, key=lambda t: (self.avail[t], -need[t]))
            exp[s.gain] += 1
        return None

    def next_module(self):
        if self.undiag:
            return DIAGNOSIS
        if self.want_samples():
            good = self.cloud_good()
            if len(good) >= min(MAX_SAMP - len(self.mine), 2):
                return DIAGNOSIS
            return SAMPLES
        if sum(self.plan.short) > 0:
            return MOLECULES
        if self.plan.seq or self.producible():
            return LABORATORY
        if self.pick_dump() is not None:
            return DIAGNOSIS
        if self.diag:
            return MOLECULES
        return SAMPLES if self.turns_left > LAST_FETCH else LABORATORY

    # ------------------------------------------------------------------ turn
    def act(self):
        cmd = self.decide()
        if cmd == STUCK:
            self.blocked += 1
        elif not cmd.startswith("WAIT"):
            self.blocked = 0
        return cmd

    def decide(self):
        if self.me.eta > 0:
            return "WAIT"
        if self.pos == LABORATORY:
            seq = self.producible()
            if seq:
                return "CONNECT %d" % seq[0].id
        if self.pos == MOLECULES:
            t = self.pick_molecule()
            if t is not None:
                return "CONNECT " + TYPES[t]
            if not sum(self.plan.short):
                t = self.deny()
                if t is not None:
                    return "CONNECT " + TYPES[t]
        if self.pos == DIAGNOSIS:
            if self.undiag:
                return "CONNECT %d" % self.undiag[0].id
            drop = self.pick_dump()
            if drop is not None:
                self.dumped.add(drop.id)
                return "CONNECT %d" % drop.id
            take = self.pick_cloud()
            if take is not None:
                return "CONNECT %d" % take.id
        if self.pos == SAMPLES and self.want_samples():
            return "CONNECT %d" % self.pick_rank()
        nxt = self.next_module()
        return STUCK if nxt == self.pos else "GOTO " + nxt

    def fallback(self):
        """Simple safe policy used if the main logic raises."""
        if self.me.eta > 0:
            return "WAIT"
        if self.pos == LABORATORY:
            for s in self.diag:
                if all(self.me.storage[t] >= max(0, s.cost[t] - self.me.expertise[t]) for t in R):
                    return "CONNECT %d" % s.id
            return "GOTO " + (DIAGNOSIS if self.undiag or self.mine else SAMPLES)
        if self.pos == SAMPLES:
            return "CONNECT 2" if len(self.mine) < MAX_SAMP else "GOTO " + DIAGNOSIS
        if self.pos == DIAGNOSIS:
            if self.undiag:
                return "CONNECT %d" % self.undiag[0].id
            return "GOTO " + MOLECULES
        if self.pos == MOLECULES:
            for s in self.diag:
                for t in R:
                    if self.me.storage[t] < max(0, s.cost[t] - self.me.expertise[t]) and self.avail[t] > 0 \
                            and sum(self.me.storage) < MAX_MOL:
                        return "CONNECT " + TYPES[t]
            return "GOTO " + LABORATORY
        return "GOTO " + SAMPLES

    def dbg(self, cmd):
        head = ""
        if self.turn == 1:
            head = "PROJECTS %s\n" % " ".join("".join(str(x) for x in p) for p in self.projects)
        return "%sT%d %s st%s ex%s rx%s av%s %d-%d hold%s plan%s short%s -> %s" % (
            head, self.turn, self.pos[:4], "".join(str(x) for x in self.me.storage),
            "".join(str(x) for x in self.me.expertise), "".join(str(x) for x in self.opp.expertise),
            "".join(str(x) for x in self.avail), self.me.score, self.opp.score,
            self.mine, [s.id for s in self.plan.seq], "".join(str(x) for x in self.plan.short), cmd)


def main():
    count = int(input())
    projects = [[int(x) for x in input().split()] for _ in range(count)]
    bot = Bot(projects)
    while True:
        me = Robot(input().split())
        opp = Robot(input().split())
        avail = [int(x) for x in input().split()]
        samples = [Sample(input().split()) for _ in range(int(input()))]
        try:
            bot.update(me, opp, avail, samples)
            cmd = bot.act()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                cmd = bot.fallback()
            except Exception:
                traceback.print_exc(file=sys.stderr)
                cmd = "WAIT"
        print(cmd)
        try:
            print(bot.dbg(cmd), file=sys.stderr)
        except Exception:
            pass


main()
