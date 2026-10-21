import sys
import random
from itertools import permutations, product
from math import gcd
import time
import heapq
import itertools
import numpy as np

Z = 30
A = 27
ALPHA = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STEP_LIMIT = 4000
STEP_RESERVE = 1  # steps reserved per remaining character when pruning step-heavy states

_d = np.abs(np.arange(Z)[:, None] - np.arange(Z)[None, :])
MD = np.minimum(_d, Z - _d).astype(np.int32)
_c = (np.arange(A)[None, :] - np.arange(A)[:, None]) % A
CD = np.minimum(_c, A - _c).astype(np.int32)
# change cost with optional reset ([-] or [+]) first
RESET = np.zeros((A, A), dtype=np.int32)
CC = CD.copy()
for _v in range(1, A):
    for _t in range(A):
        if 3 + CD[0, _t] < CD[_v, _t]:
            CC[_v, _t] = 3 + CD[0, _t]
            RESET[_v, _t] = 1

# executed steps of change_code(v, t): [-] from v takes 3v steps, [+] takes 3(27-v)
CS = np.where(RESET == 1, np.where(np.arange(A)[:, None] <= 13, 3 * np.arange(A)[:, None], 3 * (A - np.arange(A)[:, None])) + CD[0][None, :], CD).astype(np.int32)
CSL = CS.tolist()

AR = np.arange(Z)
CDL = CD.tolist()


def move_code(p: int, z: int) -> str:
    d = (z - p) % Z
    return ">" * d if d <= Z - d else "<" * (Z - d)


def delta_code(v: int, t: int) -> str:
    d = (t - v) % A
    return "+" * d if d <= A - d else "-" * (A - d)


def change_code(v: int, t: int) -> str:
    if RESET[v, t]:
        return ("[-]" if v <= 13 else "[+]") + delta_code(0, t)
    return delta_code(v, t)


def move_code_jump(mem, p: int, z: int) -> str:
    """Move code from p to z matching MT/MS: plain moves, or [>]/[<] (preceded by one step when p is zero) then plain moves."""
    mem = [int(x) for x in mem]
    best = move_code(p, z)
    for s, ch, jc in ((1, ">", "[>]"), (-1, "<", "[<]")):
        q, pre = (p, "") if mem[p] else ((p + s) % Z, ch)
        if not mem[q]:
            continue
        r = (q + s) % Z
        while mem[r] and r != q:
            r = (r + s) % Z
        if r == q:
            continue
        c = pre + jc + move_code(r, z)
        if len(c) < len(best):
            best = c
    return best


def plain_code(mem, pos: int, z: int, t: int) -> str:
    return move_code_jump(mem, pos, z) + change_code(int(mem[z]), t) + "."


# plain-move model with [>] / [<] jumps: MT[(pos * 61 + rc) * 61 + lc, z] / MS[...] = length / executed steps of the
# move from pos to z chosen by move_code_jump, where rc / lc encode the [>] / [<] landing (offset k from pos, +30 when
# the pointer first steps off a zero rune; 60 = no jump)
CCT = np.ascontiguousarray(CC.T).astype(np.int8)  # [t, v]
MD8 = MD.astype(np.int8)
CST = np.ascontiguousarray(CS.T)
BIG = 1 << 20
ZB = np.int64(1) << AR.astype(np.int64)  # zero-mask bit of each zone
FULL = (1 << Z) - 1


def _move_tables() -> tuple:
    r = np.arange(61)
    off, pre = r % 30, r // 30
    jc = np.where(r < 60, 3 + pre, BIG)
    js = np.where(r < 60, pre + 3 * (off - pre), BIG)
    pl = np.broadcast_to(MD[0][None, None, :], (61, 61, Z))
    rt = np.broadcast_to((MD[off] + jc[:, None])[:, None, :], (61, 61, Z))
    lt = np.broadcast_to((MD[(-off) % Z] + jc[:, None])[None, :, :], (61, 61, Z))
    rs = np.broadcast_to((MD[off] + js[:, None])[:, None, :], (61, 61, Z))
    ls = np.broadcast_to((MD[(-off) % Z] + js[:, None])[None, :, :], (61, 61, Z))
    use_l = lt < np.minimum(pl, rt)
    use_r = ~use_l & (rt < pl)
    cost = np.minimum(pl, np.minimum(rt, lt))
    steps = np.where(use_l, ls, np.where(use_r, rs, pl))
    rel = (AR[None, :] - AR[:, None]) % Z
    mt = np.ascontiguousarray(cost[:, :, rel].transpose(2, 0, 1, 3).reshape(Z * 61 * 61, Z)).astype(np.int8)
    ms = np.ascontiguousarray(steps[:, :, rel].transpose(2, 0, 1, 3).reshape(Z * 61 * 61, Z)).astype(np.int16)
    return mt, ms


MT, MS = _move_tables()

# rotation/reflection-invariant state hash: H = min(S1 * w^-pos, S2 * w^pos) mod HQ with S1 = sum_j HW[mem[j]] w^j,
# S2 = sum_j HW[mem[j]] w^-j and w a primitive 30th root of unity mod the prime HQ; S1, S2 update in O(1) per rune change
HQ = 2147482951
_OM = 1699070896
PW = np.array([pow(_OM, j, HQ) for j in range(Z)], dtype=np.int64)
PWI = np.array([pow(_OM, (Z - j) % Z, HQ) for j in range(Z)], dtype=np.int64)
HW = np.random.default_rng(12345).integers(1, HQ, size=A).astype(np.int64)


def simulate(prog: str, mem=None, pos: int = 0, limit: int = STEP_LIMIT):
    """Runs prog from (mem, pos); returns (output, mem, pos, steps) or None if steps reach limit."""
    mem = [0] * Z if mem is None else list(mem)
    match, st = {}, []
    for i, ch in enumerate(prog):
        if ch == "[":
            st.append(i)
        elif ch == "]":
            j = st.pop()
            match[i], match[j] = j, i
    out, ip, steps, n = [], 0, 0, len(prog)
    while ip < n:
        steps += 1
        if steps >= limit:
            return None
        ch = prog[ip]
        if ch == "+":
            mem[pos] = (mem[pos] + 1) % A
        elif ch == "-":
            mem[pos] = (mem[pos] - 1) % A
        elif ch == ">":
            pos = (pos + 1) % Z
        elif ch == "<":
            pos = (pos - 1) % Z
        elif ch == ".":
            out.append(ALPHA[mem[pos]])
        elif ch == "[":
            if mem[pos] == 0:
                ip = match[ip]
        elif ch == "]":
            if mem[pos] != 0:
                ip = match[ip]
                continue
        ip += 1
    return "".join(out), mem, pos, steps


# ---------------------------------------------------------------- macros
# A macro generator is called as gen(tv, i, mem, pos) with tv the target values, i the phrase index,
# mem a list of 30 rune values and pos the current zone. It returns a list of (j, code, new_mem, new_pos[, steps])
# such that running code from (mem, pos) prints tv[i:j] and leaves the machine in (new_mem, new_pos); steps (the
# number of executed instructions) is optional: when it is missing the framework simulates the code to get it.

_LOOPC = {}


def _loop_counter(cur: int, it: int) -> tuple:
    """Cheapest (counter setup code, counter step code) for a loop running it times from counter value cur."""
    key = (cur, it)
    if key not in _LOOPC:
        best = None
        for d in range(1, 14):
            if d % 3 == 0:
                continue
            for s in (1, -1):
                cc = change_code(cur, (it * d * s) % A)
                if best is None or len(cc) + d < best[0]:
                    best = (len(cc) + d, cc, ("-" if s == 1 else "+") * d)
        _LOOPC[key] = best[1:]
    return _LOOPC[key]


# ---- repeat-loop macro
# A periodic stretch tv[i:i+m*p] (pattern P = tv[i:i+p]) is printed by "setup [body step]" (or a two-level
# nested loop for long runs).  The body is a cyclic walk over a few "slots" (zones) laid out on a line next
# to the counter; a slot may serve several tokens (letters, or "[.+]"-style progression runs ending at space)
# with in-body changes that are undone before the next iteration.  Layouts are optimized once per pattern
# by local search; per machine state every layout is tried at all 60 placements (zone x reflection) with
# exact setup costs (existing runes are reused) and the counter init is chosen per placement from CI.
LAY_KEEP = 6
LAY_ALT = 2          # extra layouts whose homes next to the counter are set by a multiplication loop plus fixup
ALT_PER_J = 2
ALT_SLACK = 1
AP_STEPS = (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6)
REUSE_BUDGET = 400
REPEAT_STEP_CAP = 3990          # main() lowers this and re-solves when the final program runs too long
LOOP_DS = (1, 2, 4, 5, 7, 8, 10, 11, 13)
CI = np.full((A, A), 10 ** 6, dtype=np.int64)  # CI[v, it]: counter init from value v + step length, it iterations
CI_V0 = np.zeros((A, A), dtype=np.int64)
CI_SD = np.zeros((A, A), dtype=np.int64)      # signed step: d > 0 means "-" * d per iteration, d < 0 means "+" * -d
for _v in range(A):
    for _it in range(1, A):
        for _d in LOOP_DS:
            for _s in (1, -1):
                _v0 = _s * _d * _it % A
                _cost = CC[_v, _v0] + _d
                if _cost < CI[_v, _it]:
                    CI[_v, _it], CI_V0[_v, _it], CI_SD[_v, _it] = _cost, _v0, _s * _d
CI_ABS = np.abs(CI_SD)
CI1 = CI[:, 1:]
# BH[a, b]: cheapest (fresh setup of home h) + (a -> h) + (h -> b); BHH the home
BH = np.zeros((A, A), dtype=np.int64)
BHH = np.zeros((A, A), dtype=np.int64)
for _a in range(A):
    for _b in range(A):
        _t = CD[0] + CD[_a] + CD[:, _b]
        BHH[_a, _b] = int(np.argmin(_t))
        BH[_a, _b] = _t[BHH[_a, _b]]
CDL, BHL, BHHL = CD.tolist(), BH.tolist(), BHH.tolist()
# BHM[v, a, b]: like BH but the home is set from current value v (memory-aware); BHMH the home
_t = CC[:, None, None, :] + CD[None, :, None, :] + CD.T[None, None, :, :]
BHMH = _t.argmin(3)
BHML, BHMHL = np.take_along_axis(_t, BHMH[..., None], 3)[..., 0].tolist(), BHMH.tolist()
del _t
# multiplication setup "A[>B<step]" run on a zero zone (the counter or an aux zone), then fixups of the targets:
# B1C[n, d] / B1V[n, d] cheapest |B| + fixup (and B) adding d to a zone in an n-iteration loop; MS1[d] total cost
# (counter init, brackets, moves, |B|, fixup) adding d to one neighbour, MS2[d1, d2] to both neighbours (or to two
# zones on one side); *N iteration count, *B multipliers, *X fixup costs, *S extra executed steps (beyond the code length)
B1C = np.full((A, A), 99, dtype=np.int64)
B1V = np.zeros((A, A), dtype=np.int64)
for _n in range(1, A):
    for _b in sorted(range(-13, 14), key=abs):
        if _b:
            _c = abs(_b) + CD[_n * _b % A]
            _f = _c < B1C[_n]
            B1C[_n][_f], B1V[_n][_f] = _c[_f], _b
MS1, MS1N = np.full(A, 99, dtype=np.int64), np.zeros(A, dtype=np.int64)
MS2, MS2N = np.full((A, A), 99, dtype=np.int64), np.zeros((A, A), dtype=np.int64)
for _n in range(1, A):
    _c1 = CI[0, _n] + B1C[_n] + 4
    _f = _c1 < MS1
    MS1[_f], MS1N[_f] = _c1[_f], _n
    _c2 = CI[0, _n] + B1C[_n][:, None] + B1C[_n][None, :] + 6
    _f = _c2 < MS2
    MS2[_f], MS2N[_f] = _c2[_f], _n
MS1B = B1V[MS1N, np.arange(A)]
MS1X = CD[MS1N * MS1B % A, np.arange(A)]
MS2B1, MS2B2 = B1V[MS2N, np.arange(A)[:, None]], B1V[MS2N, np.arange(A)[None, :]]
MS2X1, MS2X2 = CD[MS2N * MS2B1 % A, np.arange(A)[:, None]], CD[MS2N * MS2B2 % A, np.arange(A)[None, :]]
MS1S = (MS1N - 1) * (MS1 - MS1X - (CI[0, MS1N] - CI_ABS[0, MS1N]))
MS2S = (MS2N - 1) * (MS2 - MS2X1 - MS2X2 - (CI[0, MS2N] - CI_ABS[0, MS2N]))
MS1[0] = 99
MS2[0, :] = 99
MS2[:, 0] = 99
# BH1: like BH for a slot next to the counter, whose home may be set by a multiplication loop (without fixup: MS1E;
# homes that need a fixup are left to setup_eval, so the layouts keep homes that the loop can hit exactly)
MS1E = np.full(A, 99, dtype=np.int64)
for _n in range(1, A):
    for _b in range(-13, 14):
        if _b and _n * _b % A:
            MS1E[_n * _b % A] = min(MS1E[_n * _b % A], CI[0, _n] + abs(_b) + 4)
_t = np.minimum(CD[0], MS1E)[None, None, :] + CD[:, None, :] + CD.T[None, :, :]
BH1HL = _t.argmin(2).tolist()
BH1L = _t.min(2).tolist()
_t = np.minimum(CD[0], MS1)[None, None, :] + CD[:, None, :] + CD.T[None, :, :]  # same with fixups (MS1)
BH1FHL = _t.argmin(2).tolist()
BH1FL = _t.min(2).tolist()
PL_C = np.concatenate([AR, AR])
PL_R = np.repeat(np.array([1, -1]), Z)
SWAP = str.maketrans("<>", "><")
AR26 = np.arange(26)


def step_code(sd: int) -> str:
    return "-" * sd if sd > 0 else "+" * -sd


def mvr(dx: int) -> str:
    return ">" * dx if dx > 0 else "<" * -dx


def tokenize(P: tuple) -> list:
    """Token lists for pattern P: all letters, and (if any) with progression runs '[.+k]' ending at space."""
    p = len(P)
    plain = [(("L", v), v, v, ".", 0) for v in P]  # key, value in, value out, code, extra steps
    res = [plain]
    toks, t, found = [], 0, False
    while t < p:
        best = None
        for k in AP_STEPS:
            e = t + 1
            while e < p and (P[e] - P[e - 1]) % A == k % A and (P[e - 1] + k) % A:
                e += 1
            if e - t >= 3 and P[t] and (P[e - 1] + k) % A == 0 and (best is None or e - t > best[1]):
                best = (k, e - t)
        if best:
            k, ln = best
            code = "[." + step_code(-k) + "]"
            toks.append((("P", P[t], k, ln), P[t], 0, code, (ln - 1) * len(code)))
            t += ln
            found = True
        else:
            toks.append(plain[t])
            t += 1
    if found:
        res.append(toks)
    return res


def layout_search(tin: list, tout: list, seq: list, max_evals: int, keep: int, restarts: int = 10, extra: tuple = (),
                  bh1: tuple = None) -> list:
    """Local search over slot layouts. A line is a tuple of entries: None (counter) or a sorted tuple of types.
    bh1: (cost, home) tables for slots next to the counter (default BH1L, BH1HL). Returns [(cost, line, homes)] best first."""
    bh1l, bh1hl = bh1 or (BH1L, BH1HL)
    k = len(tin)
    occ = [[] for _ in range(k)]
    for x, t in enumerate(seq):
        occ[t].append(x)
    gcache = {}

    def gcost(g):
        r = gcache.get(g)
        if r is None:
            idxs = occ[g[0]] if len(g) == 1 else sorted(x for t in g for x in occ[t])
            c = 0
            prev = tout[seq[idxs[0]]]
            for x in idxs[1:]:
                t = seq[x]
                c += CDL[prev][tin[t]]
                prev = tout[t]
            f = tin[seq[idxs[0]]]
            r = gcache[g] = (c + BHL[prev][f], BHHL[prev][f], c + bh1l[prev][f], bh1hl[prev][f])
        return r

    ecache = {}

    def ev(line):
        r = ecache.get(line)
        if r is not None:
            return r
        posT = [0] * k
        cpos = line.index(None)
        lo = hi = cpos
        tot = 0
        for x, g in enumerate(line):
            if g is None:
                continue
            c, h, c1, h1 = gcost(g)
            if x - cpos in (1, -1):
                c, h = c1, h1
            tot += c
            if h:
                if x < lo:
                    lo = x
                elif x > hi:
                    hi = x
            for t in g:
                posT[t] = x
        prev = cpos
        for t in seq:
            x = posT[t]
            tot += x - prev if x > prev else prev - x
            prev = x
        tot += abs(prev - cpos) + hi - lo + min(hi - cpos, cpos - lo)
        ecache[line] = tot
        return tot

    def neighbors(line):
        n = len(line)
        for a in range(n):
            for b in range(a + 1, n):
                l = list(line)
                l[a], l[b] = l[b], l[a]
                yield tuple(l)
        for a in range(n):
            for b in range(n):
                if b != a and b != a + 1 and b != a - 1:
                    l = list(line)
                    l.insert(b, l.pop(a))
                    yield tuple(l)
        groups = [x for x in range(n) if line[x] is not None]
        for a in groups:
            for b in groups:
                if a != b:
                    l = list(line)
                    l[a] = tuple(sorted(line[a] + line[b]))
                    del l[b]
                    yield tuple(l)
        for a in groups:
            if len(line[a]) > 1:
                for t in line[a]:
                    rest = tuple(x for x in line[a] if x != t)
                    for b in groups:
                        if b != a:
                            l = list(line)
                            l[a] = rest
                            l[b] = tuple(sorted(line[b] + (t,)))
                            yield tuple(l)
                    for b in range(n + 1):
                        l = list(line)
                        l[a] = rest
                        l.insert(b, (t,))
                        yield tuple(l)

    def climb(cur):
        cc = ev(cur)
        improved = True
        while improved and len(ecache) < max_evals:
            improved = False
            for nb in neighbors(cur):
                c = ev(nb)
                if c < cc:
                    cur, cc, improved = nb, c, True
                    break
        return cc, cur

    first = list(dict.fromkeys(seq))
    inits = [(None,) + tuple((t,) for t in first), tuple((t,) for t in first) + (None,), (None, tuple(sorted(first))),
             (tuple(sorted(first)), None)] if not extra else [(None,) + tuple((t,) for t in first)] + list(extra)
    best = min((climb(x) for x in inits), key=lambda r: r[0])
    rng = random.Random(len(seq) * 1000003 + sum(seq))
    for _ in range(restarts):
        if len(ecache) >= max_evals:
            break
        cur = best[1]
        for _ in range(rng.randint(1, 3)):
            nbs = list(neighbors(cur))
            if not nbs:
                break
            cur = rng.choice(nbs)
        r = climb(cur)
        if r[0] <= best[0]:
            best = r
    res = sorted(ecache.items(), key=lambda kv: kv[1])[:keep]
    res += [(x, ecache[x]) for x in inits[:2] if all(x != y for y, _ in res)]
    return [(c, line, [gcost(g)[3 if x - line.index(None) in (1, -1) else 1] for x, g in enumerate(line) if g is not None])
            for line, c in res]


class Lay:
    """A slot layout for one pattern: slot offsets relative to the counter, homes, bodies for u copies."""

    def __init__(self, offs: list, groups: list, homes: list, toks: list, tid: list, umax: int):
        self.offs, self.groups, self.homes, self.toks, self.tid, self.umax = offs, groups, homes, toks, tid, umax
        self.OFF = np.array(self.offs, dtype=np.int64)
        self.H = np.array(homes, dtype=np.int64)
        self.Zs = (PL_C[:, None] + PL_R[:, None] * self.OFF[None, :]) % Z
        slot_of, off_of = {}, {}
        for si, (x, g) in enumerate(zip(offs, groups)):
            for t in g:
                slot_of[t], off_of[t] = si, x
        last = {}
        for nn, t in enumerate(tid):
            last[slot_of[t]] = nn

        def copy(vals, cur, restore):
            out = []
            for nn, t in enumerate(tid):
                x, sl = off_of[t], slot_of[t]
                _, vin, vout, code, _ = toks[t]
                out.append(mvr((x - cur + 15) % Z - 15) + delta_code(vals[sl], vin) + code)
                cur, vals[sl] = x, vout
                if restore and last[sl] == nn:
                    out.append(delta_code(vout, homes[sl]))
                    vals[sl] = homes[sl]
            return "".join(out), cur

        ex = sum(toks[t][4] for t in tid)
        self.S, cur = copy(list(homes), 0, True)
        self.ret = mvr((15 - cur) % Z - 15)
        vals = list(homes)
        self.F, cur = copy(vals, 0, False)
        self.M, _ = copy(list(vals), cur, False)
        self.Lc, _ = copy(vals, cur, True)
        us = np.arange(umax + 1)
        self.BL = np.where(us == 1, len(self.S), len(self.F) + (us - 2) * len(self.M) + len(self.Lc)) + len(self.ret)
        self.BS = self.BL + us * ex
        self.BL[0] = self.BS[0] = 0
        self.i1 = offs.index(1) if 1 in offs else -1
        self.i2 = offs.index(-1) if -1 in offs else -1
        self.oa = -1 if min(self.offs) > 0 else 1 if max(self.offs) < 0 else 0
        # aux candidates (slot index, aux offset, direction aux -> slot, aux zone per placement): a zero zone next to
        # a slot, outside the layout, that can run a multiplication loop setting the slot (single / nested loop)
        aux = lambda excl: [(si, a, x - a, (PL_C + PL_R * a) % Z) for si, x in enumerate(offs) for a in (x - 1, x + 1)
                            if a not in excl]
        self.AX = aux(set(offs) | {0})
        self.AXn = aux(set(offs) | {0, self.oa}) if self.oa else []
        self._inner = None
        if self.oa:
            self.OFFn = np.append(self.OFF, 0)
            self.Hn = np.append(self.H, 0)
            self.Zn = (PL_C[:, None] + PL_R[:, None] * self.OFFn[None, :]) % Z
            self.ZA = (PL_C + PL_R * self.oa) % Z

    def body(self, u: int) -> str:
        return (self.S if u == 1 else self.F + self.M * (u - 2) + self.Lc) + self.ret

    def inner(self):
        """Inner loop table of a nested loop: for each count it_b * u the cheapest (count, cost, steps, it_b, u)."""
        if self._inner is None:
            itb = np.arange(2, A)[:, None]
            us = np.arange(1, self.umax + 1)[None, :]
            cost = CI[0, itb] + self.BL[us] + 2
            steps = (CI[0, itb] - CI_ABS[0, itb]) + itb * (self.BS[us] + CI_ABS[0, itb] + 2)
            nin = itb * us
            itb_b, us_b = np.broadcast_arrays(itb, us)
            o = np.lexsort((steps.ravel(), cost.ravel(), nin.ravel()))
            nn = nin.ravel()[o]
            o = o[np.concatenate([[True], nn[1:] != nn[:-1]])]
            self._inner = (nin.ravel()[o], cost.ravel()[o], steps.ravel()[o], itb_b.ravel()[o], us_b.ravel()[o])
        return self._inner


def walk_eval(need, OFF, e: int, q):
    lo = np.minimum(np.where(need, OFF, e).min(1), e)
    hi = np.maximum(np.where(need, OFF, e).max(1), e)
    dlo = np.abs(q - lo)
    dlo = np.minimum(dlo, Z - dlo)
    dhi = np.abs(q - hi)
    dhi = np.minimum(dhi, Z - dhi)
    return np.minimum(dlo + hi - lo + np.abs(hi - e), dhi + hi - lo + np.abs(e - lo))


def setup_eval(Zs, OFF, H, e: int, memnp, q, i1: int, i2: int, ok, ax: list = ()):
    """Exact setup cost (rune changes + walk from pos ending at relative offset e) for all 60 placements.
    Slots i1 / i2 (offsets +1 / -1, or -1 if absent) may instead be set by a multiplication loop on the counter
    where ok (options 1, 2, 3 = both); option 4 + k sets slot ax[k][0] by a multiplication loop run on the zero aux
    zone ax[k][3] (offset ax[k][1], next to the slot, outside the layout). Returns (cost, option or None, extra steps)
    per placement."""
    w = memnp[Zs]
    cz = CC[w, H]
    need = cz > 0
    tot = cz.sum(1)
    base = tot + walk_eval(need, OFF, e, q)
    if i1 >= 0:
        d1 = (H[i1] - w[:, i1]) % A
        g1 = np.where(ok, cz[:, i1] - MS1[d1], -99)
        if g1.max() <= -2:
            i1 = -1
    if i2 >= 0:
        d2 = (H[i2] - w[:, i2]) % A
        g2 = np.where(ok, cz[:, i2] - MS1[d2], -99)
        if g2.max() <= -2:
            i2 = -1
    cost, opt, ex = base, None, 0
    if i1 >= 0 or i2 >= 0:
        big = np.full(len(base), 999, dtype=np.int64)
        o1 = o2 = o3 = big
        if i1 >= 0:
            nd = need.copy()
            nd[:, i1] = MS1X[d1] > 0
            o1 = np.where(ok, tot - cz[:, i1] + MS1[d1] + walk_eval(nd, OFF, e, q), 999)
        if i2 >= 0:
            nd = need.copy()
            nd[:, i2] = MS1X[d2] > 0
            o2 = np.where(ok, tot - cz[:, i2] + MS1[d2] + walk_eval(nd, OFF, e, q), 999)
            if i1 >= 0:
                nd[:, i1] = MS2X1[d1, d2] > 0
                nd[:, i2] = MS2X2[d1, d2] > 0
                o3 = np.where(ok, tot - cz[:, i1] - cz[:, i2] + MS2[d1, d2] + walk_eval(nd, OFF, e, q), 999)
        O = np.stack([base, o1, o2, o3])
        opt = O.argmin(0)
        cost = O[opt, np.arange(len(base))]
        ex = np.zeros(len(base), dtype=np.int64)
        if i1 >= 0:
            ex = np.where(opt == 1, MS1S[d1], ex)
        if i2 >= 0:
            ex = np.where(opt == 2, MS1S[d2], ex)
            if i1 >= 0:
                ex = np.where(opt == 3, MS2S[d1, d2], ex)
    for k, (si, a, _, za) in enumerate(ax):
        d = (H[si] - w[:, si]) % A
        oka = memnp[za] == 0
        if np.where(oka, cz[:, si] - MS1[d], -99).max() <= 1:
            continue
        nd = need.copy()
        nd[:, si] = MS1X[d] > 0
        c = np.where(oka, tot - cz[:, si] + MS1[d] + walk_eval(np.c_[nd, oka], np.append(OFF, a), e, q), 999)
        f = c < cost
        if f.any():
            if opt is None:
                opt, ex = np.zeros(len(base), dtype=np.int64), np.zeros(len(base), dtype=np.int64)
            cost, opt, ex = np.where(f, c, cost), np.where(f, 4 + k, opt), np.where(f, MS1S[d], ex)
    return cost, opt, ex


def mult_code(r: int, opt: int, d1: int, d2: int) -> tuple:
    """Multiplication loop on a zero zone adding about d1 to its +r neighbour (opt 1), about d2 to its -r neighbour
    (opt 2) or both (opt 3); returns (code, value added at +r, value added at -r) (fixups are left to the caller)."""
    if opt == 3:
        n, b1, b2 = int(MS2N[d1, d2]), int(MS2B1[d1, d2]), int(MS2B2[d1, d2])
        inner = mvr(r) + step_code(-b1) + mvr(-2 * r) + step_code(-b2) + mvr(r)
    else:
        n = int(MS1N[d1] if opt == 1 else MS1N[d2])
        b1 = b2 = int(MS1B[d1] if opt == 1 else MS1B[d2])
        inner = mvr(r) + step_code(-b1) + mvr(-r) if opt == 1 else mvr(-r) + step_code(-b2) + mvr(r)
    code = delta_code(0, int(CI_V0[0, n])) + "[" + inner + step_code(int(CI_SD[0, n])) + "]"
    return code, n * b1 % A if opt != 2 else 0, n * b2 % A if opt != 1 else 0


def sweep_code(pos: int, c: int, r: int, changes: dict, e: int) -> str:
    """Code visiting all relative offsets in changes (applying their codes) and ending at relative offset e."""
    q = (r * (pos - c) + 15) % Z - 15
    keys = list(changes)
    lo, hi = min(keys + [e]), max(keys + [e])
    d1, d2 = abs(q - lo), abs(q - hi)
    d1 = min(d1, Z - d1) + hi - lo + abs(hi - e)
    d2 = min(d2, Z - d2) + hi - lo + abs(e - lo)
    start, order = (lo, sorted(keys)) if d1 <= d2 else (hi, sorted(keys, reverse=True))
    out = [move_code(pos, (c + r * start) % Z)]
    cur = start
    for x in order:
        out.append(mvr((x - cur) * r) + changes[x])
        cur = x
    out.append(mvr((e - cur) * r))
    return "".join(out)


def reuse_search(toks: list, seq: list, mem: list, pos: int, it: int, inits: list, budget: int):
    """Local search over absolute slot zones for the current memory (reuses runes already present).
    A solution is (counter zone, sorted ((types, zone), ...)). Returns (cost, solution)."""
    tin, tout = [t[1] for t in toks], [t[2] for t in toks]
    occ = [[] for _ in range(len(toks))]
    for x, t in enumerate(seq):
        occ[t].append(x)
    gst = {}

    def gstat(g):
        r = gst.get(g)
        if r is None:
            idxs = occ[g[0]] if len(g) == 1 else sorted(x for t in g for x in occ[t])
            c = 0
            prev = tout[seq[idxs[0]]]
            for x in idxs[1:]:
                t = seq[x]
                c += CDL[prev][tin[t]]
                prev = tout[t]
            r = gst[g] = (c, prev, tin[seq[idxs[0]]])
        return r

    cil = CI[:, it].tolist()
    ecache = {}

    def ev(sol):
        r = ecache.get(sol)
        if r is not None:
            return r
        c, gz = sol
        tot = cil[mem[c]]
        lo = hi = 0
        rel = {}
        for g, z in gz:
            internal, a, b = gstat(g)
            v = mem[z]
            tot += internal + BHML[v][a][b]
            x = (z - c + 15) % Z - 15
            if v != BHMHL[v][a][b]:
                if x < lo:
                    lo = x
                elif x > hi:
                    hi = x
            for t in g:
                rel[t] = x
        q = (pos - c + 15) % Z - 15
        dlo, dhi = abs(q - lo), abs(q - hi)
        tot += min(min(dlo, Z - dlo) + hi - lo + hi, min(dhi, Z - dhi) + hi - lo - lo)
        prev = 0
        for t in seq:
            x = rel[t]
            d = abs(x - prev)
            tot += d if d <= 15 else Z - d
            prev = x
        d = abs(prev)
        tot += d if d <= 15 else Z - d
        ecache[sol] = tot
        return tot

    valz = {}
    for z in range(Z):
        valz.setdefault(mem[z], []).append(z)

    def neighbors(sol):
        c, gz = sol
        used = {c} | {z for _, z in gz}
        near = {(z + d) % Z for z in used for d in (-2, -1, 1, 2)} - used
        n = len(gz)
        for a in range(n):
            g, z = gz[a]
            cand = near | ({zz for t in g for zz in valz.get(tin[t], ())} - used)
            for zz in cand:
                l = list(gz)
                l[a] = (g, zz)
                yield c, tuple(sorted(l))
        for zz in near:
            yield zz, gz
        for a in range(n):
            l = list(gz)
            l[a] = (gz[a][0], c)
            yield gz[a][1], tuple(sorted(l))
            for b in range(a + 1, n):
                l = list(gz)
                l[a], l[b] = (gz[a][0], gz[b][1]), (gz[b][0], gz[a][1])
                yield c, tuple(sorted(l))
        for a in range(n):
            for b in range(n):
                if a != b:
                    l = [x for y, x in enumerate(gz) if y != b]
                    l[a if a < b else a - 1] = (tuple(sorted(gz[a][0] + gz[b][0])), gz[a][1])
                    yield c, tuple(sorted(l))
        for a in range(n):
            g, z = gz[a]
            if len(g) > 1:
                for t in g:
                    rest = tuple(x for x in g if x != t)
                    for b in range(n):
                        if b != a:
                            l = list(gz)
                            l[a], l[b] = (rest, z), (tuple(sorted(gz[b][0] + (t,))), gz[b][1])
                            yield c, tuple(sorted(l))
                    for zz in near | (set(valz.get(tin[t], ())) - used):
                        l = list(gz)
                        l[a] = (rest, z)
                        l.append(((t,), zz))
                        yield c, tuple(sorted(l))

    best = min(((ev(x), x) for x in inits), key=lambda r: r[0])
    cur, cc = best[1], best[0]
    improved = True
    while improved and len(ecache) < budget:
        improved = False
        for nb in neighbors(cur):
            c = ev(nb)
            if c < cc:
                cur, cc, improved = nb, c, True
                break
            if len(ecache) >= budget:
                break
    c, gz = cur
    return cc, cur, [BHMHL[mem[z]][gstat(g)[1]][gstat(g)[2]] for g, z in gz]


_LAY_CACHE, _CLASS_CACHE = {}, {}  # pattern -> layouts, rotation class -> best lines (shared by all solves of a run)


def make_repeat_gen(tv: list):
    n = len(tv)
    tva = np.array(tv, dtype=np.int64)
    pmax = min(40, n // 2)
    starts = [[] for _ in range(n + 1)]
    runs = {}
    for p in range(1, pmax + 1):
        eq = tva[:-p] == tva[p:]
        n1 = len(eq)
        nf = np.minimum.accumulate(np.where(eq, n1, np.arange(n1))[::-1])[::-1]
        run = nf - np.arange(n1)
        runs[p] = run
        segs = np.nonzero((run > 0) & np.concatenate([[True], run[:-1] == 0]))[0]
        for s in segs.tolist():
            end = s + int(run[s]) + p
            cand = []
            for i in range(s, min(s + min(p, 10) + 2, end)):
                m = (end - i) // p
                if m < (6 if p == 1 else 2) or m * p < 6:
                    continue
                pat = tv[i:i + p]
                if any(p % q == 0 and pat == pat[:q] * (p // q) for q in range(1, p)):
                    continue
                cand.append((len(tokenize(tuple(pat))[-1]), i, m))
            if cand:
                best = min(c[0] for c in cand)
                for nt, i, m in cand:
                    if nt <= best + 2:
                        starts[i].append((p, m))
    lay_cache, class_cache = _LAY_CACHE, _CLASS_CACHE
    st = {"last_i": -1}

    def layouts(P: tuple):
        r = lay_cache.get(P)
        if r is None:
            p = len(P)
            umax = min(26, 30 // p + 1)
            r, variants, alt = [], [], []
            for toks in tokenize(P):
                keys = list(dict.fromkeys(t[0] for t in toks))
                big = len(keys) > 12
                kid = {k: x for x, k in enumerate(keys)}
                tid = [kid[t[0]] for t in toks]
                tt = [toks[tid.index(x)] for x in range(len(keys))]
                variants.append((tt, tid, umax))
                sk = [t[0] for t in toks]
                ck = min(tuple(sk[x:] + sk[:x]) for x in range(len(sk)))
                prev = class_cache.get(ck)
                extra = tuple(tuple(None if g is None else tuple(sorted(kid[k] for k in g)) for g in line) for line in prev or ())
                found = layout_search([t[1] for t in tt], [t[2] for t in tt], tid, (100 if prev else 300) if big else 2000, 3, 0 if prev or big else 10, extra)
                if not prev:
                    class_cache[ck] = [tuple(None if g is None else tuple(keys[t] for t in g) for g in line) for _, line, _ in found[:3]]
                # the best lines again with homes next to the counter that a multiplication loop reaches with a fixup
                exact = {(line, tuple(homes)) for _, line, homes in found}
                found_f = [x for x in layout_search([t[1] for t in tt], [t[2] for t in tt], tid, 60 if big else 600, 2, 0 if big else 3,
                                                    tuple(line for _, line, _ in found[:2]), (BH1FL, BH1FHL))
                           if (x[1], tuple(x[2])) not in exact]
                for lst, fl in ((r, found), (alt, found_f)):
                    for c, line, homes in fl:
                        cpos = line.index(None)
                        offs = [x - cpos for x, g in enumerate(line) if g is not None]
                        lst.append((c + sum(len(t[3]) for t in toks), Lay(offs, [g for g in line if g is not None], homes, tt, tid, umax)))
            r.sort(key=lambda x: x[0])
            alt.sort(key=lambda x: x[0])
            # up to LAY_ALT alternative layouts that differ from all kept ones and their mirror images (every layout is
            # tried in both orientations)
            uniq = [x[1] for x in r[:LAY_KEEP]]
            key = lambda lay, s: tuple(sorted(zip([s * x for x in lay.offs], map(tuple, lay.groups), lay.homes)))
            seen = {key(lay, s) for lay in uniq for s in (1, -1)}
            for _, lay in alt:
                if len(uniq) < LAY_KEEP + LAY_ALT and key(lay, 1) not in seen:
                    seen.update((key(lay, 1), key(lay, -1)))
                    uniq.append(lay)
            r = lay_cache[P] = (uniq, variants)
        return r

    def gen(tv: list, i: int, mem: list, pos: int) -> list:
        cs = starts[i]
        if not cs:
            return []
        first_call = st["last_i"] != i
        st["last_i"] = i
        memnp = np.array(mem, dtype=np.int64)
        q = (PL_R * (pos - PL_C) + 15) % Z - 15
        cap = REPEAT_STEP_CAP
        Js, Cs, Ss, specs = [], [], [], []

        def evaluate(lay, p, m):
            umax = min(m, len(lay.BL) - 1)
            # single loop
            S1, opt1, ex1 = setup_eval(lay.Zs, lay.OFF, lay.H, 0, memnp, q, lay.i1, lay.i2, memnp[PL_C] == 0, lay.AX)
            T = S1[:, None] + CI1[memnp[PL_C]]
            pl = T.argmin(0)
            b1 = T[pl, AR26]
            sd = CI_ABS[memnp[PL_C[pl]], AR26 + 1]
            its = (AR26 + 1)[:, None]
            us = np.arange(1, umax + 1)[None, :]
            cost = b1[:, None] + lay.BL[us] + 2
            steps = (b1 - sd + (ex1[pl] if opt1 is not None else 0))[:, None] + its * (lay.BS[us] + sd[:, None] + 2)
            N = its * us
            ok = (N <= m) & (steps <= cap - 8 * (n - N * p))
            if ok.any():
                ii, uu = np.nonzero(ok)
                Js.append(i + N[ii, uu] * p)
                Cs.append(cost[ii, uu])
                Ss.append(steps[ii, uu])
                specs.extend((0, lay, int(pl[a]), a + 1, 0, b + 1, 0 if opt1 is None else int(opt1[pl[a]]))
                             for a, b in zip(ii.tolist(), uu.tolist()))
            # nested loops for long runs
            if m > 26 and lay.oa:
                i1, i2 = (lay.i1, -1) if lay.oa < 0 else (-1, lay.i2)
                S2, opt2, ex2 = setup_eval(lay.Zn, lay.OFFn, lay.Hn, lay.oa, memnp, q, i1, i2, True, lay.AXn)
                T2 = S2[:, None] + CI1[memnp[lay.ZA]]
                pl = T2.argmin(0)[1:]
                b2 = T2[pl, AR26[1:]]
                sd = CI_ABS[memnp[lay.ZA[pl]], AR26[1:] + 1]
                ita = (AR26[1:] + 1)[:, None]
                in_n, in_cost, in_steps, in_itb, in_u = lay.inner()
                sel = in_n <= m // 2
                nin = in_n[sel][None, :]
                cost = b2[:, None] + in_cost[sel][None, :] + 4
                steps = (b2 - sd + (ex2[pl] if opt2 is not None else 0))[:, None] + ita * (in_steps[sel][None, :] + sd[:, None] + 4)
                N = ita * nin
                ok = (N <= m) & (steps <= cap - 8 * (n - N * p))
                if ok.any():
                    ii, kk = np.nonzero(ok)
                    Js.append(i + N[ii, kk] * p)
                    Cs.append(cost[ii, kk])
                    Ss.append(steps[ii, kk])
                    itb, uu = in_itb[sel], in_u[sel]
                    specs.extend((1, lay, int(pl[a]), a + 2, int(itb[b]), int(uu[b]), 0 if opt2 is None else int(opt2[pl[a]]))
                                 for a, b in zip(ii.tolist(), kk.tolist()))
            return T

        memset = set(mem)
        for p, m in cs:
            P = tuple(tv[i:i + p])
            lays, variants = layouts(P)
            tabs = [(lay, evaluate(lay, p, m)) for lay in lays]
            if first_call and REUSE_BUDGET and any(v and v in memset for v in P):
                itx = min(m, 26)
                for tt, tid, umax in variants:
                    cand = sorted(((int(T[:, itx - 1].min()) + lay.BL[1], lay, int(T[:, itx - 1].argmin())) for lay, T in tabs
                                   if lay.tid is tid), key=lambda x: x[0])[:2]
                    inits = []
                    for _, lay, plc in cand:
                        c, r = int(PL_C[plc]), int(PL_R[plc])
                        inits.append((c, tuple(sorted((g, (c + r * off) % Z) for g, off in zip(lay.groups, lay.offs)))))
                    if not inits:
                        continue
                    _, (c, gz), homes = reuse_search(tt, tid, mem, pos, itx, inits, REUSE_BUDGET)
                    evaluate(Lay([(z - c + 15) % Z - 15 for _, z in gz], [g for g, _ in gz], homes, tt, tid, umax), p, m)
        if not Js:
            return []
        J, C, S = np.concatenate(Js), np.concatenate(Cs), np.concatenate(Ss)
        o = np.lexsort((S, C, -J))
        Jo, Co = J[o], C[o]
        first = np.concatenate([[True], Jo[1:] != Jo[:-1]])
        of, Jf, Cf = o[first], Jo[first], Co[first]
        runmin = np.minimum.accumulate(Cf)
        keep = np.concatenate([[True], Cf[1:] < runmin[:-1]])
        res = []
        kj = set(Jf[keep].tolist())
        bestc = dict(zip(Jf.tolist(), Cf.tolist()))
        used = {}
        for x in o.tolist():
            j = int(J[x])
            if j not in kj or C[x] > bestc[j] + ALT_SLACK:
                continue
            kind, lay, pl, it, itb, u, opt = specs[x]
            lays_j = used.setdefault(j, [])
            if lay in lays_j or len(lays_j) >= ALT_PER_J:
                continue
            lays_j.append(lay)
            res.append(build(kind, lay, pl, it, itb, u, opt, mem, pos, j))
        return res

    def build(kind, lay, pl, it, itb, u, opt, mem, pos, j):
        c, r = int(PL_C[pl]), int(PL_R[pl])
        nm = list(mem)
        changes = {}
        for off, h in zip(lay.offs, lay.homes):
            z = (c + r * off) % Z
            if nm[z] != h:
                changes[off] = change_code(nm[z], h)
                nm[z] = h
        mc = ""
        if opt:
            # slots set by a multiplication loop (on the aux zone, or on the zero counter), then fixed up
            if opt >= 4:
                si, ao, dr, _ = (lay.AX if kind == 0 else lay.AXn)[opt - 4]
                fx = [(lay.offs[si], lay.homes[si])]
                code, add, _ = mult_code(r * dr, 1, (lay.homes[si] - mem[(c + r * lay.offs[si]) % Z]) % A, 0)
                changes[ao], adds = code, [add]
            else:
                fx = [(o, lay.homes[x]) for o, x, f in ((1, lay.i1, opt != 2), (-1, lay.i2, opt != 1)) if f]
                d1 = (lay.homes[lay.i1] - mem[(c + r) % Z]) % A if opt != 2 else 0
                d2 = (lay.homes[lay.i2] - mem[(c - r) % Z]) % A if opt != 1 else 0
                mc, add1, add2 = mult_code(r, opt, d1, d2)
                adds = [add for add, f in ((add1, opt != 2), (add2, opt != 1)) if f]
            for (off, h), add in zip(fx, adds):
                fix = delta_code((mem[(c + r * off) % Z] + add) % A, h)
                if fix:
                    changes[off] = fix
                else:
                    del changes[off]
        body = lay.body(u) if r == 1 else lay.body(u).translate(SWAP)
        # executed steps: setup (simulated when it contains a loop) + iterations of the (nested) loop, whose body
        # executes lay.BS[u] steps
        if kind == 0:
            v = mem[c]
            changes[0] = mc + change_code(v, int(CI_V0[v, it]))
            setup = sweep_code(pos, c, r, changes, 0)
            nm[c] = 0
            st = it * (int(lay.BS[u]) + int(CI_ABS[v, it]) + 2)
            return j, setup + "[" + body + step_code(int(CI_SD[v, it])) + "]", nm, c, \
                (len(setup) if "[" not in setup else simulate(setup, mem, pos)[3]) + st
        a = (c + r * lay.oa) % Z
        if nm[c] or mc:
            changes[0] = change_code(nm[c], 0) + mc
            nm[c] = 0
        v = mem[a]
        changes[lay.oa] = change_code(v, int(CI_V0[v, it]))
        setup = sweep_code(pos, c, r, changes, lay.oa)
        ini = delta_code(0, int(CI_V0[0, itb]))
        code = setup + "[" + mvr(-lay.oa * r) + ini + "[" + body + step_code(int(CI_SD[0, itb])) + "]" + mvr(lay.oa * r) + \
            step_code(int(CI_SD[v, it])) + "]"
        nm[a] = 0
        st = it * (4 + len(ini) + itb * (int(lay.BS[u]) + int(CI_ABS[0, itb]) + 2) + int(CI_ABS[v, it]))
        return j, code, nm, a, (len(setup) if "[" not in setup else simulate(setup, mem, pos)[3]) + st

    return gen


# ---------------------------------------------------------------- progression macros
# Family 1 ("linear loop"): tv[i:i+m*p] where the k-th period equals P0 + k*D (per position, mod 27). Positions with equal D
# may share a zone (intra-body adjustments). The loop condition is either a track zone that hits 0 exactly after the last
# iteration (its wrap adjustment may be split around the check, e.g. "[+.++.+]") or an external counter zone.
# Family 2 ("nested loop"): an outer counter loop repeating a block = inner sentinel loop ([.+d] or [+d.]) followed by up to
# 2 letters printed from the inner zone or from constant zones.
# Every option is a set of zone roles at relative offsets; its setup cost for all 30 bases x 2 orientations is evaluated
# with numpy, the code is only generated for the cheapest options per end index.

def sd(a: int, b: int) -> int:
    return (b - a + 13) % 27 - 13


def adj(x: int) -> str:
    return "+" * x if x > 0 else "-" * -x


def mv(x: int) -> str:
    return ">" * x if x > 0 else "<" * -x


ORD = [27 // gcd(d, 27) for d in range(27)]
ZT = [[next((t for t in range(27) if (v + t * d) % 27 == 0), -1) for d in range(27)] for v in range(27)]
CNT_C = np.full((27, A), 999, dtype=np.int64)
CNT_V = np.zeros((27, A), dtype=np.int64)
CNT_S = np.zeros((27, A), dtype=np.int64)
for _m in range(2, 27):
    for _s in sorted(range(-13, 14), key=abs):
        _c0 = (-_m * _s) % 27
        if _s == 0 or _c0 == 0 or ZT[_c0][_s % 27] != _m:
            continue
        _cost = CC[:, _c0] + abs(_s)
        _b = _cost < CNT_C[_m]
        CNT_C[_m][_b], CNT_V[_m][_b], CNT_S[_m][_b] = _cost[_b], _c0, _s
ZERO27 = np.zeros(A, dtype=np.int64)
MMIN = 3
F1_REPEATS = True
GEN_BUDGET, TOP2_MAXOPT = 0.3, 40  # seconds of progression-macro work per phrase; 2nd state only for small levels
F2_POST = 3
LABELS = [[[]]]
for _ in range(F2_POST):
    LABELS.append([q + [x] for q in LABELS[-1] for x in range(max(q, default=0) + 2)])
TRK_KEEP, TRK_SLACK, CNT_KEEP, PARETO, TRK_BUDGET = 8, 3, 4, 1, 8
CMIN = 3
GT_SPAN, GT_EXTRA, GT_MIN = 3, 8, 8  # general track checks: counts mx..mx-GT_SPAN+1 printing >= GT_MIN chars, <= GT_EXTRA extra body chars


def set_partitions(items: list):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        yield [[first]] + part
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]


def tour(perm: tuple, seq: tuple, oc: int) -> int:
    t, cur = 0, oc
    for r in seq:
        t += abs(perm[r] - cur)
        cur = perm[r]
    return t + abs(oc - cur)


_PARTS = {}


def partitions_for(D: tuple) -> list:
    """Returns [(zones, seq)] for all partitions of the positions into zones of equal D."""
    lab = {}
    pat = tuple(lab.setdefault(x, len(lab)) for x in D)
    r = _PARTS.get(pat)
    if r is None:
        r = []
        for part in set_partitions(list(range(len(D)))):
            if all(len({pat[q] for q in z}) == 1 for z in part):
                zones = sorted(sorted(z) for z in part)
                seq = [0] * len(D)
                for zi, z in enumerate(zones):
                    for q in z:
                        seq[q] = zi
                r.append((zones, tuple(seq)))
        _PARTS[pat] = r
    return r


_LAY = {}


def best_layouts(seq: tuple, g: int, cond: int) -> list:
    """Returns the cheapest [(tour, perm)] placing g zones (+ a counter if cond < 0) with the check at zone cond / counter."""
    key = (seq, g, cond)
    r = _LAY.get(key)
    if r is None:
        k = g + (cond < 0)
        rc = g if cond < 0 else cond
        lst = sorted((tour(perm, seq, perm[rc]), perm) for perm in permutations(range(k))
                     if perm <= tuple(k - 1 - x for x in perm))
        r = _LAY[key] = [x for x in lst if x[0] <= lst[0][0] + (cond < 0)][:4 if cond < 0 else 2]
    return r


_SHAPES = {}


def f1_shape(D: tuple, rel: tuple):
    """Returns (counter_layouts, track_layouts) for a period shape; entries (body_cost_without_counter, zones, perm[, tz])."""
    key = (D, rel)
    r = _SHAPES.get(key)
    if r is not None:
        return r
    p = len(D)
    cnt, trk = [], []
    for zones, seq in partitions_for(D):
        g = len(zones)
        base = 2 + p
        for z in zones:
            base += sum(abs(sd(rel[a], rel[b])) for a, b in zip(z, z[1:])) + abs(sd(rel[z[-1]], rel[z[0]] + D[z[0]]))
        for t, perm in best_layouts(seq, g, -1):
            cnt.append((base + t, zones, perm))
        for tz in range(g):
            if D[zones[tz][0]]:
                for t, perm in best_layouts(seq, g, tz):
                    trk.append((base + t, zones, perm, tz))
    cnt.sort(key=lambda x: x[0])
    trk.sort(key=lambda x: x[0])
    r = _SHAPES[key] = ([x for x in cnt if x[0] <= cnt[0][0] + 1][:4], trk)
    return r


_SC = {}


def sc_row(vals: tuple):
    """Returns (cost[u], argmin index[u]) for setting a zone holding u to the cheapest of vals."""
    r = _SC.get(vals)
    if r is None:
        t = CC[:, list(vals)]
        r = _SC[vals] = (t.min(1).astype(np.int64), t.argmin(1).tolist())
    return r


_SPLIT = {}


def split_vals(last: int, first: int, D: int):
    """Returns (w, sign, boundary values) for a zone whose last print is last and next first print is first + D."""
    key = (last, first, D)
    r = _SPLIT.get(key)
    if r is None:
        w = sd(last, first + D)
        sgn = 1 if w >= 0 else -1
        r = _SPLIT[key] = (w, sgn, tuple((last - D + sgn * x) % 27 for x in range(abs(w) + 1)))
    return r


def setup_code(mem: list, pos: int, phys: list, offs: list, ent: list, rc: int, sg: int) -> str:
    need = {offs[r]: r for r in range(len(offs)) if mem[phys[r]] != ent[r]}
    oc = offs[rc]
    lo, hi = min(min(need, default=oc), oc), max(max(need, default=oc), oc)
    plo, phi = phys[offs.index(lo)], phys[offs.index(hi)]
    if MD[pos, plo] + hi - oc <= MD[pos, phi] + oc - lo:
        code, rng = move_code(pos, plo), range(lo, hi + 1)
    else:
        code, rng = move_code(pos, phi), range(hi, lo - 1, -1)
    prev = rng[0]
    for off in rng:
        code += mv(sg * (off - prev))
        prev = off
        if off in need:
            r = need[off]
            code += change_code(mem[phys[r]], ent[r])
    return code + mv(sg * (oc - prev))


class Opt:
    """One macro option: kind (1 linear, 2 nested), start i, phrase chars per iteration step, iteration counts ms (counter
    options: several), zone role offsets offs, check role rc, per-role setup cost rows sc, body cost (code chars without the
    counter step), executed steps per iteration spi (without the counter step), cnt (has a counter role) and code data."""
    __slots__ = ("kind", "i", "step", "ms", "offs", "rc", "sc", "body", "spi", "cnt", "data")

    def __init__(self, kind, i, step, ms, offs, rc, sc, body, spi, cnt, data):
        self.kind, self.i, self.step, self.ms, self.offs, self.rc, self.sc, self.body, self.spi, self.cnt, self.data = \
            kind, i, step, ms, offs, rc, sc, body, spi, cnt, data


def f1_code(o: Opt, m: int, mem: list, pos: int, b: int, sg: int):
    P0, D, zones, cond, cjj = o.data
    offs, rc = o.offs, o.rc
    phys = [(b + sg * off) % Z for off in offs]
    g = len(zones)
    ent, js, ws = [], [], []  # entry value, change after the last print (c1), wrap per zone
    for r, z in enumerate(zones):
        w, sgn, vals = split_vals(P0[z[-1]], P0[z[0]], D[z[0]])
        if r == cond:
            c1 = cjj
            ent.append((P0[z[-1]] - D[z[0]] + c1) % 27)
        else:
            jj = sc_row(vals)[1][mem[phys[r]]]
            c1 = sgn * jj
            ent.append(vals[jj])
        js.append(c1)
        ws.append(w)
    s = 0
    if cond < 0:
        u = mem[phys[g]]
        ent.append(int(CNT_V[m][u]))
        s = int(CNT_S[m][u])
    code = setup_code(mem, pos, phys, offs, ent, rc, sg)
    zof, prv = {}, {}
    for r, z in enumerate(zones):
        for a, q in enumerate(z):
            zof[q] = r
            prv[q] = z[a - 1] if a else -1
    body, cur = "", offs[rc]
    for q in range(len(P0)):
        r = zof[q]
        body += mv(sg * (offs[r] - cur))
        cur = offs[r]
        body += (adj(sd(0, ws[r] - js[r])) if prv[q] < 0 else adj(sd(P0[prv[q]], P0[q]))) + "."
        if q == zones[r][-1]:
            body += adj(js[r])
    body += mv(sg * (offs[rc] - cur)) + adj(s)
    nm = list(mem)
    for r, z in enumerate(zones):
        nm[phys[r]] = (ent[r] + m * D[z[0]]) % 27
    if cond < 0:
        nm[phys[g]] = 0
    # executed steps: setup (straight line unless it resets a rune) + m iterations of "[" body "]"
    st = len(code) if "[" not in code else simulate(code, mem, pos)[3]
    return code + "[" + body + "]", nm, phys[rc], st + m * (len(body) + 2)


def f2_code(o: Opt, m: int, mem: list, pos: int, b: int, sg: int):
    dd, lazy, e, post, labels, ka = o.data
    offs, rc = o.offs, o.rc
    phys = [(b + sg * off) % Z for off in offs]
    seqs = [[ch for ch, lb in zip(post, labels) if lb == a] for a in range(ka + 1)]
    lv = seqs[0][-1] if seqs[0] else 0
    ws, sgs, js, ent = [], [], [], []
    for a, sq in enumerate(seqs):
        w = sd(lv, e) if a == 0 else sd(sq[-1], sq[0])
        sgn = 1 if w >= 0 else -1
        start = lv if a == 0 else sq[-1]
        vals = tuple((start + sgn * x) % 27 for x in range(abs(w) + 1))
        jj = sc_row(vals)[1][mem[phys[a]]]
        ws.append(w)
        sgs.append(sgn)
        js.append(jj)
        ent.append(vals[jj])
    u = mem[phys[rc]]
    ent.append(int(CNT_V[m][u]))
    code = setup_code(mem, pos, phys, offs, ent, rc, sg)
    body = mv(sg * (offs[0] - offs[rc])) + adj(sgs[0] * (abs(ws[0]) - js[0])) + ("[" + adj(dd) + ".]" if lazy else "[." + adj(dd) + "]")
    first, last = {}, {}
    for x, lb in enumerate(labels):
        first.setdefault(lb, x)
        last[lb] = x
    if 0 not in last:
        body += adj(sgs[0] * js[0])
    cur, val = offs[0], [0] * (ka + 1)
    for x, (ch, lb) in enumerate(zip(post, labels)):
        body += mv(sg * (offs[lb] - cur))
        cur = offs[lb]
        body += (adj(sgs[lb] * (abs(ws[lb]) - js[lb])) if lb and x == first[lb] else adj(sd(val[lb], ch))) + "."
        val[lb] = ch
        if x == last[lb]:
            body += adj(sgs[lb] * js[lb])
    body += mv(sg * (offs[rc] - cur)) + adj(int(CNT_S[m][u]))
    nm = list(mem)
    for r in range(ka + 1):
        nm[phys[r]] = ent[r]
    nm[phys[rc]] = 0
    return code + "[" + body + "]", nm, phys[rc]


_PROG_CACHE = {}


def make_prog_gen(tv: list, pmax: int = 4, keep: int = 2, top: int = 2):
    n = len(tv)
    tva = np.array(tv, dtype=np.int64)
    ps = [0] * (n + 1)
    for x in range(1, n):
        ps[x] = ps[x - 1] + int(CD[tv[x - 1], tv[x]])
    mmax = []
    for p in range(pmax + 1):
        mm = [0] * (n + 1)
        if p and n >= 2 * p:
            ok = ((tva[2 * p:] - 2 * tva[p:n - p] + tva[:n - 2 * p]) % 27 == 0).tolist()
            run = 0
            for x in range(n - 2 * p, -1, -1):
                if x < n - 2 * p:
                    run = run + 1 if ok[x] else 0
                mm[x] = min(run // p + 2, (n - x) // p)
        mmax.append(mm)
    eqrun = {}

    def reps(i: int, L: int) -> int:
        if L not in eqrun:
            r = [0] * (n + 1)
            if L < n:
                eq = (tva[L:] == tva[:n - L]).tolist()
                run = 0
                for x in range(n - L - 1, -1, -1):
                    run = run + 1 if eq[x] else 0
                    r[x] = run
            eqrun[L] = r
        return 1 + eqrun[L][i] // L

    def f1_opts(i: int, res: list):
        for p in range(1, pmax + 1):
            mx = mmax[p][i]
            if mx < 2 or mx * p < 3:
                continue
            P0 = tuple(tv[i:i + p])
            D = tuple((tv[i + p + q] - P0[q]) % 27 for q in range(p))
            if not any(D) and not F1_REPEATS:
                continue
            cnt, trk = f1_shape(D, tuple((x - P0[0]) % 27 for x in P0))
            cand, valid = [], {}
            for cost, zones, perm, tz in trk:
                vk = (id(zones), tz)
                if vk not in valid:
                    z = zones[tz]
                    dz = D[z[0]]
                    w, sgn, vals = split_vals(P0[z[-1]], P0[z[0]], dz)
                    lst, seen = [], set()
                    for jj in range(abs(w) + 1):
                        t = ZT[(P0[z[-1]] + sgn * jj) % 27][dz]
                        if 1 <= t < ORD[dz] - 1 and t + 1 <= mx and (t + 1) * p >= 3 and (t >= MMIN - 1 or p > 1):
                            lst.append((sgn * jj, t + 1, 0 < jj < abs(w), 0))
                            seen.add(t + 1)
                    # general check value: the track may hit 0 after any count m < ORD by passing its check value
                    # outside the wrap range (extra in-body changes); c1 = change after the zone's last print
                    for m in range(min(mx, ORD[dz] - 1), max(mx - GT_SPAN, 1), -1):
                        if m in seen or m < 3 or m * p < GT_MIN:
                            continue
                        c1 = sd(P0[z[-1]], (1 - m) * dz)
                        ex = abs(c1) + abs(sd(0, w - c1)) - abs(w)
                        if ex <= GT_EXTRA:
                            lst.append((c1, m, True, ex))
                    valid[vk] = lst
                for c1, m, mid, ex in valid[vk]:
                    cand.append((cost + ex, m, zones, perm, tz, c1, mid))
            # budget per period: every iteration count first gets its cheapest option (sentinel-like splits and long loops
            # first), then the remaining budget goes to alternatives within TRK_SLACK
            cand.sort(key=lambda c: c[0])
            by_m = {}
            for c in cand:
                by_m.setdefault(c[1], []).append(c)
            order = sorted(by_m, key=lambda m: (all(c[6] for c in by_m[m]), -m))
            pick = [by_m[m][0] for m in order[:TRK_BUDGET]]
            for m in order:
                for c in by_m[m][1:TRK_KEEP]:
                    if len(pick) < TRK_BUDGET and c[0] <= by_m[m][0][0] + TRK_SLACK:
                        pick.append(c)
            for cost, m, zones, perm, tz, c1, mid in pick:
                sc = []
                for r, z in enumerate(zones):
                    vals = split_vals(P0[z[-1]], P0[z[0]], D[z[0]])[2]
                    sc.append(sc_row(((P0[z[-1]] - D[z[0]] + c1) % 27,) if r == tz else vals)[0])
                res.append(Opt(1, i, p, [m], list(perm), tz, sc, cost, cost, False, (P0, D, zones, tz, c1)))
            if mx < CMIN:
                continue
            cap = min(mx, 26)
            ms = [m for m in range(cap, max(cap - 3, CMIN - 1), -1) if m * p >= 4]
            if not ms:
                continue
            for cost, zones, perm in cnt[:CNT_KEEP]:
                sc = [sc_row(split_vals(P0[z[-1]], P0[z[0]], D[z[0]])[2])[0] for z in zones]
                res.append(Opt(1, i, p, ms, list(perm), len(zones), sc + [CNT_C[ms[0]]], cost, cost, True, (P0, D, zones, -1, 0)))

    def f2_opts(i: int, res: list):
        if i + 2 > n or mmax[1][i] < 3:
            return
        v0 = tv[i]
        d = (tv[i + 1] - v0) % 27
        t = ZT[v0][d]
        if d == 0 or t < 0:
            return
        dd = sd(0, d)
        for lazy in (False, True):
            r = t + 1 if lazy else t
            e = (v0 - d) % 27 if lazy else v0
            if e == 0 or r < 3 or r > mmax[1][i]:
                continue
            for lp in range(F2_POST + 1):
                L = r + lp
                if i + 2 * L > n:
                    break
                mx = min(reps(i, L), 26)
                if mx < 2:
                    continue
                ms = list(range(mx, max(mx - 3, 1), -1))
                post = tuple(tv[i + r:i + L])
                for labels in LABELS[lp]:
                    ka = max(labels, default=0)
                    seqs = [[ch for ch, lb in zip(post, labels) if lb == a] for a in range(ka + 1)]
                    zc, prev = 0, 0
                    for ch in seqs[0]:
                        zc += abs(sd(prev, ch))
                        prev = ch
                    w = sd(prev, e)
                    base = 2 + 3 + abs(dd) + lp + zc + abs(w)
                    scs = [sc_row(tuple((prev + (1 if w >= 0 else -1) * x) % 27 for x in range(abs(w) + 1)))[0]]
                    for sq in seqs[1:]:
                        wa = sd(sq[-1], sq[0])
                        base += sum(abs(sd(x, y)) for x, y in zip(sq, sq[1:])) + abs(wa)
                        scs.append(sc_row(tuple((sq[-1] + (1 if wa >= 0 else -1) * x) % 27 for x in range(abs(wa) + 1)))[0])
                    for t2, perm in best_layouts(tuple([0] + labels), ka + 1, -1)[:2]:
                        res.append(Opt(2, i, L, ms, list(perm), ka + 1, scs + [CNT_C[ms[0]]], base + t2,
                                       base + t2 + (r - 1) * (3 + abs(dd)), True, (dd, lazy, e, post, labels, ka)))

    cache, bcache = _PROG_CACHE.setdefault((tuple(tv), pmax), ({}, {}))  # option tables per index, shared by all solves
    scap = ((STEP_LIMIT - 200) / n, 2 * (STEP_LIMIT - 200) / n)  # executed steps allowed per printed char (long, short loops)
    bs = np.concatenate([AR, AR])
    sgs = np.concatenate([np.ones(Z, dtype=np.int64), -np.ones(Z, dtype=np.int64)])

    def build(i: int):
        res = []
        f2_opts(i, res)
        key = None if res else (tuple(tv[i:i + 2 * pmax]), tuple(min(mmax[p][i], 27) for p in range(1, pmax + 1)))
        if key in bcache:
            return bcache[key]
        f1_opts(i, res)
        if not res:
            bcache[key] = None
            return None
        kmax = max(len(o.offs) for o in res)
        no = len(res)
        offb = np.zeros((no, kmax), dtype=np.int64)
        isc = np.zeros((no, kmax), dtype=bool)
        rows = []
        for x, o in enumerate(res):
            k = len(o.offs)
            offb[x, :k] = o.offs
            offb[x, k:] = o.offs[o.rc]
            isc[x, o.rc] = True
            rows += o.sc + [ZERO27] * (kmax - k)
        scf = np.array(rows, dtype=np.int64)
        rowid = np.broadcast_to(np.arange(no * kmax).reshape(no, 1, kmax), (no, 2 * Z, kmax))
        phb = (bs[None, :, None] + sgs[None, :, None] * offb[:, None, :]) % Z
        oc = offb[np.arange(no), [o.rc for o in res]][:, None]
        body = np.array([o.body for o in res], dtype=np.int64)[:, None]
        r = res, phb, rowid, scf, offb[:, None, :], isc[:, None, :], oc, body
        if key is not None:
            bcache[key] = r
        return r

    calls = [0] * (n + 1)
    spent = [0.0]

    def gen(tv: list, i: int, mem: list, pos: int) -> list:
        calls[i] += 1
        if spent[0] > GEN_BUDGET or calls[i] > top:
            return []
        t0 = time.perf_counter()
        out = run(i, mem, pos)
        spent[0] += time.perf_counter() - t0
        return out

    def run(i: int, mem: list, pos: int) -> list:
        if i not in cache:
            cache[i] = build(i)
        bt = cache[i]
        if bt is None or calls[i] > 1 and (len(bt[0]) > TOP2_MAXOPT or spent[0] > GEN_BUDGET / 2):
            return []
        res, phb, rowid, scf, offb, isc, oc, body = bt
        vals = np.array(mem, dtype=np.int64)[phb]
        cst = scf[rowid, vals]
        need = (cst > 0) | isc
        lo = np.where(need, offb, 99).min(2)
        hi = np.where(need, offb, -99).max(2)
        span = hi - lo
        mdp = MD[pos]
        tot = cst.sum(2) + np.minimum(mdp[(bs + sgs * lo) % Z] + hi - oc, mdp[(bs + sgs * hi) % Z] + oc - lo) + span + body
        bi = tot.argmin(1).tolist()
        bc = tot.min(1).tolist()
        u0 = mem[pos]
        cands = []
        for x, o in enumerate(res):
            c = bc[x]
            if not o.cnt:
                if o.spi <= scap[o.ms[0] < 4] * o.step:
                    cands.append((c, x, o.ms[0]))
                continue
            u = mem[int(phb[x, bi[x], o.rc])]
            c -= CNT_C[o.ms[0]][u]
            for m in o.ms:
                if o.spi + abs(CNT_S[m][u]) <= scap[m < 4] * o.step:
                    cands.append((c + CNT_C[m][u], x, m))
        cands.sort()
        byj = {}
        out = []
        bestc = {}
        for c, x, m in cands:
            j = i + m * res[x].step
            bestc[j] = min(bestc.get(j, 999), c)
        dom, lim = {}, 999
        for j in sorted(bestc, reverse=True):
            dom[j] = lim
            lim = min(lim, bestc[j])
        for c, x, m in cands:
            o = res[x]
            j = i + m * o.step
            if c > CD[u0, tv[i]] + j - i + ps[j - 1] - ps[i] or c >= dom[j] + PARETO:
                continue
            cnt = byj.get(j, 0)
            if cnt >= keep:
                continue
            byj[j] = cnt + 1
            b = bi[x]
            out.append((j,) + (f1_code if o.kind == 1 else f2_code)(o, m, mem, pos, b % Z, 1 if b < Z else -1))
        return out

    return gen


def nav_codes(mem: list, pos: int) -> list:
    """Shortest code from pos to every zone using > < [>] [<] (scan loops jump to the next zero zone), ties broken
    by executed steps; returns (codes, steps) lists."""
    dist = [1 << 30] * Z
    code = [""] * Z
    dist[pos] = 0
    h = [(0, pos)]
    while h:
        d, x = heapq.heappop(h)
        if d > dist[x]:
            continue
        nb = [((x + 1) % Z, 1001, ">"), ((x - 1) % Z, 1001, "<")]
        if mem[x]:
            for dr, op in ((1, "[>]"), (-1, "[<]")):
                y, w = (x + dr) % Z, 3003
                while mem[y]:
                    y = (y + dr) % Z
                    w += 3
                nb.append((y, w, op))
        for y, w, op in nb:
            if d + w < dist[y]:
                dist[y] = d + w
                code[y] = code[x] + op
                heapq.heappush(h, (d + w, y))
    return code, [d % 1000 for d in dist]


def make_word_gen(tv: list):
    """Prints runs of nonzero zones already in memory: [.>] / [.<] print one run and stop on the next zero zone;
    [[.>].>] / [[.<].<] print zero-separated runs, each followed by a space, until two zero zones in a row."""
    n = len(tv)

    def gen(tv: list, i: int, mem: list, pos: int) -> list:
        res = []
        t0 = tv[i]
        if not t0 or 0 not in mem:
            return res
        nc = None
        for p in range(Z):
            if mem[p] != t0:
                continue
            for dr, op, op2 in ((1, "[.>]", "[[.>].>]"), (-1, "[.<]", "[[.<].<]")):
                q, k, runs = p, i, 0
                while True:
                    while mem[q] and k < n and mem[q] == tv[k]:
                        q = (q + dr) % Z
                        k += 1
                    if mem[q]:
                        break
                    runs += 1
                    if runs == 1 and k - i >= 2:
                        if nc is None:
                            nc, ns = nav_codes(mem, pos)
                        res.append((k, nc[p] + op, mem, q, ns[p] + 4 * (k - i)))
                    if k >= n or tv[k]:
                        break
                    k += 1
                    q = (q + dr) % Z
                    if not mem[q]:
                        if runs >= 2:
                            if nc is None:
                                nc, ns = nav_codes(mem, pos)
                            res.append((k, nc[p] + op2, mem, q, ns[p] + 4 * (k - i)))
                        break
        return res

    return gen


def make_stored_loop_gen(tv: list):
    """Repeats a stored run (optionally followed by a space) with a counter loop: the counter sits on the zero
    zone just before the run; the body enters the run, prints it with [.>] (or [.<]), prints the space from the
    terminating zero zone if needed and returns to the counter by scanning back (or by plain moves)."""
    n = len(tv)

    def gen(tv: list, i: int, mem: list, pos: int) -> list:
        res = []
        t0 = tv[i]
        if not t0:
            return res
        nc = None
        for p in range(Z):
            if mem[p] != t0:
                continue
            for dr, op in ((1, "[.>]"), (-1, "[.<]")):
                c = (p - dr) % Z
                if mem[c]:
                    continue
                q, L = p, 0
                while mem[q] and i + L < n and mem[q] == tv[i + L]:
                    q = (q + dr) % Z
                    L += 1
                if mem[q] or L < 2 or q == c:
                    continue
                sp = i + L < n and tv[i + L] == 0
                P = L + sp
                m = 1
                while i + (m + 1) * P <= n and tv[i + m * P:i + (m + 1) * P] == tv[i:i + P]:
                    m += 1
                if m < 2:
                    continue
                fw, bw = (">", "<") if dr == 1 else ("<", ">")
                back, bst = bw + "[" + bw + "]" + fw, 3 * L + 5
                if mem[(c - dr) % Z] or L + 1 <= len(back):
                    back, bst = bw * (L + 1), L + 1
                body = fw + op + "." * sp + back
                bst += 1 + 4 * L + sp
                if nc is None:
                    nc, ns = nav_codes(mem, pos)
                for u in range(1, 3):
                    it = min(m // u, 26)
                    if it < 2 and u > 1:
                        break
                    cc, dd = _loop_counter(0, it)
                    res.append((i + it * u * P, nc[c] + cc + "[" + body * u + dd + "]", mem, c,
                                ns[c] + len(cc) + it * (bst * u + len(dd) + 2)))
        return res

    return gen


# ---------------------------------------------------------------- memory setup (prefill)

def _counter_table() -> dict:
    """For each loop iteration count n: cheapest (cost, start value, step) of a counter that reaches 0 after n steps."""
    best = {}
    for v in range(1, A):
        for d in range(-13, 14):
            if d == 0:
                continue
            x, k = v, 0
            while True:
                x = (x - d) % A
                k += 1
                if x == 0 or k > A:
                    break
            if x == 0:
                c = int(CD[0, v]) + abs(d)
                if k not in best or c < best[k][0]:
                    best[k] = (c, v, d)
    return best


CNT = dict(sorted(_counter_table().items()))


def _mult_tables() -> tuple:
    """BA[n][t] = (cost, a) minimizing |a| + CD[a*n, t] (a = 0 allowed); BS[n][t] = same with fixup cost CC and
    a such that a*m != 0 (mod 27) for m = 1..n (zone stays nonzero during an n-iteration loop), None if impossible."""
    ar = np.arange(-13, 14)
    BA, BS = {}, {}
    for n in CNT:
        vals = (ar * n) % A
        c1 = np.abs(ar)[:, None] + CD[vals]
        ok = np.array([all((a * m) % A for m in range(1, n + 1)) for a in ar])
        c2 = np.where(ok[:, None], np.abs(ar)[:, None] + CC[vals], 1 << 20)
        BA[n] = [(int(c1[:, t].min()), int(ar[c1[:, t].argmin()])) for t in range(A)]
        BS[n] = [(int(c2[:, t].min()), int(ar[c2[:, t].argmin()])) if c2[:, t].min() < 1 << 20 else None for t in range(A)]
    return BA, BS


BA, BS = _mult_tables()


def _fixup_walks(code: str, mem: list, zones: dict) -> list:
    """Appends fixup walks (left end first or right end first) setting zones {rel zone: value} exactly."""
    fz = [z for z in zones if mem[z % Z] != zones[z]] + [0]
    res = []
    for order in ((min(fz), max(fz)), (max(fz), min(fz))):
        c2, p, m2 = code, 0, list(mem)
        for end in order:
            step = 1 if end > p else -1
            while p != end:
                p += step
                c2 += ">" if step > 0 else "<"
                if m2[p % Z] != zones.get(p, 0):
                    c2 += change_code(m2[p % Z], zones[p])
                    m2[p % Z] = zones[p]
        res.append((c2, m2, p % Z))
    return res


def setup_inits(layout: str) -> list:
    """Builds code that writes layout (letters, spaces = zero zones) into memory using one multiplication loop
    plus fixups; returns list of (code, mem, pos) initial states (the beam picks the best).
    Variant 1: counter on a zero zone of/next to the layout, body walks out and back on both sides.
    Variant 2: counter just left/right of the layout, body walks over the whole layout and returns with [<]> / [>]<
    (all layout zones, separators included, stay nonzero during the loop; separators are cleared afterwards)."""
    t = [ALPHA.index(c) for c in layout]
    L = len(t)
    res = []
    for cidx in [-1] + [k for k in range(L) if t[k] == 0] + [L]:
        zones = {k - cidx: t[k] for k in range(L) if k != cidx}
        lo, hi = min(min(zones), 0), max(max(zones), 0)
        best = None
        for n, (cc, v, d) in CNT.items():
            ba = BA[n]
            per = {z: ba[tz] for z, tz in zones.items()}
            tot = cc + 2 + sum(CDL[0][tz] for tz in zones.values())
            ext = []
            for sg, lim in ((-1, -lo), (1, hi)):
                acc, be = 0, (0, 0)
                for e in range(1, lim + 1):
                    z = sg * e
                    acc += 2 + (per[z][0] - CDL[0][zones[z]] if z in zones else 0)
                    if acc < be[0]:
                        be = (acc, e)
                tot += be[0]
                ext.append(be[1])
            if ext == [0, 0]:
                continue
            if best is None or tot < best[0]:
                eL, eR = ext
                best = (tot, n, v, d, eL, eR, {z: (per[z][1] if -eL <= z <= eR else 0) for z in per})
        if best is None:
            continue
        tot, n, v, d, eL, eR, az = best
        code = delta_code(0, v) + "["
        for z in range(1, eR + 1):
            code += ">" + delta_code(0, az.get(z, 0) % A)
        code += "<" * eR
        for z in range(-1, -eL - 1, -1):
            code += "<" + delta_code(0, az.get(z, 0) % A)
        code += ">" * eL + ("-" if d > 0 else "+") * abs(d) + "]"
        mem = [0] * Z
        for z, a in az.items():
            mem[z % Z] = (a * n) % A
        res += _fixup_walks(code, mem, zones)
    # variant 0: direct writes
    for sg in (1, -1):
        code, m2 = "", [0] * Z
        for k in range(L):
            z = sg * k if sg == 1 else -(L - 1 - k)
            code += change_code(0, t[k] if sg == 1 else t[L - 1 - k]) + (">" if sg == 1 else "<")
            m2[(sg * k) % Z] = t[k] if sg == 1 else t[L - 1 - k]
        res.append((code[:-1], m2, (sg * (L - 1)) % Z))
    # variant 2: scan return
    for sg in (1, -1):
        zones = {sg * (k + 1): t[k] for k in range(L)} if sg == 1 else {-(L - k): t[k] for k in range(L)}
        best = None
        for n, (cc, v, d) in CNT.items():
            tot, az, bs = cc + 2 + L + 4, {}, BS[n]
            for z, tz in zones.items():
                bb = bs[tz]
                if bb is None:
                    break
                tot += bb[0]
                az[z] = bb[1]
            else:
                if best is None or tot < best[0]:
                    best = (tot, n, v, d, az)
        if best is None:
            continue
        tot, n, v, d, az = best
        mv, back = (">", "[<]>") if sg == 1 else ("<", "[>]<")
        code = delta_code(0, v) + "["
        for k in range(1, L + 1):
            code += mv + delta_code(0, az[sg * k] % A)
        code += back + ("-" if d > 0 else "+") * abs(d) + "]"
        mem = [0] * Z
        for z, a in az.items():
            mem[z % Z] = (a * n) % A
        res += _fixup_walks(code, mem, zones)
    return res


def est_layout(phrase: str, layout: str, plain: float = 3) -> float:
    """Estimates the printing cost of phrase when layout is in memory: any stored run (from any start zone to the
    next zero, either direction) can be printed with [.>] / [.<] for 4 + exact navigation from the previous run end
    (+1 for a following space); other characters cost plain each."""
    mem = [0] * Z
    for k, ch in enumerate(layout):
        mem[k + 1] = ALPHA.index(ch)
    runs = []
    for p in range(1, len(layout) + 1):
        for dr in (1, -1):
            q, s = p, ""
            while mem[q]:
                s += ALPHA[mem[q]]
                q += dr
            if len(s) >= 2:
                runs.append((s, p, q))
    n = len(phrase)
    at = [[] for _ in range(n)]
    for r in runs:
        s = r[0]
        i = phrase.find(s)
        while i >= 0:
            at[i].append(r)
            i = phrase.find(s, i + 1)
    nav = {}
    dp = [dict() for _ in range(n + 1)]
    dp[0][-1] = 0
    for i in range(n):
        for z, c in dp[i].items():
            if c + plain < dp[i + 1].get(-1, 1 << 30):
                dp[i + 1][-1] = c + plain
            if not at[i]:
                continue
            if z >= 0 and z not in nav:
                nav[z] = [len(x) for x in nav_codes(mem, z)[0]]
            for w, p, q in at[i]:
                j = i + len(w)
                c2 = c + 4 + (3 if z < 0 else nav[z][p])
                if j < n and phrase[j] == " ":
                    j, c2 = j + 1, c2 + 1
                if c2 < dp[j].get(q, 1 << 30):
                    dp[j][q] = c2
    return min(dp[n].values())


def dict_layouts(phrase: str, max_layouts: int = 6, thresh: float = 0, ret_scores: bool = False, plain: float = 3) -> list:
    """Picks sets of frequent space-free substrings worth storing in memory (zero separated, printed with [.>]);
    returns candidate layouts (spaces = zero zones; words in every order and orientation), most promising first."""
    n = len(phrase)
    occ = {}
    for L in range(2, 17):
        for i in range(n - L + 1):
            s = phrase[i:i + L]
            if " " not in s:
                occ.setdefault(s, []).append(i)
    words = []
    for s, lst in occ.items():
        if len(lst) < 2:
            continue
        lc = {phrase[i - 1] if i else " " for i in lst}
        rc = {phrase[i + len(s)] if i + len(s) < n else " " for i in lst}
        if (len(lc) < 2 and " " not in lc) or (len(rc) < 2 and " " not in rc):
            continue
        c, last = 0, -1
        for i in lst:
            if i >= last:
                c += 1
                last = i + len(s)
        if c >= 2 and 3 * len(s) > 9:
            words.append((c * (3 * len(s) - 9), s))
    words = [w for _, w in sorted(words, reverse=True)[:16]]
    starts = {w: set(occ[w]) for w in words}

    def est(ws: tuple) -> float:
        dp = [0] + [1 << 30] * n
        for i in range(n):
            c = dp[i]
            if c + 3 < dp[i + 1]:
                dp[i + 1] = c + 3
            for w in ws:
                if i in starts[w]:
                    j = i + len(w)
                    if c + 9 < dp[j]:
                        dp[j] = c + 9
                    if j < n and phrase[j] == " " and c + 10 < dp[j + 1]:
                        dp[j + 1] = c + 10
        return dp[n] + 5.5 * sum(map(len, ws)) + 6 - 3 * n

    dicts = [(w,) for w in words] + [(a, b) for k, a in enumerate(words) for b in words[k + 1:]] +             [(a, b, c) for k, a in enumerate(words[:9]) for m, b in enumerate(words[k + 1:9]) for c in words[k + m + 2:9]]
    dicts = [d for d in dicts if sum(map(len, d)) + len(d) <= 25 and all(x not in y for x in d for y in d if x != y)]
    scored = [d for g, d in sorted((est(d), d) for d in dicts)[:12] if g < thresh]
    scored = sorted((est_layout(phrase, " ".join(d), plain) + 5.5 * sum(map(len, d)), d) for d in scored)[:4]
    cands = []
    for _, d in scored:
        lays = set()
        for perm in itertools.permutations(d):
            for flips in itertools.product((1, -1), repeat=len(d)):
                lay = " ".join(w[::f] for w, f in zip(perm, flips))
                if lay[::-1] not in lays:
                    lays.add(lay)
        setup = 5.5 * sum(map(len, d)) + 6
        cands += [(est_layout(phrase, lay, plain) + setup, lay) for lay in lays]
    cands.sort()
    if ret_scores:
        return cands[:max_layouts]
    return [lay for _, lay in cands[:max_layouts]]


def palette_layouts(phrase: str, top: int = 3) -> list:
    """For texts over few distinct letters: arrangements of the letters (one zone each, optionally one zero zone
    inside for spaces) minimizing the total move distance between consecutive characters."""
    letters = sorted(set(phrase) - {" "})
    k = len(letters)
    if not 2 <= k <= 6:
        return []
    cnt = {}
    for a, b in zip(phrase, phrase[1:]):
        if a != b:
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    res = []
    for perm in itertools.permutations(letters):
        if perm[0] > perm[-1]:
            continue
        for sp in ([None] + list(range(1, k))) if " " in phrase else [None]:
            lay = list(perm) if sp is None else list(perm[:sp]) + [" "] + list(perm[sp:])
            pos = {ch: i for i, ch in enumerate(lay) if ch != " "}
            zeros = [-1, len(lay)] + ([sp] if sp is not None else [])
            c = 0
            for (a, b), w in cnt.items():
                if a == " " or b == " ":
                    c += w * min(abs(pos[b if a == " " else a] - z) for z in zeros)
                else:
                    c += w * abs(pos[a] - pos[b])
            res.append((c, "".join(lay)))
    res.sort()
    pure = [lay for _, lay in res[:12]]

    def dp_cost(lay: str) -> int:
        zs = {}
        for k, ch in enumerate(lay):
            zs.setdefault(ch, []).append(k)
        zs.setdefault(" ", []).extend([-1, len(lay)])
        prev = {z: 0 for z in zs[phrase[0]]}
        for ch in phrase[1:]:
            prev = {z: min(v + abs(z - p) for p, v in prev.items()) for z in zs[ch]}
        return min(prev.values())

    dup = set()
    if k <= 5:
        for lay in pure:
            for ch in letters:
                for p in range(len(lay) + 1):
                    dup.add(lay[:p] + ch + lay[p:])
    dup = [lay for _, lay in sorted((dp_cost(lay) + 3.5 * len(lay.replace(" ", "")), lay) for lay in dup)]
    return list(dict.fromkeys(pure[:top - 1] + dup[:2]))


# ---------------------------------------------------------------- search

def _setup_counters() -> dict:
    """For each iteration count k: cheapest (cost, init code, step code) of a counter started from 0."""
    best = {}
    for v in range(1, A):
        for d in range(-13, 14):
            if d % 3 == 0:
                continue
            k = next(j for j in range(1, A + 1) if (v - j * d) % A == 0)
            c = int(CD[0, v]) + abs(d)
            if k not in best or c < best[k][0]:
                best[k] = (c, delta_code(0, v), ("-" if d > 0 else "+") * abs(d))
    return best


SCT = _setup_counters()


def setup_states(tv: list, win: int, mmax: int, amax: int) -> list:
    """Candidate start states [(code, mem, pos)] produced by one multiplication loop from the empty forest: a counter
    zone runs k iterations, each adding a_j to the zones around it, so they end at k * a_j (letters of tv[:win])."""
    w = tv[:win]
    letters = [c for c in dict.fromkeys(w) if c != 0]
    first = {c: w.index(c) for c in letters}
    res = {}
    for k, (cc, init, dec) in SCT.items():
        opts = []
        for L in letters:
            best = min((abs(a) + 2 + int(CD[(k * a) % A, L]), a) for a in range(-amax, amax + 1) if a)
            if CD[0, L] + 1 - best[0] > 0:
                opts.append((CD[0, L] + 1 - best[0], first[L], best[1]))
        opts.sort(key=lambda o: (-o[0], o[1]))
        for m in range(1, min(mmax, len(opts)) + 1):
            sel = [a for _, _, a in sorted(opts[:m], key=lambda o: o[1])]
            for q in range(m + 1):
                # zones: sel[0..q-1] left of the counter (sel[q-1] adjacent), sel[q..] right of it
                offs = list(range(-q, 0)) + list(range(1, m - q + 1))
                body = "".join("<" + delta_code(0, sel[j] % A) for j in range(q - 1, -1, -1))
                if m > q:
                    body += ">" * (q + 1) + delta_code(0, sel[q] % A) + "".join(">" + delta_code(0, sel[j] % A) for j in range(q + 1, m))
                    body += "<" * (m - q)
                else:
                    body += ">" * q
                code = init + "[" + body + dec + "]"
                mem = [0] * Z
                for o, a in zip(offs, sel):
                    mem[o % Z] = (k * a) % A
                key = tuple(mem)
                if key not in res or len(code) < len(res[key][0]):
                    res[key] = (code, mem, 0)
    return list(res.values())


def greedy_cost(tv: list, mem: np.ndarray, pos: np.ndarray, upto: int) -> np.ndarray:
    """Cost of greedily printing tv[:upto] (cheapest zone per char) from each state."""
    mem = mem.copy()
    tot = np.zeros(len(pos), dtype=np.int64)
    r = np.arange(len(pos))
    for t in tv[:upto]:
        c = MD[pos] + CC[mem, t]
        z = c.argmin(1)
        tot += c[r, z]
        mem[r, z] = t
        pos = z
    return tot


def state_aux(mem: np.ndarray) -> tuple:
    """Hash sums S1, S2 and zero masks of the states mem (B,30)."""
    w = HW[mem]
    return (w * PW % HQ).sum(1) % HQ, (w * PWI % HQ).sum(1) % HQ, (mem == 0) @ ZB


def move_idx(zm: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Row index into MT / MS for zero masks zm and pointers pos."""
    rm = ((zm >> pos) | (zm << (Z - pos))) & FULL
    z0 = rm & 1
    r1 = rm & (FULL - 1)
    has = r1 != 0
    kr = np.frexp((r1 & -r1).astype(np.float64))[1] - 1
    kl = Z + 1 - np.frexp(r1.astype(np.float64))[1]
    rc = np.where(has & ((z0 == 0) | (kr >= 2)), kr + 30 * z0, 60)
    lc = np.where(has & ((z0 == 0) | (kl >= 2)), kl + 30 * z0, 60)
    return (pos * 61 + rc) * 61 + lc


def group_rank(keys: np.ndarray) -> np.ndarray:
    """Rank of every element among the earlier elements with the same key."""
    so = np.argsort(keys, kind="stable")
    ks = keys[so]
    st = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    rank = np.empty(len(keys), dtype=np.int64)
    rank[so] = np.arange(len(keys)) - np.repeat(st, np.diff(np.r_[st, len(keys)]))
    return rank


def rollout(tv: list, i: int, look: int, mem: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Greedy cost (plain moves, cheapest zone per char) of printing tv[i:i+look] from each state (mem is not modified)."""
    r = np.arange(len(pos))
    us = tv[i:i + look]
    if len(us) > 2:
        mem = mem.copy()
    tot = np.zeros(len(pos), dtype=np.int64)
    prev = -1
    for k, t in enumerate(us):
        c = MD8.take(pos, axis=0) + CCT[t].take(mem)
        if k == 1 and len(us) == 2:
            c[r, pos] = CC[prev, t]
        z = c.argmin(1)
        tot += c[r, z]
        if len(us) > 2:
            mem[r, z] = t
        prev, pos = t, z
    return tot


def solve(phrase: str, width: int, use_macros: bool = True, macro_top: int = 4, init: list = None, word_top: int = 4,
          setups: bool = True, n_setups: int = 30, quota: int = 16, horizon: int = 40, kmul: float = 2, ke: int = 10,
          look: int = 2, rmul: float = 2, deadline: float = None, target: float = None, wmax: int = 3000) -> str:
    """Beam search over the phrase index. init: optional list of (code, mem, pos) start states; without it the empty
    forest plus up to n_setups multiplication-loop start states (setups) are used; while i < horizon each start state
    keeps a quota of beam slots. Macro generators are (gen, top) pairs called for the top cheapest states of each level.
    Executed steps are tracked per state: states that cannot finish below STEP_LIMIT are dropped and the state with the
    fewest steps is always kept. Beyond the start phase, the width best states are picked by cost plus a greedy
    rollout of the next look characters among rmul * width cheapest. Returns None if no state survives or the
    perf_counter deadline passes. With a target (perf_counter time) the width is re-chosen at every level, between
    20 and wmax, so that the projected finish (measured generator time per level plus core time per beam slot, both
    smoothed) meets the target."""
    tv = [ALPHA.index(ch) for ch in phrase]
    n = len(tv)
    gens = [(make_repeat_gen(tv), macro_top), (make_prog_gen(tv), macro_top), (make_word_gen(tv), word_top),
            (make_stored_loop_gen(tv), word_top)] if use_macros else []
    gtop = max((g[1] for g in gens), default=0)
    start = [("", [0] * Z, 0)] if init is None else list(init)
    if setups and init is None:
        cand = setup_states(tv, 40, 8, 4)
        if cand:
            cm = np.array([c[1] for c in cand], dtype=np.int8)
            sc = np.array([len(c[0]) for c in cand]) + greedy_cost(tv, cm, np.zeros(len(cand), dtype=np.int64), horizon)
            start += [cand[j] for j in np.argsort(sc, kind="stable")[:n_setups]]
    S = len(start)
    m0 = np.array([s[1] for s in start], dtype=np.int8)
    # level data: mem, pos, cost, par_lvl, par_idx, code_id, codes (macro codes; at level 0 the start codes), steps
    levels = [None] * (n + 1)
    aux = [None] * (n + 1)  # per level: hash sums S1, S2, zero masks, origin (start state index)
    pending = [[] for _ in range(n + 1)]
    levels[0] = (m0, np.array([s[2] for s in start], dtype=np.int64), np.array([len(s[0]) for s in start], dtype=np.int64),
                 np.full(S, -1), np.full(S, -1), np.arange(S), [s[0] for s in start],
                 np.array([simulate(s[0])[3] for s in start], dtype=np.int64))
    aux[0] = state_aux(m0) + (np.arange(S),)
    t_lvl, gt, g_rate, c_rate = time.perf_counter(), 0.0, None, None  # level start, generator time, smoothed rates
    for i in range(n):
        now = time.perf_counter()
        if deadline is not None and now > deadline:
            return None
        if target is not None and i:
            cd = (now - t_lvl - gt) / width
            g_rate, c_rate = (gt, cd) if g_rate is None else (0.8 * g_rate + 0.2 * gt, 0.8 * c_rate + 0.2 * cd)
            if i >= 3:
                width = int(min(wmax, max(20, (target - now - g_rate * (n - i)) / (c_rate * (n - i)))))
        t_lvl, gt = now, 0.0
        mem, pos, cost, st = levels[i][0], levels[i][1], levels[i][2], levels[i][7]
        S1, S2, zm, org = aux[i]
        B = len(pos)
        if gens:
            order = np.argsort(cost, kind="stable")[:gtop]
            for r, b in enumerate(order):
                ml, pp = mem[b].tolist(), int(pos[b])
                for g, top in gens:
                    if r >= top:
                        continue
                    tg = time.perf_counter()
                    for res in g(tv, i, ml, pp):
                        j, code, nm, np_ = res[:4]
                        if len(res) > 4:
                            s = res[4]
                        else:
                            sim = simulate(code, ml, pp)
                            if sim is None:
                                continue
                            s = sim[3]
                        pending[j].append((int(cost[b]) + len(code), nm, np_, i, int(b), code, int(st[b]) + s, int(org[b])))
                    gt += time.perf_counter() - tg
        t = tv[i]
        early = S > 1 and i < horizon
        mi = move_idx(zm, pos)
        tot = (MT.take(mi, axis=0) + CCT[t].take(mem)) + (cost[:, None] + 1).astype(np.int32)
        flat = tot.ravel()
        if early and ke < Z:
            cand = (np.argpartition(tot, ke - 1, 1)[:, :ke] + (np.arange(B) * Z)[:, None]).ravel()
        else:
            k = int(kmul * width)
            cand = np.argpartition(flat, k - 1)[:k] if k < len(flat) and not early else np.arange(len(flat))
        par, zz = cand // Z, cand % Z
        ncost = flat[cand].astype(np.int64)
        old = mem[par, zz]
        nst = st[par] + MS[mi[par], zz] + CST[t][old] + 1
        dw = (HW[t] - HW[old]) % HQ
        nS1 = (S1[par] + dw * PW[zz]) % HQ
        nS2 = (S2[par] + dw * PWI[zz]) % HQ
        nzm = zm[par] | ZB[zz] if t == 0 else zm[par] & ~ZB[zz]
        npos = zz
        plvl = np.full(len(par), i)
        cid = np.full(len(par), -1)
        norg = org[par]
        codes = []
        pend = pending[i + 1]
        pending[i + 1] = None
        nk = len(cand)
        if pend:
            pend.sort(key=lambda r: r[0])
            pend = pend[:int(kmul * width)]
            codes = [r[5] for r in pend]
            pm = np.array([r[1] for r in pend], dtype=np.int8)
            a1, a2, az = state_aux(pm)
            ncost = np.concatenate([ncost, np.array([r[0] for r in pend], dtype=np.int64)])
            nst = np.concatenate([nst, np.array([r[6] for r in pend], dtype=np.int64)])
            nS1, nS2, nzm = np.concatenate([nS1, a1]), np.concatenate([nS2, a2]), np.concatenate([nzm, az])
            npos = np.concatenate([npos, np.array([r[2] for r in pend], dtype=np.int64)])
            plvl = np.concatenate([plvl, np.array([r[3] for r in pend])])
            par = np.concatenate([par, np.array([r[4] for r in pend])])
            cid = np.concatenate([cid, np.arange(len(pend))])
            norg = np.concatenate([norg, np.array([r[7] for r in pend])])
        ok = np.flatnonzero(nst + STEP_RESERVE * (n - i - 1) < STEP_LIMIT)
        if not len(ok):
            return None
        ms = ok[np.argmin(nst[ok])]
        h = np.minimum(nS1[ok] * PWI[npos[ok]] % HQ, nS2[ok] * PW[npos[ok]] % HQ)
        o = np.argsort(h * (1 << 22) + ncost[ok])
        hs = h[o]
        u = ok[o[np.r_[True, hs[1:] != hs[:-1]]]]
        use_look = look and i + 1 < n and not early and len(u) > width
        if early:
            u = u[np.argsort(ncost[u], kind="stable")]
            u = u[(np.arange(len(u)) < width) | (group_rank(norg[u]) < quota)]
        elif use_look:
            R = int(rmul * width)
            if len(u) > R:
                u = u[np.argpartition(ncost[u], R - 1)[:R]]
        elif len(u) > width:
            u = u[np.argpartition(ncost[u], width - 1)[:width]]
        isp = u < nk
        nmem = np.empty((len(u), Z), dtype=np.int8)
        ps = u[isp]
        nmem[isp] = mem[par[ps]]
        nmem[np.flatnonzero(isp), zz[ps]] = t
        if not isp.all():
            nmem[~isp] = pm[u[~isp] - nk]
        if use_look:
            f = ncost[u] + rollout(tv, i + 1, look, nmem, npos[u])
            oo = np.argpartition(f, width - 1)[:width]
            u, nmem = u[oo], nmem[oo]
        if ms not in u:
            u = np.append(u, ms)
            nmem = np.concatenate([nmem, (pm[ms - nk] if ms >= nk else np.where(AR == zz[ms], t, mem[par[ms]]))[None, :].astype(np.int8)])
        levels[i + 1] = (nmem, npos[u], ncost[u], plvl[u], par[u], cid[u], codes, nst[u])
        aux[i + 1] = (nS1[u], nS2[u], nzm[u], norg[u])
    b = int(np.argmin(levels[n][2]))
    lv = n
    out = []
    while lv > 0:
        _, pos, _, plvl, par, cid, codes = levels[lv][:7]
        pl, pb = int(plvl[b]), int(par[b])
        if cid[b] >= 0:
            out.append(codes[cid[b]])
        else:
            out.append(plain_code(levels[pl][0][pb], int(levels[pl][1][pb]), int(pos[b]), tv[pl]))
        lv, b = pl, pb
    out.append(levels[0][6][levels[0][5][b]])
    return "".join(reversed(out))


TIME_BUDGET = 0.8  # seconds of compute after startup


def valid(prog: str, phrase: str) -> bool:
    if prog is None:
        return False
    r = simulate(prog)
    return r is not None and r[0] == phrase


def main():
    t0 = time.perf_counter()
    phrase = sys.stdin.readline().rstrip("\r\n")
    end = t0 + TIME_BUDGET
    best = None
    # plain text (no letter palette, no dictionary worth storing): one beam run whose width fills the time budget
    if not palette_layouts(phrase) and not dict_layouts(phrase, plain=3.4):
        prog = solve(phrase, 300, True, target=end - 0.06, deadline=end + 0.3)
        if not valid(prog, phrase):
            prog = solve(phrase, 100, False)
        print(prog)
        return
    for use in (True, False):
        prog = solve(phrase, 300 if use else 100, use, deadline=t0 + 1.0 if use else None)
        if valid(prog, phrase):
            best = prog
            break
    t_base = time.perf_counter() - t0
    # memory prefill: a zero-separated dictionary of frequent words (printed with [.>]) or a letter palette,
    # written by a multiplication loop, then the beam continues from that memory
    lays = list(dict.fromkeys(dict_layouts(phrase, plain=len(best) / len(phrase)) + palette_layouts(phrase)))
    best_lay, est = None, t_base * 0.6
    for lay in lays:
        if time.perf_counter() - t0 + est > TIME_BUDGET:
            break
        ts = time.perf_counter()
        prog = solve(phrase, 100, True, init=setup_inits(lay), word_top=16, deadline=end - 0.03)
        est = time.perf_counter() - ts
        if prog is not None and len(prog) < len(best) and valid(prog, phrase):
            best, best_lay = prog, lay
    # remaining time: a wider beam from the best prefill (or from scratch)
    if time.perf_counter() + 1.5 * (est if best_lay is not None else t_base) < end:
        prog = solve(phrase, 300, True, init=setup_inits(best_lay) if best_lay is not None else None, word_top=16 if best_lay else 4,
                     target=end - 0.06, deadline=end + 0.05)
        if prog is not None and len(prog) < len(best) and valid(prog, phrase):
            best = prog
    print(best)


if __name__ == "__main__":
    main()
