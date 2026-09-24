#!/usr/bin/env python3
"""Seabed Security bot "denial": exact monster avoidance with multi-turn lookahead, radar/mirror/particle tracking,
race-aware surfacing and fish herding (denial of the opponent's missing fish)."""
import sys
import math
import time
import random
import traceback

INF = float("inf")
HAB = ((2500, 5000), (5000, 7500), (7500, 9999))
NPART = 40
RNG = random.Random(987654321)
R1 = 520 ** 2        # collision radius (with margin) for the first (exact) turn
R2 = 560 ** 2        # collision radius for lookahead turns
DIR32 = [(math.cos(k * math.pi / 16), math.sin(k * math.pi / 16)) for k in range(32)]
ESC = [(int(600 * math.cos(k * math.pi / 8)), int(600 * math.sin(k * math.pi / 8))) for k in range(16)] + [(0, 300)]
TIME_BUDGET = 0.025
NPF = 96
PRNG = random.Random(12345)
PRED_T = 10
import os
DUMP = os.environ.get("DENIAL_DUMP")


def jround(v):
    return math.floor(round(v * 1e7) / 1e7 + 0.5)


DIR8 = [(jround(math.cos(k * math.pi / 4) * 200), jround(math.sin(k * math.pi / 4) * 200)) for k in range(8)]


def leg(x, y, cost, p, t0):
    """Moves from (x, y) to the scan disk of route point p = (fx, fy, R, vx, vy, lo, hi), predicting the fish position at
    the arrival time (t0 turns ahead of the cost origin). Returns the new position and accumulated cost."""
    fx, fy, R, vx, vy, lo, hi = p
    if vx or vy:
        t = t0 + (cost + max(0, math.hypot(fx - x, fy - y) - R)) / 600
        if t > PRED_T:
            t = PRED_T
        fx += vx * t
        fy += vy * t
        if fy < lo:
            fy = 2 * lo - fy
        elif fy > hi:
            fy = 2 * hi - fy
        if fx < 0:
            fx = -fx
        elif fx > 9999:
            fx = 19998 - fx
    dx, dy = fx - x, fy - y
    dd = math.hypot(dx, dy)
    if dd > R:
        s = (dd - R) / dd
        return x + dx * s, y + dy * s, cost + dd - R
    return x, y, cost


def rl():
    s = sys.stdin.readline()
    if not s:
        raise EOFError
    return s


def surf_turns(y):
    return 0 if y <= 500 else (y - 500 + 599) // 600


def box_d2(x0, x1, y0, y1, px, py):
    dx = x0 - px if px < x0 else (px - x1 if px > x1 else 0)
    dy = y0 - py if py < y0 else (py - y1 if py > y1 else 0)
    return dx * dx + dy * dy


def collides(ax, ay, avx, avy, mx, my, mvx, mvy, r2):
    dx = mx - ax
    dy = my - ay
    c = dx * dx + dy * dy - r2
    if c <= 0:
        return True
    vx = mvx - avx
    vy = mvy - avy
    a = vx * vx + vy * vy
    if a <= 0:
        return False
    b = 2 * (dx * vx + dy * vy)
    delta = b * b - 4 * a * c
    if delta < 0:
        return False
    t = (-b - math.sqrt(delta)) / (2 * a)
    return 0 < t <= 1


def mon_vel(x, y, vx, vy, tg, others):
    """Next monster speed per referee rules. tg: eligible drones (x, y, r2) at end of turn; others: other monster positions.
    Returns vx, vy, target x (or None), target y."""
    bd = -1
    cnt = 0
    sx = sy = 0
    for dx, dy, r2 in tg:
        dd = (dx - x) ** 2 + (dy - y) ** 2
        if dd <= r2:
            if bd < 0 or dd < bd:
                bd, sx, sy, cnt = dd, dx, dy, 1
            elif dd == bd:
                sx += dx
                sy += dy
                cnt += 1
    if cnt:
        tx, ty = sx / cnt, sy / cnt
        ddx, ddy = tx - x, ty - y
        l = math.hypot(ddx, ddy)
        if l > 540:
            ddx, ddy = ddx / l * 540, ddy / l * 540
        return jround(ddx), jround(ddy), tx, ty
    l = math.hypot(vx, vy)
    if l > 270:
        vx, vy = jround(vx / l * 270), jround(vy / l * 270)
    if vx or vy:
        bd = -1
        cnt = 0
        sx = sy = 0
        for ox, oy in others:
            dd = (ox - x) ** 2 + (oy - y) ** 2
            if bd < 0 or dd < bd:
                bd, sx, sy, cnt = dd, ox, oy, 1
            elif dd == bd:
                sx += ox
                sy += oy
                cnt += 1
        if cnt and bd <= 360000:
            ax, ay = x - sx / cnt, y - sy / cnt
            l = math.hypot(ax, ay)
            if l > 0:
                vx, vy = jround(ax / l * 200), jround(ay / l * 200)
    px, py = x + vx, y + vy
    if (px < 0 and px < x) or (px > 9999 and px > x):
        vx = -vx
    if (py < 2500 and py < y) or (py > 9999 and py > y):
        vy = -vy
    return vx, vy, None, None


def fish_vel(x, y, vx, vy, low, high, motors, others):
    """Next fish speed per referee rules. motors: motor-on drone positions; others: other fish positions."""
    bd = -1
    cnt = 0
    sx = sy = 0
    for dx, dy in motors:
        dd = (dx - x) ** 2 + (dy - y) ** 2
        if bd < 0 or dd < bd:
            bd, sx, sy, cnt = dd, dx, dy, 1
        elif dd == bd:
            sx += dx
            sy += dy
            cnt += 1
    if cnt and bd <= 1960000:
        ax, ay = x - sx / cnt, y - sy / cnt
        l = math.hypot(ax, ay)
        if l > 0:
            return jround(ax / l * 400), jround(ay / l * 400)
        return 0, 0
    l = math.hypot(vx, vy)
    fx, fy = (vx / l * 200, vy / l * 200) if l > 0 else (0, 0)
    bd = -1
    cnt = 0
    sx = sy = 0
    for ox, oy in others:
        dd = (ox - x) ** 2 + (oy - y) ** 2
        if bd < 0 or dd < bd:
            bd, sx, sy, cnt = dd, ox, oy, 1
        elif dd == bd:
            sx += ox
            sy += oy
            cnt += 1
    if cnt and bd <= 360000:
        ax, ay = x - sx / cnt, y - sy / cnt
        l = math.hypot(ax, ay)
        fx, fy = (ax / l * 200, ay / l * 200) if l > 0 else (0, 0)
    px, py = x + fx, y + fy
    if (px < 0 and px < x) or (px > 9999 and px > x):
        fx = -fx
    if (py < low and py < y) or (py > high and py > y):
        fy = -fy
    return jround(fx), jround(fy)


class Fish:
    __slots__ = ("id", "col", "typ", "low", "high", "x", "y", "vx", "vy", "exact", "bx0", "bx1", "by0", "by1",
                 "twin", "alive", "samples", "P", "obs", "std", "rb", "negs", "poss")

    def __init__(self, i, c, t):
        self.id, self.col, self.typ = i, c, t
        self.low, self.high = HAB[t]
        self.x, self.y, self.vx, self.vy = 5000, self.low + 1250, 0, 0
        self.exact = False
        self.bx0, self.bx1, self.by0, self.by1 = 1000, 8999, self.low + 1000, self.low + 1500
        self.twin = None
        self.alive = True
        self.samples = [(self.x, self.y)]
        self.P = None
        self.obs = False
        self.std = 1000


class Mon:
    __slots__ = ("id", "x", "y", "vx", "vy", "known", "bx0", "bx1", "by0", "by1", "twin", "parts", "still", "seen",
                 "spawn", "replay_ok")

    def __init__(self, i):
        self.id = i
        self.x, self.y, self.vx, self.vy = 5000, 7500, 0, 0
        self.known = False
        self.bx0, self.bx1, self.by0, self.by1 = 0, 9999, 5000, 9999
        self.twin = None
        self.parts = []
        self.still = True
        self.seen = -1
        self.spawn = None
        self.replay_ok = True


class Dr:
    __slots__ = ("id", "x", "y", "em", "bat", "mine", "light", "motor", "fmotor", "elig", "vx", "vy", "pem")


class Bot:
    def __init__(self):
        n = int(rl())
        self.fish = {}
        self.mons = {}
        for _ in range(n):
            i, c, t = map(int, rl().split())
            if t < 0:
                self.mons[i] = Mon(i)
            else:
                self.fish[i] = Fish(i, c, t)
        byct = {(f.col, f.typ): f for f in self.fish.values()}
        for f in self.fish.values():
            f.twin = byct.get((f.col ^ 1, f.typ))
        ids = sorted(self.mons)
        for k in range(0, len(ids) - 1, 2):
            a, b = self.mons[ids[k]], self.mons[ids[k + 1]]
            a.twin, b.twin = b, a
        fl = list(self.fish.values())
        self.combos = [(frozenset(f.id for f in fl if f.typ == k), 4) for k in range(3)] + \
            [(frozenset(f.id for f in fl if f.col == k), 3) for k in range(4)]
        self.turn = 0
        self.prev = {}
        self.cmd = {}
        self.saved_t = [{}, {}]
        self.surf = {}
        self.herd = {}
        self.routes = {}
        self.dbg = []
        self.mode = {}
        self.tg_hist = []
        self.pscans = {}

    # ------------------------------------------------------------------ parsing
    def parse(self):
        self.my_score = int(rl())
        self.tstart = time.perf_counter()
        self.foe_score = int(rl())
        self.my_saved = set(int(rl()) for _ in range(int(rl())))
        self.foe_saved = set(int(rl()) for _ in range(int(rl())))
        self.my_list = [tuple(map(int, rl().split())) for _ in range(int(rl()))]
        self.foe_list = [tuple(map(int, rl().split())) for _ in range(int(rl()))]
        self.scans = {}
        for _ in range(int(rl())):
            a, b = map(int, rl().split())
            self.scans.setdefault(a, set()).add(b)
        self.vis = {}
        for _ in range(int(rl())):
            a = list(map(int, rl().split()))
            self.vis[a[0]] = a[1:]
        self.radar = {}
        for _ in range(int(rl())):
            a, b, r = rl().split()
            self.radar[(int(a), int(b))] = r

    # ------------------------------------------------------------------ tracking
    def update(self):
        t = self.turn
        for p, s in ((0, self.my_saved), (1, self.foe_saved)):
            for f in s:
                self.saved_t[p].setdefault(f, t - 1)
        self.dr = {}
        for lst, mine in ((self.my_list, True), (self.foe_list, False)):
            for i, x, y, em, bat in lst:
                d = Dr()
                d.id, d.x, d.y, d.em, d.bat, d.mine = i, x, y, em, bat, mine
                p = self.prev.get(i)
                if p:
                    px, py, pem, pbat = p
                    d.light = bat == pbat - 5
                    d.vx, d.vy = x - px, y - py
                    if mine:
                        d.motor = self.cmd.get(i, False) and not pem
                    else:
                        d.motor = not pem and not (d.vx == 0 and d.vy == min(300, 9999 - py))
                    d.pem = pem
                else:
                    d.light = d.motor = False
                    d.vx = d.vy = 0
                    d.pem = 0
                d.fmotor = d.motor and not em
                d.elig = not d.pem and not em
                self.dr[i] = d
        self.my_ids = [a[0] for a in self.my_list]
        self.foe_ids = [a[0] for a in self.foe_list]
        self.prev = {i: (d.x, d.y, d.em, d.bat) for i, d in self.dr.items()}
        self.my = [self.dr[i] for i in self.my_ids]
        self.foe = [self.dr[i] for i in self.foe_ids]
        self.update_fish()
        self.update_mons()

    def radar_box(self, cid):
        x0, x1, y0, y1 = 0, 9999, 0, 10000
        for d in self.my:
            r = self.radar.get((d.id, cid))
            if r is None:
                continue
            if r[0] == "T":
                y1 = min(y1, d.y)
            else:
                y0 = max(y0, d.y + 1)
            if r[1] == "L":
                x1 = min(x1, d.x)
            else:
                x0 = max(x0, d.x + 1)
        return x0, x1, y0, y1

    def update_fish(self):
        t = self.turn
        alive = {c for (_, c) in self.radar if c in self.fish}
        for f in self.fish.values():
            if f.alive and f.id not in alive:
                f.alive = False
                f.P = None
        fl = [f for f in self.fish.values() if f.alive]
        self.fl = fl
        vis = self.vis
        circ = [(d.x, d.y, (2000 if d.light else 800) ** 2) for d in self.my]
        fcon = [(e.x, e.y, (2000 if e.light else 800) ** 2, self.scans.get(e.id, ()), self.pscans.get(e.id, ()))
                for e in self.foe if e.elig and t > 0]
        fsaved = self.foe_saved
        for f in fl:
            x0, x1, y0, y1 = self.radar_box(f.id)
            f.rb = (max(x0, 0), min(x1, 9999), max(y0, f.low), min(y1, f.high))
            negs = [] if f.id in vis else list(circ)
            poss = []
            if f.id not in fsaved:
                for ex, ey, r2, sc, psc in fcon:
                    if f.id in sc:
                        if f.id not in psc:
                            poss.append((ex, ey, r2))
                    else:
                        negs.append((ex, ey, r2))
            f.negs, f.poss = negs, poss
        try:
            self.pf_step(fl, t)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            for f in fl:
                f.obs = False
                try:
                    f.P = None
                    f.P = self.pf_seed(f)
                except Exception:
                    x0, x1, y0, y1 = f.rb
                    f.P = [[(x0 + x1) // 2, (y0 + y1) // 2, 0, 0, 300] for _ in range(NPF)]
                self.pf_stats(f)
        self.pscans = self.scans

    def pf_step(self, fl, t):
        """Advances the fish particle filter one turn: init/propagate, constrain and resample, velocity update, stats."""
        if t == 0:
            self.pf_init(fl)
        else:
            for f in fl:
                if f.P is None:
                    f.P = self.pf_seed(f)
                    continue
                lo, hi = f.low, f.high
                for p in f.P:
                    p[0] += p[2]
                    y = p[1] + p[3]
                    p[1] = lo if y < lo else (hi if y > hi else y)
        self.pf_motors = [(d.x, d.y) for d in self.dr.values() if d.fmotor]
        done = set()
        for f in fl:
            if f.id in done:
                continue
            done.add(f.id)
            g = f.twin
            if g is not None and g.alive:
                done.add(g.id)
                self.pf_pair(f, g)
            else:
                self.pf_single(f)
        motors = [(d.x, d.y) for d in self.dr.values() if d.fmotor]
        self.pf_vel(fl, motors)
        for f in fl:
            self.pf_stats(f)

    def pf_init(self, fl):
        n = NPF
        done = set()
        for f in fl:
            if f.id in done:
                continue
            g = f.twin
            done.add(f.id)
            x0, x1, y0, y1 = f.rb
            y0, y1 = max(y0, f.low + 1000), min(y1, f.low + 1499)
            if g is not None and g.alive:
                done.add(g.id)
                x0, x1 = max(x0, 9999 - g.rb[1]), min(x1, 9999 - g.rb[0])
                y0, y1 = max(y0, g.rb[2]), min(y1, g.rb[3])
            segs = [(max(x0, a), min(x1, b)) for a, b in ((1000, 3999), (6001, 8999)) if max(x0, a) <= min(x1, b)]
            if not segs or y0 > y1:
                segs, y0, y1 = [(max(f.rb[0], 1000), min(f.rb[1], 8999))], f.low + 1000, f.low + 1499
                if segs[0][0] > segs[0][1]:
                    segs = [(1000, 8999)]
            tot = sum(b - a + 1 for a, b in segs)
            c = math.sqrt(tot * (y1 - y0 + 1) * 8 / n) / 2
            P, Q = [], []
            for k in range(n):
                u = (k // 8 + PRNG.random()) / (n // 8) * tot
                for a, b in segs:
                    if u < b - a + 1:
                        x = a + int(u)
                        break
                    u -= b - a + 1
                else:
                    x = segs[-1][1]
                y = PRNG.randint(y0, y1)
                vx, vy = DIR8[k % 8]
                P.append([x, y, vx, vy, c])
                Q.append([9999 - x, y, -vx, vy, c])
            f.P = P
            f.obs = False
            if g is not None and g.alive:
                g.P = Q
                g.obs = False

    def pf_ok(self, f, P):
        x0, x1, y0, y1 = f.rb
        if not P:
            return []
        mx0 = min(p[0] for p in P)
        mx1 = max(p[0] for p in P)
        my0 = min(p[1] for p in P)
        my1 = max(p[1] for p in P)
        negs = [c for c in f.negs if box_d2(mx0, mx1, my0, my1, c[0], c[1]) <= c[2]]
        poss = [c for c in f.poss if max((mx0 - c[0]) ** 2, (mx1 - c[0]) ** 2) + max((my0 - c[1]) ** 2, (my1 - c[1]) ** 2) > c[2]]
        if not negs and not poss:
            return [x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in P]
        res = []
        for p in P:
            x, y = p[0], p[1]
            ok = x0 <= x <= x1 and y0 <= y <= y1
            if ok:
                for cx, cy, r2 in negs:
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                        ok = False
                        break
            if ok:
                for cx, cy, r2 in poss:
                    if (x - cx) ** 2 + (y - cy) ** 2 > r2:
                        ok = False
                        break
            res.append(ok)
        return res

    def pf_ok1(self, f, p):
        x0, x1, y0, y1 = f.rb
        x, y = p[0], p[1]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        for cx, cy, r2 in f.negs:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                return False
        for cx, cy, r2 in f.poss:
            if (x - cx) ** 2 + (y - cy) ** 2 > r2:
                return False
        return True

    def pf_single(self, f):
        v = self.vis.get(f.id)
        if v is not None:
            f.P = [[v[0], v[1], v[2], v[3], 0] for _ in range(NPF)]
            f.obs = True
            return
        ok = self.pf_ok(f, f.P)
        S = [i for i, o in enumerate(ok) if o]
        f.P = self.pf_resample(f, None, S) if S else self.pf_seed(f)

    def pf_pair(self, a, b):
        n = NPF
        vis = self.vis
        va, vb = vis.get(a.id), vis.get(b.id)
        if va is not None and vb is not None:
            a.P = [[va[0], va[1], va[2], va[3], 0] for _ in range(n)]
            b.P = [[vb[0], vb[1], vb[2], vb[3], 0] for _ in range(n)]
            a.obs = b.obs = True
            return
        if va is not None or vb is not None:
            s, o, v = (a, b, va) if va is not None else (b, a, vb)
            m = [9999 - v[0], v[1], -v[2], v[3], 0]
            mok = self.pf_ok1(o, m)
            dist = any((mx - v[0]) ** 2 + (my - v[1]) ** 2 <= 1960000 for mx, my in self.pf_motors) or                 any(g is not s and g.P is not None and any((q[0] - v[0]) ** 2 + (q[1] - v[1]) ** 2 <= 360000 for q in g.P[::4]) for g in self.fl)
            cand, w = [], []
            nsym = 0
            for ps, po in zip(s.P, o.P):
                d2 = (ps[0] - v[0]) ** 2 + (ps[1] - v[1]) ** 2
                if po[0] + ps[0] == 9999 and po[1] == ps[1] and po[2] == -ps[2] and po[3] == ps[3]:
                    if mok:
                        cand.append([m[0], m[1], po[2], po[3], 0] if dist else m)
                        w.append(d2)
                        nsym += 1
                elif self.pf_ok1(o, po):
                    cand.append(po)
                    w.append(d2)
            s.P = [[v[0], v[1], v[2], v[3], 0] for _ in range(n)]
            s.obs = True
            if not cand:
                o.P = self.pf_seed(o)
                return
            dmin = min(w)
            sig2 = max(40000, 2 * dmin)
            ws = [math.exp(-(d2 - dmin) / sig2) for d2 in w]
            o.P = [list(q) for q in PRNG.choices(cand, weights=ws, k=n)]
            o.obs = all(q[4] == 0 for q in o.P)
            if not o.obs:
                for q in o.P:
                    if q[4] == 0:
                        q[4] = 1e-9
            return
        okA = self.pf_ok(a, a.P)
        okB = self.pf_ok(b, b.P)
        S = [i for i in range(n) if okA[i] and okB[i]]
        if S:
            a.P, b.P = self.pf_resample(a, b, S)
            return
        SA = [i for i in range(n) if okA[i]]
        SB = [i for i in range(n) if okB[i]]
        a.P = self.pf_resample(a, None, SA) if SA else self.pf_seed(a)
        b.P = self.pf_resample(b, None, SB) if SB else self.pf_seed(b)

    def pf_resample(self, a, b, S):
        """Resamples particles (jointly with twin b when given) from surviving indices S back to NPF particles,
        splitting each survivor's cell among its copies."""
        n = NPF
        k = len(S)
        P = a.P
        Q = b.P if b is not None else None
        if k == n:
            return (P, Q) if b is not None else P
        cnt = [n // k] * k
        for i in PRNG.sample(range(k), n - k * (n // k)):
            cnt[i] += 1
        NP, NQ = [], []
        ra = a.rb
        rbb = b.rb if b is not None else None
        for j, i in enumerate(S):
            p = P[i]
            c = cnt[j]
            q = Q[i] if Q is not None else None
            sym = q is not None and q[0] + p[0] == 9999 and q[1] == p[1] and q[2] == -p[2] and q[3] == p[3]
            cp = p[4]
            cq = q[4] if q is not None else 0
            sc = 1 / math.sqrt(c)
            NP.append([p[0], p[1], p[2], p[3], cp * sc])
            if q is not None:
                NQ.append([q[0], q[1], q[2], q[3], cq * sc])
            for _ in range(c - 1):
                dx = round((PRNG.random() * 2 - 1) * cp)
                dy = round((PRNG.random() * 2 - 1) * cp)
                x = min(ra[1], max(ra[0], p[0] + dx))
                y = min(ra[3], max(ra[2], p[1] + dy))
                NP.append([x, y, p[2], p[3], cp * sc])
                if q is not None:
                    if sym:
                        NQ.append([9999 - x, y, q[2], q[3], cq * sc])
                    else:
                        dx = round((PRNG.random() * 2 - 1) * cq)
                        dy = round((PRNG.random() * 2 - 1) * cq)
                        NQ.append([min(rbb[1], max(rbb[0], q[0] + dx)), min(rbb[3], max(rbb[2], q[1] + dy)), q[2], q[3], cq * sc])
        if b is not None:
            return NP, NQ
        return NP

    def pf_seed(self, f):
        """Draws fresh particles uniformly in the constraint region (radar box near the previous cloud)."""
        n = NPF
        f.obs = False
        x0, x1, y0, y1 = f.rb
        P = f.P
        out = []
        for e in (600, None):
            if e is not None and P:
                bx0, bx1 = max(x0, min(p[0] for p in P) - e), min(x1, max(p[0] for p in P) + e)
                by0, by1 = max(y0, min(p[1] for p in P) - e), min(y1, max(p[1] for p in P) + e)
            else:
                bx0, bx1, by0, by1 = x0, x1, y0, y1
            if bx0 > bx1 or by0 > by1:
                continue
            c = math.sqrt((bx1 - bx0 + 1) * (by1 - by0 + 1) / n) / 2
            for _ in range(4 * n):
                x, y = PRNG.randint(bx0, bx1), PRNG.randint(by0, by1)
                p = [x, y, 0, 0, c]
                if self.pf_ok1(f, p):
                    a = PRNG.random() * 6.283185307179586
                    p[2], p[3] = jround(200 * math.cos(a)), jround(200 * math.sin(a))
                    out.append(p)
                    if len(out) >= n:
                        break
            if out:
                break
        if not out:
            out = [[(x0 + x1) // 2, (y0 + y1) // 2, 0, 0, 300]]
        while len(out) < n:
            q = PRNG.choice(out)
            out.append([q[0], q[1], q[2], q[3], q[4]])
        return out

    def pf_vel(self, fl, motors):
        vis = self.vis
        bb = {}
        for f in fl:
            P = f.P
            bb[f.id] = (min(p[0] for p in P), max(p[0] for p in P), min(p[1] for p in P), max(p[1] for p in P))
        for f in fl:
            if f.id in vis:
                continue
            x0, x1, y0, y1 = bb[f.id]
            mn = [m for m in motors if box_d2(x0, x1, y0, y1, m[0], m[1]) <= 1960000]
            nb = []
            for g in fl:
                if g is f:
                    continue
                c = bb[g.id]
                if max(0, c[0] - x1, x0 - c[1]) ** 2 + max(0, c[2] - y1, y0 - c[3]) ** 2 <= 360000:
                    nb.append(g.P)
            lo, hi = f.low, f.high
            for i, p in enumerate(f.P):
                x, y = p[0], p[1]
                if mn:
                    bd = -1
                    for mx, my in mn:
                        dd = (mx - x) ** 2 + (my - y) ** 2
                        if bd < 0 or dd < bd:
                            bd, sx, sy, cnt = dd, mx, my, 1
                        elif dd == bd:
                            sx += mx
                            sy += my
                            cnt += 1
                    if bd <= 1960000:
                        ax, ay = x - sx / cnt, y - sy / cnt
                        l = math.hypot(ax, ay)
                        if l > 0:
                            p[2], p[3] = jround(ax / l * 400), jround(ay / l * 400)
                        else:
                            p[2] = p[3] = 0
                        continue
                vx, vy = p[2], p[3]
                l = math.hypot(vx, vy)
                fx, fy = (vx / l * 200, vy / l * 200) if l > 0 else (0, 0)
                if nb:
                    bd = -1
                    for G in nb:
                        q = G[i]
                        dd = (q[0] - x) ** 2 + (q[1] - y) ** 2
                        if bd < 0 or dd < bd:
                            bd, sx, sy, cnt = dd, q[0], q[1], 1
                        elif dd == bd:
                            sx += q[0]
                            sy += q[1]
                            cnt += 1
                    if bd <= 360000:
                        ax, ay = x - sx / cnt, y - sy / cnt
                        l = math.hypot(ax, ay)
                        fx, fy = (ax / l * 200, ay / l * 200) if l > 0 else (0, 0)
                px, py = x + fx, y + fy
                if (px < 0 and px < x) or (px > 9999 and px > x):
                    fx = -fx
                if (py < lo and py < y) or (py > hi and py > y):
                    fy = -fy
                p[2], p[3] = jround(fx), jround(fy)

    def pf_stats(self, f):
        P = f.P
        n = len(P)
        xs = [p[0] for p in P]
        ys = [p[1] for p in P]
        f.bx0, f.bx1, f.by0, f.by1 = min(xs), max(xs), min(ys), max(ys)
        p0 = P[0]
        if f.bx0 == f.bx1 and f.by0 == f.by1 and all(p[2] == p0[2] and p[3] == p0[3] for p in P):
            f.exact = True
            f.x, f.y, f.vx, f.vy = p0[0], p0[1], p0[2], p0[3]
            f.std = 0
            f.samples = [(f.x + f.vx, min(f.high, max(f.low, f.y + f.vy)))]
            return
        if f.obs:
            f.obs = False
            for p in P:
                if p[4] == 0:
                    p[4] = 1e-9
        f.exact = False
        f.x = sum(xs) / n
        f.y = sum(ys) / n
        f.vx = sum(p[2] for p in P) / n
        f.vy = sum(p[3] for p in P) / n
        f.std = math.sqrt(max(0, sum(x * x for x in xs) / n - f.x * f.x + sum(y * y for y in ys) / n - f.y * f.y))
        lo, hi = f.low, f.high
        f.samples = [(p[0] + p[2], min(hi, max(lo, p[1] + p[3]))) for p in P[::max(1, n // 16)]]

    def update_mons(self):
        vcirc = [(d.x, d.y, ((2000 if d.light else 800) + 300) ** 2) for d in self.my]
        tg = [(d.x, d.y, (2000 if d.light else 800) ** 2) for d in self.dr.values() if d.elig]
        ftg = [(d.x, d.y, (2000 if d.light else 800) ** 2) for d in self.foe if d.elig]
        ml = list(self.mons.values())
        first = self.turn == 0
        self.tg_hist.append(tg)
        if not first:
            for m in ml:
                if m.known:
                    m.x += m.vx
                    m.y = min(9999, max(2500, m.y + m.vy))
                else:
                    for p in m.parts:
                        p[0] += p[2]
                        p[1] = min(9999, max(2500, p[1] + p[3]))
            kpos = [(m.x, m.y, m) for m in ml if m.known]
            for m in ml:
                if m.known:
                    m.vx, m.vy, _, _ = mon_vel(m.x, m.y, m.vx, m.vy, tg, [(a, b) for a, b, o in kpos if o is not m])
            kp = [(a, b) for a, b, o in kpos]
            for m in ml:
                if not m.known:
                    for p in m.parts:
                        p[2], p[3], _, _ = mon_vel(p[0], p[1], p[2], p[3], tg, kp)
            for m in ml:
                if not m.still:
                    m.bx0 -= 540
                    m.bx1 += 540
                    m.by0 -= 540
                    m.by1 += 540
        for m in ml:
            rx0, rx1, ry0, ry1 = self.radar_box(m.id)
            nb = (max(m.bx0, rx0, 0), min(m.bx1, rx1, 9999), max(m.by0, ry0, 2500), min(m.by1, ry1, 9999))
            if nb[0] > nb[1] or nb[2] > nb[3]:
                nb = (max(rx0, 0), min(rx1, 9999), max(ry0, 2500), min(ry1, 9999))
                m.still = False
                m.known = False
                m.parts = []
            m.bx0, m.bx1, m.by0, m.by1 = nb
        for m in ml:
            t = m.twin
            if t is not None and m.still and t.still and not m.known and not t.known:
                nx0, nx1 = max(m.bx0, 9999 - t.bx1), min(m.bx1, 9999 - t.bx0)
                ny0, ny1 = max(m.by0, t.by0), min(m.by1, t.by1)
                if nx0 <= nx1 and ny0 <= ny1:
                    m.bx0, m.bx1, m.by0, m.by1 = nx0, nx1, ny0, ny1
                    t.bx0, t.bx1 = max(t.bx0, 9999 - nx1), min(t.bx1, 9999 - nx0)
                    t.by0, t.by1 = max(t.by0, ny0), min(t.by1, ny1)
        for m in ml:
            v = self.vis.get(m.id)
            if v is not None:
                if m.known and m.still and (v[0] != m.x or v[1] != m.y):
                    m.still = False
                if m.still and m.spawn is None:
                    m.spawn = (v[0], v[1])
                m.x, m.y, m.vx, m.vy = v
                if m.vx or m.vy:
                    m.still = False
                m.known = True
                m.parts = []
                m.seen = self.turn
                m.bx0 = m.bx1 = m.x
                m.by0 = m.by1 = m.y
            elif m.known:
                if not (m.bx0 <= m.x <= m.bx1 and m.by0 <= m.y <= m.by1) or \
                        any((m.x - cx) ** 2 + (m.y - cy) ** 2 <= r2 for cx, cy, r2 in vcirc):
                    m.known = False
                    m.still = False
                    m.parts = []
                elif m.vx or m.vy:
                    m.still = False
                    m.bx0 = m.bx1 = m.x
                    m.by0 = m.by1 = m.y
        for m in ml:
            t = m.twin
            if t is not None and m.spawn is not None and t.spawn is None:
                t.spawn = (9999 - m.spawn[0], m.spawn[1])
        for m in ml:
            if m.known or m.spawn is None or not m.replay_ok:
                continue
            x, y = m.spawn
            vx = vy = 0
            th = self.tg_hist
            for k in range(1, self.turn + 1):
                x += vx
                y = min(9999, max(2500, y + vy))
                vx, vy, _, _ = mon_vel(x, y, vx, vy, th[k], ())
            if m.bx0 <= x <= m.bx1 and m.by0 <= y <= m.by1 and                     all((x - cx) ** 2 + (y - cy) ** 2 > r2 for cx, cy, r2 in vcirc):
                m.x, m.y, m.vx, m.vy = x, y, vx, vy
                m.known = True
                m.parts = []
                m.still = (x, y) == m.spawn and not vx and not vy
                m.bx0 = m.bx1 = x
                m.by0 = m.by1 = y
            else:
                m.replay_ok = False
        for m in ml:
            if m.known:
                continue
            if m.still and any(box_d2(m.bx0, m.bx1, m.by0, m.by1, x, y) <= r2 for x, y, r2 in ftg):
                m.still = False
            ps = [p for p in m.parts if m.bx0 <= p[0] <= m.bx1 and m.by0 <= p[1] <= m.by1 and
                  all((p[0] - cx) ** 2 + (p[1] - cy) ** 2 > r2 for cx, cy, r2 in vcirc)]
            fresh = self.sample_parts(m, vcirc, NPART // 4)
            if not ps:
                ps = self.sample_parts(m, vcirc, NPART - len(fresh))
            else:
                need = NPART - len(fresh)
                if len(ps) > need:
                    ps = RNG.sample(ps, need)
                else:
                    extra = []
                    for _ in range(need - len(ps)):
                        q = RNG.choice(ps)
                        x = min(m.bx1, max(m.bx0, q[0] + RNG.randint(-250, 250)))
                        y = min(m.by1, max(m.by0, q[1] + RNG.randint(-250, 250)))
                        if all((x - cx) ** 2 + (y - cy) ** 2 > r2 for cx, cy, r2 in vcirc):
                            extra.append([x, y, q[2], q[3]])
                    ps += extra
            ps += fresh
            m.parts = ps
            m.x = sum(p[0] for p in ps) / len(ps)
            m.y = sum(p[1] for p in ps) / len(ps)
        self.ml = ml

    def sample_parts(self, m, vcirc, n):
        ps = []
        for _ in range(n * 4):
            x = RNG.randint(int(m.bx0), int(m.bx1))
            y = RNG.randint(int(m.by0), int(m.by1))
            if any((x - cx) ** 2 + (y - cy) ** 2 <= r2 for cx, cy, r2 in vcirc):
                continue
            if m.still:
                vx = vy = 0
            else:
                a = RNG.random() * 6.283185307179586
                s = RNG.choice((0, 270, 270, 270, 540))
                vx, vy = jround(s * math.cos(a)), jround(s * math.sin(a))
            ps.append([x, y, vx, vy])
            if len(ps) >= n:
                break
        if not ps:
            ps = [[(m.bx0 + m.bx1) // 2, (m.by0 + m.by1) // 2, 0, 0]]
        return ps

    # ------------------------------------------------------------------ scoring
    def score(self, mine, theirs):
        s = 0
        fish = self.fish
        for f, t in mine.items():
            s += (fish[f].typ + 1) * (2 if f not in theirs or theirs[f] >= t else 1)
        for fs, bonus in self.combos:
            if all(f in mine for f in fs):
                tm = max(mine[f] for f in fs)
                if all(f in theirs for f in fs):
                    s += bonus * (2 if max(theirs[f] for f in fs) >= tm else 1)
                else:
                    s += bonus * 2
        return s

    def max_possible(self, p, mine, theirs, carried, t):
        m = dict(mine)
        for f in self.fl:
            m.setdefault(f.id, t)
        for f in carried:
            m.setdefault(f, t)
        return self.score(m, theirs)

    # ------------------------------------------------------------------ decision
    def decide(self):
        t = self.turn
        my, foe = self.my, self.foe
        my_saved, foe_saved = self.my_saved, self.foe_saved
        self.carry = {d.id: self.scans.get(d.id, set()) - my_saved for d in my}
        self.fcarry = {d.id: self.scans.get(d.id, set()) - foe_saved for d in foe}
        mine_have = set(my_saved)
        for s in self.carry.values():
            mine_have |= s
        foe_have = set(foe_saved)
        for s in self.fcarry.values():
            foe_have |= s
        self.mine_have, self.foe_have = mine_have, foe_have
        alive = {f.id for f in self.fl}
        needed = [f for f in self.fl if f.id not in mine_have]
        assign = self.assign_fish(needed)
        deny = [f for f in self.fl if f.id in mine_have and f.id not in foe_have and
                0 <= f.x + (f.vx if f.exact else 0) <= 9999]
        modes = self.strategy(assign, deny)
        self.mode = modes
        # decide moves, most endangered first
        def danger(d):
            return min((math.hypot(m.x - d.x, m.y - d.y) for m in self.ml if m.known), default=INF)
        plan = {}
        for d in sorted(my, key=danger):
            if d.em:
                plan[d.id] = (0, 300, False, False)
                continue
            plan[d.id] = self.choose(d, modes[d.id], plan)
        out = []
        for d in my:
            vx, vy, motor, light = plan[d.id]
            self.cmd[d.id] = motor
            if motor:
                out.append(f"MOVE {d.x + vx} {d.y + vy} {int(light)}")
            else:
                out.append(f"WAIT {int(light)}")
        return out

    def foe_eta(self, e, cont):
        if not cont or e.vy < -100:
            return surf_turns(e.y)
        if e.y >= 7500:
            return surf_turns(e.y) + 1
        return (7500 - e.y + 599) // 600 + surf_turns(7500)

    def strategy(self, assign, deny):
        t = self.turn
        my, foe = self.my, self.foe
        LATE = t + 45
        alive = {f.id for f in self.fl}
        base_my = {f: v for f, v in self.saved_t[0].items()}
        for f in alive | self.mine_have:
            base_my.setdefault(f, LATE)
        self.base_my = base_my
        scens = []
        any_cont = any(not e.em and self.fcarry[e.id] and e.vy >= -100 for e in foe)
        for cont, w in ((False, 0.35), (True, 0.65)) if any_cont else ((False, 1.0),):
            bf = dict(self.saved_t[1])
            for e in foe:
                if e.em:
                    continue
                T = min(200, t + self.foe_eta(e, cont))
                for f in self.fcarry[e.id]:
                    if bf.get(f, INF) > T:
                        bf[f] = T
            for f in alive:
                bf.setdefault(f, LATE)
            scens.append((bf, w))
        self.scens = scens
        opts = {d.id: self.options(d, assign[d.id], deny) for d in my}
        choice = {d.id: 0 for d in my}
        for _ in range(2):
            for d in my:
                bi, bv = choice[d.id], -INF
                for i in range(len(opts[d.id])):
                    choice[d.id] = i
                    v = self.eval_choice(opts, choice)
                    if v > bv:
                        bi, bv = i, v
                choice[d.id] = bi
        self.strat_val = bv
        modes = {}
        for d in my:
            o = opts[d.id][choice[d.id]]
            modes[d.id] = (o[0], o[1])
            if o[0] == "HERD":
                self.herd[d.id] = o[1].id
            else:
                self.herd.pop(d.id, None)
        return modes

    def options(self, d, tl, deny):
        t = self.turn
        if d.em:
            return [("EM", None, 0, frozenset(), frozenset(), 0, 1)]
        C = frozenset(self.carry[d.id])
        prev = self.mode.get(d.id, ("", None))
        cv = sum(2 * (self.fish[f].typ + 1) for f in C)
        res = []
        st = surf_turns(d.y)
        if tl:
            route = self.get_route(d, tl)
            order = self.routes[d.id][2]
            x, y, cost = d.x, d.y, 0
            fs = set(C)
            for k, pt in enumerate(route):
                x, y, cost = leg(x, y, cost, pt, 0)
                fs.add(order[k])
                T = (cost + max(0, y - 500)) / 600
                if k == len(route) - 1 or (fs - C and k < 3):
                    b = 1.5 if prev[0] == "EXP" else 0
                    res.append(("EXP", route[:k + 1], T, frozenset(fs), frozenset(), b - 0.004 * T * cv, 1))
            res.reverse()
        if C:
            b = 1.5 if prev[0] == "SURF" else 0
            if self.win_if_saved(d):
                b += 500
            res.append(("SURF", None, st, C, frozenset(), b - 0.004 * st * cv, 1))
        hc = []
        for f in deny:
            c = self.herd_cost(d, f)
            if c > 14 or t + c > 198:
                continue
            hc.append((self.deny_value(f) / (c + 2), c, f))
        hc.sort(key=lambda z: -z[0])
        for _, c, f in hc[:2]:
            T = c + surf_turns(f.y)
            foe_t = min((max(0, math.hypot(e.x - f.x, e.y - f.y) - 2000) / 600 for e in self.foe if not e.em), default=99)
            p = 0.3 if foe_t < c else 0.9
            b = 1.5 if prev[0] == "HERD" and prev[1] is f else 0
            res.append(("HERD", f, T, C, frozenset((f.id,)), b - 0.004 * T * cv - 0.02 * c, p))
        if not C and not tl:
            res.append(("IDLE", None, 0, frozenset(), frozenset(), 0, 1))
        return res

    def eval_choice(self, opts, choice):
        t = self.turn
        my_t = dict(self.base_my)
        removed = set()
        bonus = 0
        p = 1
        herds = set()
        for did, i in choice.items():
            mode, arg, T, fs, rem, b, pr = opts[did][i]
            if mode == "HERD":
                if arg.id in herds:
                    return -INF
                herds.add(arg.id)
            tt = min(200, t + T)
            for f in fs:
                if my_t.get(f, INF) > tt:
                    my_t[f] = tt
            removed |= rem
            bonus += b
            p *= pr
        val = 0
        for bf, w in self.scens:
            v1 = self.score(my_t, bf) - self.score(bf, my_t)
            if removed:
                bf2 = {f: v for f, v in bf.items() if f not in removed or f in self.foe_have}
                v2 = self.score(my_t, bf2) - self.score(bf2, my_t)
                v1 = p * v2 + (1 - p) * v1
            val += w * v1
        return val + bonus

    def assign_fish(self, needed):
        my = self.my
        if self.turn == 0 or not hasattr(self, "home"):
            o = sorted(my, key=lambda d: d.x)
            self.home = {o[0].id: 0, o[-1].id: 1}
            self.prev_assign = {}
        st = {}
        for d in my:
            if d.em:
                st[d.id] = (d.y / 300 + 1, d.x, 0)
            elif self.mode.get(d.id, ("",))[0] == "SURF" and self.carry[d.id]:
                st[d.id] = (surf_turns(d.y), d.x, 500)
            else:
                st[d.id] = (0, d.x, d.y)
        A = {d.id: [] for d in my}
        pa = self.prev_assign
        for f in needed:
            side = 0 if f.x < 5000 else 1
            bd, bc = None, INF
            for d in my:
                s0, sx, sy = st[d.id]
                c = s0 + math.hypot(f.x - sx, f.y - sy) / 600 + (0 if side == self.home[d.id] else 4)
                if pa.get(f.id) == d.id:
                    c -= 1.5
                if c < bc:
                    bd, bc = d.id, c
            A[bd].append(f)
        # rebalance: a free drone takes the far end of the other drone's list
        for a in my:
            if a.em or A[a.id] or self.carry[a.id]:
                continue
            for b in my:
                if b is a or len(A[b.id]) < 2:
                    continue
                r = self.routes.get(b.id)
                if r is None:
                    continue
                ids = [i for i in r[2] if any(f.id == i for f in A[b.id])]
                sb, bx, by = st[b.id]
                t_b = sb
                arr = {}
                x, y = bx, by
                for i in ids:
                    f = self.fish[i]
                    t_b += math.hypot(f.x - x, f.y - y) / 600
                    x, y = f.x, f.y
                    arr[i] = t_b
                for i in reversed(ids[1:]):
                    f = self.fish[i]
                    ta = st[a.id][0] + math.hypot(f.x - a.x, f.y - a.y) / 600
                    if ta + 1 < arr[i]:
                        A[b.id].remove(f)
                        A[a.id].append(f)
                    else:
                        break
        self.prev_assign = {f.id: did for did, fl in A.items() for f in fl}
        return A

    def get_route(self, d, tl):
        key = frozenset(f.id for f in tl)
        r = self.routes.get(d.id)
        if r is None or r[0] != key or self.turn - r[1] >= 3:
            order = self.plan_route(d, tl)
            if r is not None:
                old = [self.fish[i] for i in r[2] if i in key]
                old += [f for f in order if f not in old]
                light_ok = d.bat >= 5
                if self.route_cost(d, old, light_ok) <= self.route_cost(d, order, light_ok) + 400:
                    order = old
            self.routes[d.id] = (key, self.turn, [f.id for f in order])
        else:
            order = [self.fish[i] for i in r[2]]
        light_ok = d.bat >= 5
        return [self.rpt(f, light_ok) for f in order]

    def rpt(self, f, light_ok):
        return f.x, f.y, self.scan_r(f, light_ok), f.vx if f.exact else 0, f.vy if f.exact else 0, f.low, f.high

    def route_cost(self, d, order, light_ok):
        x, y, cost = d.x, d.y, 0
        for f in order:
            x, y, cost = leg(x, y, cost, self.rpt(f, light_ok), 0)
        return cost + max(0, y - 500)

    def scan_r(self, f, light_ok):
        if not light_ok:
            return 650
        if f.exact:
            return 1850
        return max(600, min(1850, 1950 - 1.2 * f.std))

    def plan_route(self, d, tl):
        if len(tl) > 6:
            tl = sorted(tl, key=lambda f: (f.x - d.x) ** 2 + (f.y - d.y) ** 2)[:6]
        n = len(tl)
        light_ok = d.bat >= 5
        pts = [self.rpt(f, light_ok) for f in tl]
        best = [INF, list(range(n))]
        order = []
        used = [False] * n

        def dfs(x, y, cost, k):
            if cost >= best[0]:
                return
            if k == n:
                c = cost + max(0, y - 500)
                if c < best[0]:
                    best[0] = c
                    best[1] = list(order)
                return
            for i in range(n):
                if used[i]:
                    continue
                nx, ny, c = leg(x, y, cost, pts[i], 0)
                used[i] = True
                order.append(i)
                dfs(nx, ny, c, k + 1)
                order.pop()
                used[i] = False

        dfs(d.x, d.y, 0, 0)
        return [tl[i] for i in best[1]]

    def win_if_saved(self, d):
        t = self.turn
        T = t + surf_turns(d.y)
        mine = dict(self.saved_t[0])
        for f in self.carry[d.id]:
            mine.setdefault(f, T)
        theirs = self.saved_t[1]
        my_sc = self.score(mine, theirs)
        fc = set()
        for s in self.fcarry.values():
            fc |= s
        return my_sc > self.max_possible(1, theirs, mine, fc, T + 1)

    def deny_value(self, f):
        v = (f.typ + 1) * (1 if f.id in self.my_saved else 1.5)
        fh = self.foe_have
        alive = {g.id for g in self.fl}
        for fs, bonus in self.combos:
            if f.id in fs and all(g in fh or g in alive for g in fs):
                done = all(g in self.my_saved for g in fs)
                v += bonus * (1 if done else 1.5) * 0.6
        return v

    def herd_cost(self, d, f):
        s = -1 if f.x < 5000 else 1
        edge = f.x + 1 if s < 0 else 10000 - f.x
        bx = f.x - s * 700
        c = math.hypot(d.x - bx, d.y - f.y) / 600
        if (d.x - f.x) * s > 0:
            c += 2
        return c + edge / 400

    # ------------------------------------------------------------------ move selection
    def choose(self, d, mode, plan):
        md, arg = mode
        x, y = d.x, d.y
        needed = [f for f in self.fl if f.id not in self.mine_have]
        # other drones predicted motion
        others = []
        for o in self.my + self.foe:
            if o.id == d.id or o.em:
                continue
            if o.id in plan:
                vx, vy, motor, light = plan[o.id]
                lr = 4000000 if light else 640000
            else:
                vx, vy, motor, lr = o.vx, o.vy, o.fmotor, 640000
                if o.mine:
                    vx, vy = 0, 0
            others.append((o.x, o.y, vx, vy, lr, motor))
        motors_other = [(ox + vx, oy + vy) for ox, oy, vx, vy, lr, mt in others if mt]
        # goal
        if md == "EXP":
            route = arg

            def goal(px, py, motor):
                cx, cy, cost = px, py, 0
                for pt in route:
                    cx, cy, cost = leg(cx, cy, cost, pt, 1)
                return -(cost + max(0, cy - 500))
        elif md == "SURF":
            def goal(px, py, motor):
                return -max(py, 300) - 0.02 * abs(px - x)
        elif md == "HERD":
            f = arg
            goal = self.herd_goal_fn(d, f, motors_other)
        elif md == "PT":
            gx, gy = arg

            def goal(px, py, motor):
                return -math.hypot(px - gx, py - gy)
        else:
            def goal(px, py, motor):
                return -abs(py - 300) - 0.05 * abs(px - x)
        cands = {}
        for cx, cy in DIR32:
            for sp in (600, 300):
                tx = min(9999, max(0, x + int(cx * sp)))
                ty = min(9999, max(0, y + int(cy * sp)))
                cands[(tx - x, ty - y, True)] = 1
        cands[(0, 0, True)] = 1
        cands[(0, 300, False)] = 1
        light_ok = d.bat >= 5
        lthr = 0.45 if d.bat >= 25 else 0.65 if d.bat >= 15 else 0.85 if d.bat >= 10 else 1.05
        clist = []
        for vx, vy, motor in cands:
            if motor:
                px, py = x + vx, y + vy
            else:
                px, py = x, min(9999, y + 300)
            g = goal(px, py, motor)
            clist.append([g, vx, vy, motor, False, px, py])
            if light_ok:
                lg = self.light_gain(px, py, needed)
                if lg >= lthr:
                    clist.append([g + 500 * min(lg, 2.5), vx, vy, motor, True, px, py])
        ctx = self.safety_ctx(d, others)
        # turn-1 pass
        scored = []
        for c in clist:
            g, vx, vy, motor, light, px, py = c
            lvl, pen = self.turn1(d, vx, vy, light, px, py, ctx)
            if lvl == 0:
                scored.append((-1, g - pen - 1e6, c))
            else:
                scored.append((1, g - pen, c))
        scored.sort(key=lambda s: -s[1])
        best, bkey = None, None
        for k, (l1, sc, c) in enumerate(scored):
            if l1 < 0:
                if best is None:
                    best, bkey = c, (0, sc)
                break
            if time.perf_counter() - self.tstart > TIME_BUDGET and best is not None:
                break
            lvl = self.deep_level(d, c, ctx)
            key = (lvl, sc)
            if bkey is None or key > bkey:
                best, bkey = c, key
            if lvl >= 3:
                break
        tag = ""
        if md == "HERD":
            tag = str(arg.id)
        elif md == "EXP" and arg:
            tag = "(%d,%d,%d)" % (arg[0][0], arg[0][1], arg[0][2])
        self.dbg.append(f"{d.id}:{md}{tag} L{bkey[0]} {int(bkey[1])} {'*' if best[4] else ''}")
        return best[1], best[2], best[3], best[4]

    def light_gain(self, px, py, needed):
        g = 0
        for f in needed:
            w = 1 + 0.25 * f.typ
            if f.exact:
                fx, fy = f.x + f.vx, min(f.high, max(f.low, f.y + f.vy))
                d2 = (fx - px) ** 2 + (fy - py) ** 2
                if 640000 < d2 <= 3880000:
                    g += w
            else:
                S = f.samples
                c = 0
                for sx, sy in S:
                    d2 = (sx - px) ** 2 + (sy - py) ** 2
                    if 640000 < d2 <= 3800000:
                        c += 1
                g += w * c / len(S)
        return g

    def herd_goal_fn(self, d, f, motors_other):
        s = -1 if f.x < 5000 else 1
        f1x = f.x + f.vx
        f1y = min(f.high, max(f.low, f.y + f.vy))
        fvx, fvy = f.vx, f.vy
        low, high = f.low, f.high
        ideal_x = f1x - s * 650

        def goal(px, py, motor):
            if f1x < 0 or f1x > 9999:
                return 1e5 - abs(py - 300)
            bd, bx, by = INF, 0, 0
            if motor:
                bd, bx, by = (f1x - px) ** 2 + (f1y - py) ** 2, px, py
            for ox, oy in motors_other:
                dd = (f1x - ox) ** 2 + (f1y - oy) ** 2
                if dd < bd:
                    bd, bx, by = dd, ox, oy
            if bd <= 1960000:
                ax, ay = f1x - bx, f1y - by
                l = math.hypot(ax, ay)
                nvx = jround(ax / l * 400) if l > 0 else 0
            else:
                l = math.hypot(fvx, fvy)
                nvx = fvx / l * 200 if l > 0 else 0
            f2x = f1x + nvx
            if f2x < 0 or f2x > 9999:
                return 5e4 - math.hypot(px - ideal_x, py - f1y)
            return 2.0 * s * f2x - 0.7 * math.hypot(px - ideal_x, py - f1y)
        return goal

    def safety_ctx(self, d, others):
        x, y = d.x, d.y
        mons = []
        near = []
        for m in self.ml:
            if not m.known:
                continue
            dd = math.hypot(m.x - x, m.y - y)
            if dd < 6000:
                if dd < 4200:
                    near.append(len(mons))
                mons.append((m.x, m.y, m.vx, m.vy))
        tg1 = []
        tg2 = []
        tg3 = []
        for ox, oy, vx, vy, lr, mt in others:
            p1x, p1y = min(9999, max(0, ox + vx)), min(9999, max(0, oy + vy))
            p2x, p2y = min(9999, max(0, p1x + vx)), min(9999, max(0, p1y + vy))
            tg1.append((p1x, p1y, lr))
            tg2.append((p2x, p2y, 640000))
            tg3.append((min(9999, max(0, p2x + vx)), min(9999, max(0, p2y + vy)), 640000))
        parts = []
        for m in self.ml:
            if m.known or not m.parts:
                continue
            if box_d2(m.bx0, m.bx1, m.by0, m.by1, x, y) > 3000 ** 2:
                continue
            w = 1 / len(m.parts)
            for p in m.parts:
                if (p[0] - x) ** 2 + (p[1] - y) ** 2 < 3000 ** 2:
                    parts.append((p[0], p[1], p[2], p[3], w))
        return mons, near, tg1, tg2, tg3, parts

    def turn1(self, d, vx, vy, light, px, py, ctx):
        mons, near, tg1, tg2, tg3, parts = ctx
        x, y = d.x, d.y
        pen = 0
        for i in near:
            mx, my, mvx, mvy = mons[i]
            if collides(x, y, vx, vy, mx, my, mvx, mvy, R1):
                return 0, 0
        for qx, qy, qvx, qvy, w in parts:
            if collides(x, y, vx, vy, qx, qy, qvx, qvy, 600 ** 2):
                pen += 6000 * w
            elif (qx + qvx - px) ** 2 + (qy + qvy - py) ** 2 < 900 ** 2:
                pen += 400 * w
        if near:
            M1 = [(mx + mvx, min(9999, max(2500, my + mvy))) for mx, my, mvx, mvy in mons]
            tg = tg1 + [(px, py, 4000000 if light else 640000)]
            for i in near:
                m1x, m1y = M1[i]
                nvx, nvy, tx, ty = mon_vel(m1x, m1y, mons[i][2], mons[i][3], tg, M1[:i] + M1[i + 1:])
                dd = math.hypot(m1x - px, m1y - py)
                if tx is not None and tx == px and ty == py:
                    pen += 125 + max(0, 1600 - dd) * 0.75
                else:
                    pen += max(0, 1100 - dd) * 0.5
        return 1, pen

    def deep_level(self, d, c, ctx):
        mons, near, tg1, tg2, tg3, parts = ctx
        if not near:
            return 3
        g, vx, vy, motor, light, px, py = c
        M1 = [(mx + mvx, min(9999, max(2500, my + mvy))) for mx, my, mvx, mvy in mons]
        tg = tg1 + [(px, py, 4000000 if light else 640000)]
        V2 = []
        for i, (m1x, m1y) in enumerate(M1):
            nvx, nvy, _, _ = mon_vel(m1x, m1y, mons[i][2], mons[i][3], tg, M1[:i] + M1[i + 1:])
            V2.append((nvx, nvy))
        # escape ordering: away from nearest monster
        nm = min(near, key=lambda i: (M1[i][0] - px) ** 2 + (M1[i][1] - py) ** 2)
        ax, ay = px - M1[nm][0], py - M1[nm][1]
        esc = sorted(ESC, key=lambda e: -(e[0] * ax + e[1] * ay))
        best = 1
        M2 = [(M1[i][0] + V2[i][0], min(9999, max(2500, M1[i][1] + V2[i][1]))) for i in range(len(M1))]
        for ex, ey in esc:
            if (ex, ey) != (0, 300):
                # a MOVE target is clamped to the map, so its velocity is too (WAIT keeps its unclamped sink speed)
                ex, ey = min(9999, max(0, px + ex)) - px, min(9999, max(0, py + ey)) - py
            ok = True
            for i in near:
                if collides(px, py, ex, ey, M1[i][0], M1[i][1], V2[i][0], V2[i][1], R2):
                    ok = False
                    break
            if not ok:
                continue
            best = 2
            p2x, p2y = min(9999, max(0, px + ex)), min(9999, max(0, py + ey))
            tgb = tg2 + [(p2x, p2y, 640000)]
            V3 = []
            for i, (m2x, m2y) in enumerate(M2):
                nvx, nvy, _, _ = mon_vel(m2x, m2y, V2[i][0], V2[i][1], tgb, M2[:i] + M2[i + 1:])
                V3.append((nvx, nvy))
            bx, by = p2x - M2[nm][0], p2y - M2[nm][1]
            for fx, fy in sorted(ESC, key=lambda e: -(e[0] * bx + e[1] * by)):
                if (fx, fy) != (0, 300):
                    fx, fy = min(9999, max(0, p2x + fx)) - p2x, min(9999, max(0, p2y + fy)) - p2y
                ok = True
                for i in near:
                    if collides(p2x, p2y, fx, fy, M2[i][0], M2[i][1], V3[i][0], V3[i][1], R2):
                        ok = False
                        break
                if ok:
                    return 3
        return best

    # ------------------------------------------------------------------ main loop
    def fallback(self):
        out = []
        for a in self.my_list:
            out.append(f"MOVE {a[1]} {max(0, a[2] - 600)} 0")
        return out

    def dump(self):
        import json
        rec = {"t": self.turn, "fish": {f.id: [round(f.x), round(f.y), f.vx, f.vy, int(f.exact), f.bx0, f.bx1, f.by0, f.by1] for f in self.fl},
               "mons": {m.id: [round(m.x), round(m.y), m.vx, m.vy, int(m.known), len(m.parts), int(m.still), m.bx0, m.bx1, m.by0, m.by1] for m in self.ml},
               "mode": {k: v[0] for k, v in self.mode.items()}}
        with open(DUMP + str(min(self.my_ids)), "a") as fh:
            fh.write(json.dumps(rec) + chr(10))

    def run(self):
        while True:
            try:
                self.parse()
            except EOFError:
                return
            except Exception as e:
                print(f"parse error {e!r}", file=sys.stderr)
                return
            try:
                self.dbg = []
                self.update()
                out = self.decide()
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                out = self.fallback()
                for a in self.my_list:
                    self.cmd[a[0]] = True
            dt = (time.perf_counter() - self.tstart) * 1000
            if DUMP:
                self.dump()
            print(f"t{self.turn} {self.my_score}-{self.foe_score} {dt:.1f}ms " + " ".join(self.dbg), file=sys.stderr)
            print("\n".join(out))
            sys.stdout.flush()
            self.turn += 1


if __name__ == "__main__":
    Bot().run()
