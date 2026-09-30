"""Selenia City bot: greedy network construction guided by an exact month simulator."""
import base64
import math
import os
import sys
import time
import traceback
import zlib
from bisect import bisect_right
from collections import defaultdict
from heapq import heapify, heappop, heappush

T_START = time.perf_counter()  # before the numpy import: the time limit of the first turn may include the start-up

import numpy as np

ENV = os.environ if os.environ.get("SELENIA_TUNE") else {}  # tuning overrides (development only)
LOG = not os.environ.get("SELENIA_NOLOG")  # per-month diagnostics on stderr, decoded by logtool.py
LOG_CHUNK = 1000  # maximal payload characters per encoded log line
SIM_COUNT = [0]  # simulate() calls, for the log
INF = 10 ** 9
DAYS = 20
MONTHS = 20
POD_CAPACITY = 10
TELEPORT_COST = 5000
POD_COST = 1000
POD_REFUND = 750
TOPO_ACTS = {"do_tube", "do_teleport"}
SLOW = int(ENV.get("SLOW", "1"))
SLOW_DAY = float(ENV.get("SLOW_DAY", "1"))
TIME_MARGIN = float(ENV.get("TIME_MARGIN", "0.05"))
BUDGET0 = float(ENV.get("BUDGET0", "0.65"))  # planning time of the first turn (s, before TIME_MARGIN)
BUDGET1 = float(ENV.get("BUDGET1", "0.35"))  # planning time of later turns (s, before TIME_MARGIN)
STEP_FRAC = float(ENV.get("STEP_FRAC", "0.3"))
THRESH = float(ENV.get("THRESH", "0.5"))
DROP = int(ENV.get("DROP", "1"))
DET = int(ENV.get("DET", "0"))  # testing only: deterministic virtual clock driven by a work model
VCLOCK = [0.]
VSCALE = float(ENV.get("VSCALE", "1"))
FRESH_MIN = int(ENV.get("FRESH_MIN", "0"))
CH2 = int(ENV.get("CH2", "1"))
CH2_N = int(ENV.get("CH2_N", "20"))
REFRESH = int(ENV.get("REFRESH", "0"))
PILOT_K = int(ENV.get("PILOT_K", "4"))
PILOT_FRAC = float(ENV.get("PILOT_FRAC", "0.7"))
MAX_STEPS = float(ENV.get("MAX_STEPS", "20"))
STEP_MIN = float(ENV.get("STEP_MIN", "0.01"))
NT = int(ENV.get("NT", "5"))
NN = int(ENV.get("NN", "8"))
NM = int(ENV.get("NM", "3"))
MLIM = float(ENV.get("MLIM", "1.5"))
DROP_FRAC = float(ENV.get("DROP_FRAC", "0.8"))
QUANTILE = float(ENV.get("QUANTILE", "0.995"))
TH_K = float(ENV.get("TH_K", "5000"))  # acceptance threshold (marginal value of money) = min(TH_MAX, TH_K / resources)
TH_MAX = float(ENV.get("TH_MAX", "1.5"))
TH_M = float(ENV.get("TH_M", "14"))  # the threshold decreases linearly to 0 (last month) over the last TH_M + 1 months
TH_HI = float(ENV.get("TH_HI", "4"))  # marginal value of money inside the reserve for future buildings
CB = float(ENV.get("CB", "1500"))  # expected cost of serving one new building
HOLD = int(ENV.get("HOLD", "1"))  # without observed income, keep all resources when a HOLD_K months later build would pay more
HOLD_K = int(ENV.get("HOLD_K", "2"))
HOLD_ML = int(ENV.get("HOLD_ML", "11"))  # only while more than HOLD_ML months are left
HOLD_FRAC = float(ENV.get("HOLD_FRAC", "0.6"))  # share of the turn budget for the regular plan when a hold is possible
HOLD_INC = int(ENV.get("HOLD_INC", "1"))  # also consider holds with observed income (both paths receive the forecast income)
DETOURS = int(ENV.get("DETOURS", "1"))  # re-route existing pods through needed tubes instead of buying new pods
DETOUR_PODS = int(ENV.get("DETOUR_PODS", "4"))  # pods per visited building considered for a detour insertion
DETOUR_LEN = int(ENV.get("DETOUR_LEN", "13"))  # max stops of a re-routed pod path
DETOUR_VCOST = float(ENV.get("DETOUR_VCOST", "250"))  # virtual cost added per re-routed pod (longer loops serve their tubes less often)
DETOUR_LOSS = float(ENV.get("DETOUR_LOSS", "30"))  # estimated points lost per astronaut hop no longer carried by a re-routed pod
DETOUR_NEW = int(ENV.get("DETOUR_NEW", "1"))  # detours over new tube candidates too
DETOUR_NB = int(ENV.get("DETOUR_NB", "0"))  # new-tube detours also into pods visiting tube neighbours
SHORTEN = int(ENV.get("SHORTEN", "1"))  # remove (almost) unused round trips from looping pods
SHORTEN_MAX = int(ENV.get("SHORTEN_MAX", "0"))  # max boardings on a removed round trip
DETOUR_CHAIN = int(ENV.get("DETOUR_CHAIN", "1"))  # detours over 2-hop chain candidates b-x-m
SPLIT = int(ENV.get("SPLIT", "1"))  # split a pod's back-and-forth over a saturated tube into its own shuttle instead of upgrading
DETOUR_RES = float(ENV.get("DETOUR_RES", "20000"))  # re-routing candidates only while resources are below this
KNN = int(ENV.get("KNN", "10"))  # candidate new tubes of route searches: pairs among the KNN nearest neighbours of each building
HOP_PENALTY = int(ENV.get("HOP_PENALTY", "5000"))  # route search: resource-equivalent penalty per hop
ROUTE_GROUPS = int(ENV.get("ROUTE_GROUPS", "8"))  # largest stuck groups that get a routed candidate
SPEED_GROUPS = int(ENV.get("SPEED_GROUPS", "32"))  # served groups (largest speed loss) that get a shorter-route candidate
BAL_GROUPS = int(ENV.get("BAL_GROUPS", "32"))  # largest flows into crowded modules that get a route to a less crowded module
ROUTE_IMPROVE_K = float(ENV.get("ROUTE_IMPROVE_K", "0.3"))  # estimate multiplier of improvement (speed / balance) route candidates
ROUTE_BAL_MIN = int(ENV.get("ROUTE_BAL_MIN", "25"))  # monthly arrivals above which a module counts as crowded
LOW_PENALTY = int(ENV.get("LOW_PENALTY", "100"))  # hop penalty of the second (cheap, network-reusing) route of stuck groups (0 = off)
RB = int(ENV.get("RB", "1"))  # stuck groups also get a route to the modules of their type with fewer than ROUTE_BAL_MIN arrivals
REST = int(ENV.get("REST", "1"))  # stuck-route estimate from hops and the target module's balance points (else 60 - 3 hops)
SERVED = int(ENV.get("SERVED", "1"))  # served-group candidates (served_candidates)
W_TOP = int(ENV.get("W_TOP", "30"))  # (building, type) pairs with the most waiting astronaut-days considered
W_K = float(ENV.get("W_K", "0.05"))  # estimated gain per waiting astronaut-day (low: served candidates act as a fallback pool)
LANE = int(ENV.get("LANE", "1"))  # phase-aligned shuttles along the whole shortest route of a waiting pair
LANE_MAX = int(ENV.get("LANE_MAX", "6"))
LANE_K = float(ENV.get("LANE_K", "0.15"))
W_MIDS = int(ENV.get("W_MIDS", "1"))  # tube + shuttle from a waiting non-pad building to its nearest target modules
W_MIDS_N = int(ENV.get("W_MIDS_N", "2"))
FILL = int(ENV.get("FILL", "1"))  # keep evaluating fresh candidates while the best exact value is below the threshold
BAL = int(ENV.get("BAL", "1"))  # balancing candidates (diversions from crowded modules)
BAL_FLOWS = int(ENV.get("BAL_FLOWS", "40"))  # largest arrival flows into crowded modules considered per generation
BAL_F = int(ENV.get("BAL_F", "3"))  # minimum monthly arrivals of a flow
BAL_C = int(ENV.get("BAL_C", "10"))  # minimum monthly arrivals at the crowded module
BAL_D = int(ENV.get("BAL_D", "10"))  # alternative modules need fewer arrivals than the crowded module minus BAL_D
BAL_K = int(ENV.get("BAL_K", "3"))  # alternative modules per flow
BAL_LEGAL = int(ENV.get("BAL_LEGAL", "1"))  # tube diversions to the first BAL_K alternatives reachable by an existing or legal new tube
BAL_PRIO = int(ENV.get("BAL_PRIO", "1"))  # re-create the pods feeding a crowded module with higher ids so rival pods fill first
BAL_VH = float(ENV.get("BAL_VH", "6e-7"))  # testing only: virtual-clock charge per recorded departure in detailed simulations
BAL_VF = float(ENV.get("BAL_VF", "1e-5"))  # testing only: virtual-clock charge per flow examined by balance_candidates
TELE = int(ENV.get("TELE", "1"))  # teleporter candidates from buildings astronauts stand at (or hubs next to them)
TELE_PER_GROUP = int(ENV.get("TELE_PER_GROUP", "2"))  # exits per (building, type)
HUB_PODS = int(ENV.get("HUB_PODS", "2"))  # maximum shuttle pods feeding a hub entrance
TELE_K = float(ENV.get("TELE_K", "0.5"))  # multiplier of teleporter candidate estimates


def orient(ax, ay, bx, by, cx, cy):
    v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (v > 0) - (v < 0)


def on_segment(ax, ay, bx, by, cx, cy):
    if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) != 0:
        return False
    return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)


def segments_intersect(p1, p2, p3, p4):
    d1, d2 = orient(*p3, *p4, *p1), orient(*p3, *p4, *p2)
    d3, d4 = orient(*p1, *p2, *p3), orient(*p1, *p2, *p4)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return (d1 == 0 and on_segment(*p3, *p4, *p1)) or (d2 == 0 and on_segment(*p3, *p4, *p2)) or \
        (d3 == 0 and on_segment(*p1, *p2, *p3)) or (d4 == 0 and on_segment(*p1, *p2, *p4))


class State:
    """Network state: buildings (shared, immutable), tubes, teleporters, pods and resources."""

    def __init__(self):
        self.pos = {}  # building id -> (x, y)
        self.btype = {}  # building id -> type (0 = landing pad)
        self.pad_astronauts = {}  # pad id -> list of astronaut types
        self.tubes = {}  # (a, b) with a < b -> capacity
        self.adj = defaultdict(set)
        self.teleports = {}  # entrance -> exit
        self.tele_used = set()
        self.pods = {}  # pod id -> path
        self.resources = 0
        self.actions = []
        self.blacklist = set()
        self.groups = None  # pad id -> {type: sorted astronaut ids}, filled lazily by simulate
        self.legal_ok = set()  # tubes known to pass the geometric legality checks (shared by copies)
        self.fresh = set()  # pods (re)created by this turn's actions (their POD action can be rewritten for free)

    def copy(self) -> "State":
        s = State.__new__(State)
        s.pos, s.btype, s.pad_astronauts, s.blacklist, s.groups, s.legal_ok = self.pos, self.btype, self.pad_astronauts, self.blacklist, self.groups, self.legal_ok
        s.tubes = dict(self.tubes)
        s.adj = defaultdict(set, self.adj)  # copy-on-write: do_tube replaces the neighbor sets it changes
        s.teleports = dict(self.teleports)
        s.tele_used = set(self.tele_used)
        s.pods = dict(self.pods)
        s.resources = self.resources
        s.actions = list(self.actions)
        s.fresh = set(self.fresh)
        return s

    # ------------------------------------------------------------ costs / legality
    def tube_cost(self, a: int, b: int) -> int:
        (ax, ay), (bx, by) = self.pos[a], self.pos[b]
        return math.isqrt(100 * ((ax - bx) ** 2 + (ay - by) ** 2))

    def tube_legal(self, a: int, b: int) -> bool:
        if DET:
            VCLOCK[0] += 2e-7 * (len(self.tubes) + len(self.pos))
        key = (min(a, b), max(a, b))
        if a == b or key in self.tubes or key in self.blacklist or len(self.adj[a]) >= 5 or len(self.adj[b]) >= 5:
            return False
        pa, pb = self.pos[a], self.pos[b]
        x0, x1, y0, y1 = min(pa[0], pb[0]), max(pa[0], pb[0]), min(pa[1], pb[1]), max(pa[1], pb[1])
        for c, pc in self.pos.items():
            if c != a and c != b and x0 <= pc[0] <= x1 and y0 <= pc[1] <= y1 and on_segment(*pa, *pb, *pc):
                return False
        for (c, d) in self.tubes:
            if c == a or c == b or d == a or d == b:
                continue
            pc, pd = self.pos[c], self.pos[d]
            if max(pc[0], pd[0]) < x0 or min(pc[0], pd[0]) > x1 or max(pc[1], pd[1]) < y0 or min(pc[1], pd[1]) > y1:
                continue
            if segments_intersect(pa, pb, pc, pd):
                return False
        return True

    # ------------------------------------------------------------ actions (mirror the referee)
    def do_tube(self, a: int, b: int) -> bool:
        cost = self.tube_cost(a, b)
        if cost > self.resources:
            return False
        if (min(a, b), max(a, b)) in self.legal_ok:
            if (min(a, b), max(a, b)) in self.tubes or len(self.adj[a]) >= 5 or len(self.adj[b]) >= 5:
                return False
        elif not self.tube_legal(a, b):
            return False
        self.resources -= cost
        self.tubes[(min(a, b), max(a, b))] = 1
        self.adj[a] = self.adj[a] | {b}
        self.adj[b] = self.adj[b] | {a}
        self.actions.append(f"TUBE {a} {b}")
        return True

    def do_upgrade(self, a: int, b: int) -> bool:
        key = (min(a, b), max(a, b))
        if key not in self.tubes:
            return False
        cost = self.tube_cost(a, b) * (self.tubes[key] + 1)
        if cost > self.resources:
            return False
        self.resources -= cost
        self.tubes[key] += 1
        self.actions.append(f"UPGRADE {a} {b}")
        return True

    def do_teleport(self, a: int, b: int) -> bool:
        if TELEPORT_COST > self.resources or a in self.tele_used or b in self.tele_used or a == b:
            return False
        self.resources -= TELEPORT_COST
        self.teleports[a] = b
        self.tele_used |= {a, b}
        self.actions.append(f"TELEPORT {a} {b}")
        return True

    def do_pod(self, path: list[int], pid: int | None = None) -> bool:
        if POD_COST > self.resources:
            return False
        if pid is None:
            pid = self.free_pod_id()
        if pid is None or pid in self.pods:
            return False
        if any((min(u, v), max(u, v)) not in self.tubes for u, v in zip(path, path[1:])):
            return False
        self.resources -= POD_COST
        self.pods[pid] = list(path)
        self.fresh.add(pid)
        self.actions.append(f"POD {pid} {' '.join(map(str, path))}")
        return True

    def do_reroute(self, pid: int, old: tuple, path: tuple) -> bool:
        """Changes the path of a pod: rewrites its POD action (moved to the end, after the tubes it may need) if the pod was (re)created
        this turn, otherwise destroys and re-creates it with the same id (net cost POD_COST - POD_REFUND), keeping its priority.
        :param pid: pod id
        :param old: path the change was planned for (the change fails if the pod has another path now)
        :param path: new path
        :return: whether the change was applied"""
        if self.pods.get(pid) != list(old) or any((min(u, v), max(u, v)) not in self.tubes for u, v in zip(path, path[1:])):
            return False
        if pid in self.fresh:
            self.actions.remove(f"POD {pid} " + " ".join(map(str, old)))
        elif self.resources < POD_COST - POD_REFUND:
            return False
        else:
            self.resources -= POD_COST - POD_REFUND
            self.actions.append(f"DESTROY {pid}")
            self.fresh.add(pid)
        self.pods[pid] = list(path)
        self.actions.append(f"POD {pid} " + " ".join(map(str, path)))
        return True

    def do_destroy(self, pid: int) -> bool:
        if pid not in self.pods:
            return False
        path = self.pods.pop(pid)
        if pid in self.fresh:  # (re)created this turn: drop its POD action instead
            self.fresh.discard(pid)
            self.actions.remove(f"POD {pid} " + " ".join(map(str, path)))
            self.resources += POD_COST
        else:
            self.resources += POD_REFUND
            self.actions.append(f"DESTROY {pid}")
        return True

    def free_pod_id(self) -> int | None:
        for i in range(1, 501):
            if i not in self.pods:
                return i
        return None

    # ------------------------------------------------------------ distances
    def distances(self, t: int) -> dict:
        """Computes hop distances of all buildings to the nearest module of type t (teleporters cost 0)."""
        dist = dict.fromkeys(self.pos, INF)
        rev = {v: u for u, v in self.teleports.items()}
        front = [b for b, bt in self.btype.items() if bt == t]
        for b in front:
            dist[b] = 0
        d = 0
        while front:
            # close front under reverse teleporters (0-cost)
            i = 0
            while i < len(front):
                u = front[i]
                if u in rev and dist[rev[u]] > d:
                    dist[rev[u]] = d
                    front.append(rev[u])
                i += 1
            nxt = []
            for u in front:
                for v in self.adj[u]:
                    if dist[v] > d + 1:
                        dist[v] = d + 1
                        nxt.append(v)
            front = nxt
            d += 1
        return dist


def extend_distances(st: State, dist: dict, acts: tuple) -> dict:
    """Updates per-type distance maps after tubes and teleporters of an action bundle were added (distances can only decrease, so a
    label-correcting relaxation from the new links suffices); unchanged maps are shared.
    :param st: state after the bundle was applied
    :param dist: per-type distance maps of the state before the bundle
    :param acts: applied action bundle
    :return: per-type distance maps of st"""
    links = [(a[1], a[2], 1) for a in acts if a[0] == "do_tube"] + [(a[1], a[2], 0) for a in acts if a[0] == "do_teleport"]
    rev = None
    new = {}
    for t, d in dist.items():
        front = []
        for a, b, w in links:
            if w:
                if d[a] + 1 < d[b]:
                    front.append((b, d[a] + 1))
                elif d[b] + 1 < d[a]:
                    front.append((a, d[b] + 1))
            elif d[b] < d[a]:
                front.append((a, d[b]))
        if not front:
            new[t] = d
            continue
        if rev is None:
            rev = {v: u for u, v in st.teleports.items()}
        d = new[t] = dict(d)
        for u, du in front:
            if du < d[u]:
                d[u] = du
        adj = st.adj
        i = 0
        while i < len(front):
            u = front[i][0]
            i += 1
            du = d[u]
            for v in adj[u]:
                if d[v] > du + 1:
                    d[v] = du + 1
                    front.append((v, 0))
            e = rev.get(u)
            if e is not None and d[e] > du:
                d[e] = du
                front.append((e, 0))
    return new


class PairSet:
    """Candidate new tubes (pairs among the KNN nearest neighbours of each building) with vectorized geometric legality checks.
    :var pairs: candidate building pairs (a, b), a < b
    :var cost: tube cost of each pair
    :var xy: pair endpoint coordinates as four arrays (ax, ay, bx, by)
    :var ends: pair endpoint ids as two arrays (a, b)
    :var tubes0: tubes of the turn's initial state
    :var ok0: mask of the pairs legal in the turn's initial state (no building on the segment, no crossing of a tube of tubes0)
    :var rows: tube added during the turn -> mask of the pairs it crosses"""

    def __init__(self, st: State, k: int):
        ids = np.array(list(st.pos), dtype=np.int64)
        xy = np.array([st.pos[b] for b in ids.tolist()], dtype=np.int64)
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, 1 << 40)
        nn = np.argsort(d2, axis=1, kind="stable")[:, :k]
        i, j = np.repeat(np.arange(len(ids)), nn.shape[1]), nn.ravel()
        pr = np.unique(np.stack([np.minimum(ids[i], ids[j]), np.maximum(ids[i], ids[j])], 1)[i != j], axis=0)
        self.pairs = list(map(tuple, pr.tolist()))
        self.cost = [st.tube_cost(a, b) for a, b in self.pairs]
        pa, pb = np.array([st.pos[a] for a, _ in self.pairs], dtype=np.int64), np.array([st.pos[b] for _, b in self.pairs], dtype=np.int64)
        self.xy, self.ends = (pa[:, 0], pa[:, 1], pb[:, 0], pb[:, 1]), (pr[:, 0], pr[:, 1])
        ax, ay, bx, by = (v[:, None] for v in self.xy)
        cx, cy = xy[None, :, 0], xy[None, :, 1]
        on = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax) == 0) & (np.minimum(ax, bx) <= cx) & (cx <= np.maximum(ax, bx)) & \
            (np.minimum(ay, by) <= cy) & (cy <= np.maximum(ay, by)) & (ids[None, :] != pr[:, :1]) & (ids[None, :] != pr[:, 1:])
        self.tubes0 = set(st.tubes)
        self.ok0 = ~on.any(axis=1) & ~self.crossing(st, list(st.tubes))
        self.rows = {}
        if DET:
            VCLOCK[0] += 2e-3 + 3e-8 * len(self.pairs) * (len(ids) + len(st.tubes))

    def crossing(self, st: State, tubes: list) -> np.ndarray:
        """Finds the pairs crossing any of the given tubes (tubes sharing an endpoint with a pair do not count).
        :param st: state (building positions)
        :param tubes: tube keys
        :return: boolean mask over the pairs"""
        if not tubes:
            return np.zeros(len(self.pairs), dtype=bool)
        q = np.array([st.pos[a] + st.pos[b] for a, b in tubes], dtype=np.int64)
        ta, tb = np.array([a for a, _ in tubes], dtype=np.int64)[None, :], np.array([b for _, b in tubes], dtype=np.int64)[None, :]
        a, b = self.ends[0][:, None], self.ends[1][:, None]
        p1x, p1y, p2x, p2y = (v[:, None] for v in self.xy)
        p3x, p3y, p4x, p4y = q[None, :, 0], q[None, :, 1], q[None, :, 2], q[None, :, 3]
        d1, d2 = np.sign((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)), np.sign((p4x - p3x) * (p2y - p3y) - (p4y - p3y) * (p2x - p3x))
        d3, d4 = np.sign((p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)), np.sign((p2x - p1x) * (p4y - p1y) - (p2y - p1y) * (p4x - p1x))
        inter = (d1 * d2 < 0) & (d3 * d4 < 0)
        for d, cx, cy, ux, uy, vx, vy in ((d1, p1x, p1y, p3x, p3y, p4x, p4y), (d2, p2x, p2y, p3x, p3y, p4x, p4y), (d3, p3x, p3y, p1x, p1y, p2x, p2y),
                                          (d4, p4x, p4y, p1x, p1y, p2x, p2y)):
            inter |= (d == 0) & (np.minimum(ux, vx) <= cx) & (cx <= np.maximum(ux, vx)) & (np.minimum(uy, vy) <= cy) & (cy <= np.maximum(uy, vy))
        return (inter & ~((a == ta) | (a == tb) | (b == ta) | (b == tb))).any(axis=1)

    def legal(self, st: State) -> dict:
        """Computes the candidate new tubes legal in state st (a state of the current turn).
        :param st: state
        :return: adjacency dict building -> list of (neighbour, tube cost) over the legal new tubes"""
        ok = self.ok0
        for key in st.tubes.keys() - self.tubes0:
            r = self.rows.get(key)
            if r is None:
                r = self.rows[key] = self.crossing(st, [key])
                if DET:
                    VCLOCK[0] += 7e-5 + 5e-9 * len(self.pairs)
            ok = ok & ~r
        full = {u for u, vs in st.adj.items() if len(vs) >= 5}
        adj = defaultdict(list)
        sel = np.flatnonzero(ok).tolist()
        if DET:
            VCLOCK[0] += 5e-5 + 1e-6 * len(sel)
        for i in sel:
            a, b = self.pairs[i]
            if a in full or b in full or (a, b) in st.tubes or (a, b) in st.blacklist:
                continue
            c = self.cost[i]
            adj[a].append((b, c))
            adj[b].append((a, c))
        return adj


class SimResult:
    __slots__ = ("score", "arrived", "total", "stuck", "pod_load", "dist", "mod_cnt", "group", "leg_load", "flow", "wait", "hop_flow", "pass_flow", "pres")


def bundle_cost(st: State, acts: tuple) -> int:
    """Returns the resource cost of an action bundle applied to state st (negative for pod destructions).
    :param st: network state
    :param acts: action bundle
    :return: cost"""
    c = 0
    for a in acts:
        if a[0] == "do_tube":
            c += st.tube_cost(a[1], a[2])
        elif a[0] == "do_upgrade":
            c += st.tube_cost(a[1], a[2]) * (st.tubes[(min(a[1], a[2]), max(a[1], a[2]))] + 1)
        elif a[0] == "do_pod":
            c += POD_COST
        elif a[0] == "do_teleport":
            c += TELEPORT_COST
        else:
            c -= POD_REFUND
    return c


def balance_pts(c: int, n: int) -> int:
    """Returns the total balancing points of n arrivals at a module already reached by c astronauts this month."""
    m = min(n, 50 - c)
    return m * (101 - 2 * c - m) // 2 if m > 0 else 0


def simulate(st: State, want_detail: bool = False, dist: dict | None = None) -> SimResult:
    """Simulates one month exactly as the referee does and returns the score (and diagnostics).
    Astronauts are grouped per (building, type) into id-sorted lists; since all astronauts of a type at a building share the same
    options, each pod takes a prefix of every type list, and pod j (in id order) takes the 10 smallest ids among the remaining
    astronauts it is useful for.
    :param st: network state
    :param want_detail: whether to fill the diagnostics: stuck astronauts, pod loads (per pod and per departure index), distance maps,
        module arrival counts, per-group arrivals, arrival flows (pad, type, module), (last departure building, module) and (any departure
        building on the route, module), waiting astronaut-days per (building, type) and presences
    :param dist: precomputed per-type distance maps (valid when the topology of st is unchanged)
    :return: simulation result"""
    if DET:
        VCLOCK[0] += VSCALE * 7e-5 + VSCALE * 1e-5 * sum(map(len, st.pods.values())) + (4e-8 * len(st.pos) * len(st.groups or ()) * 5 if dist is None else 0)
    SIM_COUNT[0] += 1
    if st.groups is None:
        st.groups = {p: {t: [p * 1000 + i for i, x in enumerate(lst) if x == t] for t in set(lst)} for p, lst in st.pad_astronauts.items()}
    at = {p: dict(g) for p, g in st.groups.items()}
    if dist is None:
        dist = {t: st.distances(t) for t in {t for g in at.values() for t in g}}
    btype, tubes = st.btype, st.tubes
    tele = list(st.teleports.items())
    pods = sorted(st.pods.items())
    paths = [path for _, path in pods]
    keys = [[(u, v) if u < v else (v, u) for u, v in zip(path, path[1:])] for path in paths]
    ends = [len(path) - 1 for path in paths]
    loops = [path[0] == path[-1] for path in paths]
    npods = len(pods)
    pidx = [0] * npods
    pod_load = [0] * npods
    leg_load = [[0] * len(path) for path in paths] if want_detail else None  # boardings per pod and departure index
    cnt = defaultdict(int)
    ucache = {}
    arr_log, hop_log = [], []
    wlog = []
    via = defaultdict(list) if want_detail else None  # astronaut id -> departure buildings so far
    score = arrived = 0
    for day in range(1, DAYS + 1):
        spd = 50 - day
        for e, x in tele:
            g = at.get(e)
            if not g:
                continue
            moved = [t for t in g if dist[t][e] < INF and dist[t][x] <= dist[t][e]]
            if not moved:
                continue
            xt = btype[x]
            gx = at.get(x)
            if gx is None:
                gx = at[x] = {}
            for t in moved:
                lst = g.pop(t)
                if want_detail:
                    for i in lst:
                        via[i].append(e)
                if xt == t:
                    n, c = len(lst), cnt[x]
                    score += n * (spd + 1) + balance_pts(c, n)  # teleport arrival: day - 1 days needed
                    cnt[x] = c + n
                    arrived += n
                    if want_detail:
                        arr_log.append((day, t, lst, x, e, 1))
                elif t in gx:
                    gx[t] = sorted(gx[t] + lst)
                else:
                    gx[t] = lst
        used = {}
        departing = {}
        for k in range(npods):
            i = pidx[k]
            if i >= ends[k]:
                continue
            key = keys[k][i]
            c = used.get(key, 0)
            if c < tubes[key]:
                used[key] = c + 1
                u = paths[k][i]
                if u in departing:
                    departing[u].append(k)
                else:
                    departing[u] = [k]
        landed = []
        for b, deps in departing.items():
            g = at.get(b)
            if g:
                taken = {}
                for k in deps:
                    v = paths[k][pidx[k] + 1]
                    ut = ucache.get((b, v))
                    if ut is None:
                        ut = ucache[(b, v)] = {t for t, d in dist.items() if d[v] < d[b]}
                    cands = []
                    for t, lst in g.items():
                        if t in ut:
                            p = taken.get(t, 0)
                            if p < len(lst):
                                cands.append((t, lst, p))
                    if not cands:
                        continue
                    if len(cands) == 1:
                        t, lst, p = cands[0]
                        q = min(p + POD_CAPACITY, len(lst))
                        landed.append((v, t, lst[p:q], b))
                        taken[t] = q
                        pod_load[k] += q - p
                        if want_detail:
                            leg_load[k][pidx[k]] += q - p
                        continue
                    pool = []
                    for t, lst, p in cands:
                        pool += lst[p:p + POD_CAPACITY]
                    if len(pool) > POD_CAPACITY:
                        pool.sort()
                        theta = pool[POD_CAPACITY - 1]
                        for t, lst, p in cands:
                            q = bisect_right(lst, theta, p, min(p + POD_CAPACITY, len(lst)))
                            if q > p:
                                landed.append((v, t, lst[p:q], b))
                                taken[t] = q
                        pod_load[k] += POD_CAPACITY
                        if want_detail:
                            leg_load[k][pidx[k]] += POD_CAPACITY
                    else:
                        for t, lst, p in cands:
                            landed.append((v, t, lst[p:], b))
                            taken[t] = len(lst)
                        pod_load[k] += len(pool)
                        if want_detail:
                            leg_load[k][pidx[k]] += len(pool)
                for t, q in taken.items():
                    lst = g[t]
                    if q == len(lst):
                        del g[t]
                    else:
                        g[t] = lst[q:]
            for k in deps:
                pidx[k] += 1
                if loops[k] and pidx[k] == ends[k]:
                    pidx[k] = 0
        if want_detail and day < DAYS:
            hop_log += [(v, t, day, lst) for v, t, lst, _ in landed if btype[v] != t]
        touched = set()
        for v, t, lst, u in landed:
            if want_detail:
                for i in lst:
                    via[i].append(u)
            if btype[v] == t:
                n, c = len(lst), cnt[v]
                score += n * spd + balance_pts(c, n)
                cnt[v] = c + n
                arrived += n
                if want_detail:
                    arr_log.append((day, t, lst, v, u, 0))
            else:
                gv = at.get(v)
                if gv is None:
                    at[v] = {t: lst}
                elif t in gv:
                    gv[t] = gv[t] + lst
                    touched.add((v, t))
                else:
                    gv[t] = lst
        for v, t in touched:
            at[v][t].sort()
        if want_detail:
            wlog += [(b, t, lst) for b, g in at.items() for t, lst in g.items() if lst and dist[t][b] < INF]
    r = SimResult()
    r.score, r.arrived, r.total, r.dist = score, arrived, sum(map(len, st.pad_astronauts.values())), dist
    if want_detail:
        r.stuck = {(b, t): len(lst) for b, g in at.items() for t, lst in g.items() if lst}
        r.pod_load, r.mod_cnt, r.leg_load = dict(zip([p for p, _ in pods], pod_load)), cnt, dict(zip([p for p, _ in pods], leg_load))
        r.wait = defaultdict(int)
        lost = defaultdict(set)  # type -> ids of astronauts that never arrive
        for g in at.values():
            for t, lst in g.items():
                lost[t].update(lst)
        if DET:
            VCLOCK[0] += 1e-7 * sum(len(lst) for _, _, lst in wlog)
        for b, t, lst in wlog:
            r.wait[(b, t)] += len(lst) - (len(lost[t].intersection(lst)) if t in lost else 0)
        if DET:
            VCLOCK[0] += BAL_VH * sum(map(len, via.values()))
        r.group, r.flow, r.hop_flow, r.pass_flow = defaultdict(lambda: [0, 0]), defaultdict(int), defaultdict(int), defaultdict(int)
        for day, t, lst, m, u, _ in arr_log:
            r.hop_flow[(u, m)] += len(lst)
            for i in lst:
                g = r.group[(i // 1000, t)]
                g[0] += 1
                g[1] += day
                r.flow[(i // 1000, t, m)] += 1
                for b in via[i]:
                    r.pass_flow[(b, m)] += 1
        r.pres = presences(st, arr_log, hop_log)
    return r


def presences(st: State, arr_log: list, hop_log: list) -> dict:
    """Aggregates, per (building, type), the astronauts standing at a non-target building before the last day (at their pad on day 0
    or landed there by pod), with the speed points a teleporter from there taken the next day would gain and their actual modules.
    :param st: simulated state
    :param arr_log: arrivals (day, type, astronaut ids, module, departure building, 1 if by teleporter else 0)
    :param hop_log: landings at non-target buildings (building, type, day, astronaut ids)
    :return: (building, type) -> [astronaut count, speed gain, {module: arrivals}]"""
    if DET:
        VCLOCK[0] += 7e-7 * sum(len(lst) for _, _, _, lst in hop_log) + 5e-7 * sum(len(lst) for _, _, lst, _, _, _ in arr_log)
    arr = {}
    for day, t, lst, m, _, tp in arr_log:
        days = (day - tp, m)
        for i in lst:
            arr[i] = days
    pres = {}
    for v, t, day, lst in [(p, t, 0, lst) for p, g in st.groups.items() for t, lst in g.items()] + hop_log:
        if (e := pres.get((v, t))) is None:
            e = pres[(v, t)] = [0, 0, defaultdict(int)]
        e[0] += len(lst)
        g, mods = 0, e[2]
        for i in lst:
            if (a := arr.get(i)) is None:
                g += 50 - day
            else:
                g += a[0] - day
                mods[a[1]] += 1
        e[1] += g
    return pres


CLOCK = time.process_time if os.environ.get("SELENIA_CPU_CLOCK") else time.perf_counter
if DET:
    CLOCK = lambda: VCLOCK[0]


class Bot:
    def __init__(self):
        VCLOCK[0] = 0.
        self.st = State()
        self.month = 0
        self.start = 0.
        self.issued_tubes = []  # tubes requested last turn
        self.blacklist = set()  # tubes rejected by the referee (e.g. passing through a future building)
        self.order_cache, self.mid_cache, self.tgt_cache, self.geo = {}, {}, {}, None  # position-only geometry caches, reset when buildings appear
        self.pairset = None  # candidate new tubes of route searches, rebuilt every turn
        self.tord_cache = {}  # (building, type) -> all modules of the type by distance, reset when buildings appear
        self.bal_bad = set()  # tubes found illegal in the current greedy run by the balancing candidates
        self.gen0 = 0.  # recent duration of a full candidate generation
        self.t0 = None  # start of the time limit of the first turn when it precedes play_turn (set by main)
        self.tele_geo = None  # (building count, module -> distance to the nearest pad, type -> module ids)
        self.hist_new, self.hist_inc, self.prev_after, self.res = [], [], 0, 0  # new buildings, inferred income per month, last leftover, reserve
        self.new_rows = []  # raw input lines of the buildings that appeared this turn (for the log)
        self.expect = None  # (tubes, pods, teleporters) expected in the next input, to detect rejected actions
        self.cum = 0  # predicted score of all months so far

    # ------------------------------------------------------------ parsing
    def parse(self, lines: list[str]):
        it = iter(lines)
        st = self.st
        st.resources = int(next(it))
        st.tubes, st.adj, st.teleports, st.tele_used, st.pods = {}, defaultdict(set), {}, set(), {}
        for _ in range(int(next(it))):
            a, b, c = map(int, next(it).split())
            if c == 0:
                st.teleports[a] = b
                st.tele_used |= {a, b}
            else:
                st.tubes[(min(a, b), max(a, b))] = c
                st.adj[a].add(b)
                st.adj[b].add(a)
        for _ in range(int(next(it))):
            v = list(map(int, next(it).split()))
            st.pods[v[0]] = v[2:2 + v[1]]
        self.new_rows = []
        for _ in range(int(next(it))):
            row = next(it).strip()
            self.new_rows.append(row)
            v = list(map(int, row.split()))
            st.pos[v[1]] = (v[2], v[3])
            self.order_cache, self.mid_cache, self.tgt_cache, self.geo, self.tord_cache = {}, {}, {}, None, {}
            st.btype[v[1]] = v[0]
            if v[0] == 0:
                st.pad_astronauts[v[1]] = v[5:5 + v[4]]
                st.groups = None
        st.actions, st.fresh = [], set()
        self.bal_bad = set()

    def play_turn(self, lines: list[str]) -> str:
        self.start = CLOCK() if self.month or self.t0 is None else self.t0
        boot = time.perf_counter() - T_START if not self.month else 0
        n_old = len(self.st.pos)
        self.parse(lines)
        if LOG and self.expect is not None:
            try:
                self.log_desync()
            except Exception:  # diagnostics must never cost the game
                traceback.print_exc()
        self.hist_new.append(len(self.st.pos) - n_old)
        self.hist_inc.append(self.st.resources - self.prev_after - self.prev_after // 10)
        self.res = self.reserve()
        for key in self.issued_tubes:
            if key not in self.st.tubes:
                self.blacklist.add(key)
        self.st.blacklist = self.blacklist
        start = self.st
        try:
            self.pairset = PairSet(self.st, KNN) if len(self.st.pos) > 1 else None
            self.plan()
        except Exception:  # a bug in a rare planning path must not forfeit the game: play WAIT this month (plan never mutates start)
            traceback.print_exc()
            log(f"ERR{self.month + 1:02d} planning failed, playing WAIT")
            self.st = start
        self.prev_after = self.st.resources
        self.issued_tubes = [(min(a, b), max(a, b)) for a, b in (map(int, s.split()[1:]) for s in self.st.actions if s.startswith("TUBE"))]
        out = ";".join(self.st.actions) or "WAIT"
        if LOG:
            try:
                self.log_turn(start.resources, out, boot)
            except Exception:  # diagnostics must never cost the game
                traceback.print_exc()
        self.month += 1
        return out

    def log_turn(self, res0: int, out: str, boot: float):
        """Writes this turn's summary and the encoded new buildings and actions to stderr (decoded by logtool.py).
        :param res0: resources at the start of the turn
        :param out: action line sent to the referee
        :param boot: seconds between the module import and the first turn (first turn only)"""
        t_plan = self.elapsed()
        st, m = self.st, self.month + 1
        r = simulate(st)
        self.cum += r.score
        log(f"T{m:02d} t={self.elapsed() * 1000:.0f} plan={t_plan * 1000:.0f} boot={boot * 1000:.0f} res={res0} inc={self.hist_inc[-1]} left={st.resources} "
            f"new={len(self.new_rows)} b={len(st.pos)} pred={r.score} arr={r.arrived}/{r.total} cum={self.cum} act={len(st.actions)} sims={SIM_COUNT[0]} "
            f"tubes={len(st.tubes)} pods={len(st.pods)} tele={len(st.teleports)}")
        SIM_COUNT[0] = 0
        log_blob(f"M{m:02d}", "\n".join([str(res0)] + self.new_rows))
        log_blob(f"A{m:02d}", out)
        self.expect = (dict(st.tubes), {k: list(v) for k, v in st.pods.items()}, dict(st.teleports))
        if m == MONTHS:
            log(f"END cum={self.cum}")

    def log_desync(self):
        """Reports differences between the network the bot expected after its last actions and the one in this turn's input."""
        tubes, pods, tele = self.expect
        st, m, msgs = self.st, self.month + 1, []
        if tubes != st.tubes:
            msgs.append(f"tubes missing {sorted(set(tubes) - set(st.tubes))[:8]} extra {sorted(set(st.tubes) - set(tubes))[:8]} "
                        f"cap {[(k, tubes[k], st.tubes[k]) for k in tubes if k in st.tubes and tubes[k] != st.tubes[k]][:8]}")
        if pods != st.pods:
            msgs.append(f"pods differ {[(k, pods.get(k), st.pods.get(k)) for k in sorted(set(pods) | set(st.pods)) if pods.get(k) != st.pods.get(k)][:5]}")
        if tele != st.teleports:
            msgs.append(f"teleporters expected {sorted(tele.items())[:8]} got {sorted(st.teleports.items())[:8]}")
        if self.hist_inc and st.resources - self.prev_after - self.prev_after // 10 < 0:
            msgs.append(f"resources {st.resources} below expected minimum {self.prev_after + self.prev_after // 10}")
        for msg in msgs:
            log(f"X{m:02d} {msg}")

    def elapsed(self) -> float:
        return CLOCK() - self.start

    # ------------------------------------------------------------ strategy
    def plan(self):
        """Builds this month's network: greedy construction, then (time permitting) rounds of dropping purchased bundles that no longer
        pay off followed by greedy re-construction."""
        budget = full = (BUDGET0 if self.month == 0 else BUDGET1) - TIME_MARGIN
        start = self.st
        months_left = MONTHS - self.month
        st, chosen, tops = self.greedy(start, [], budget)
        inc = sum(self.hist_inc[1:]) / self.month if self.month else 0  # mean observed income
        hold = HOLD and months_left > HOLD_ML and (HOLD_INC or not inc) and st.resources < POD_COST  # money-bound
        if hold:
            budget *= HOLD_FRAC
        j_best = simulate(st).score * months_left + self.util(st.resources)
        for acts in tops[1:PILOT_K]:
            if self.elapsed() > budget * PILOT_FRAC:
                break
            s2 = start.copy()
            if not all(getattr(s2, a[0])(*a[1:]) for a in acts):
                continue
            s3, ch3, _ = self.greedy(s2, [acts], budget * PILOT_FRAC)
            if (j := simulate(s3).score * months_left + self.util(s3.resources)) > j_best:
                st, chosen, j_best = s3, ch3, j
        while DROP and chosen and self.elapsed() < budget * DROP_FRAC:
            base = simulate(st)
            best = None
            for i in range(len(chosen)):
                if self.elapsed() > budget * DROP_FRAC:
                    break
                rest = chosen[:i] + chosen[i + 1:]
                s2 = start.copy()
                if not all(getattr(s2, a[0])(*a[1:]) for acts in rest for a in acts):
                    continue
                r = simulate(s2)
                j = (r.score - base.score) * months_left + self.util(s2.resources) - self.util(st.resources)
                if j > 0 and (best is None or j > best[0]):
                    best = (j, s2, rest)
            if best is None:
                break
            st, chosen, _ = self.greedy(best[1], best[2], budget)
        if hold and chosen and st.resources < POD_COST:
            # compare with keeping all resources for HOLD_K months (interest) and building a larger network at once
            # (with income, the build-now path also spends the forecast income of the next HOLD_K months on top of its network)
            g, v, sa = 1.1 ** HOLD_K, start.copy(), simulate(st).score
            v.resources, fut = int(start.resources * g + inc * (g - 1) * 10), int(inc * (g - 1) * 10)
            sb, sa2 = simulate(self.greedy(v, [], (budget + full) / 2 if fut > 0 else full)[0]).score, sa
            if fut > 0:
                a2 = st.copy()
                a2.resources = int(st.resources * g) + fut
                sa2 = simulate(self.greedy(a2, [], full)[0]).score
            if sb * (months_left - HOLD_K) + simulate(start).score * HOLD_K > sa2 * (months_left - HOLD_K) + sa * HOLD_K:
                st = start
        self.st = st

    def greedy(self, st: State, chosen: list, budget: float) -> tuple:
        """Extends a state greedily by lazy (CELF-style) evaluation of a persistent candidate pool with the exact simulator.
        Fresh candidates are evaluated in the order of their analytic estimates while the step's share of the remaining time lasts,
        afterwards only while their estimate scaled by a calibration factor (a high quantile of the observed actual/estimated value
        ratios) exceeds the best exact value; evaluated candidates are re-evaluated lazily when they reach the top of the heap.
        :param st: state to extend
        :param chosen: bundles already applied to st this month (extended in place)
        :param budget: time budget of the turn
        :return: final state, the list of all bundles applied this month and the bundles worth applying at the first step, best first"""
        def evaluate(acts: tuple):
            for a in acts:
                if a[0] == "do_tube" and (key := (min(a[1], a[2]), max(a[1], a[2]))) not in st.legal_ok:
                    if key in bad or not st.tube_legal(a[1], a[2]):
                        bad.add(key)
                        del pool[acts]
                        dead.add(acts)
                        return
                    st.legal_ok.add(key)
            c = pool[acts]
            s2 = st.copy()
            if not all(getattr(s2, a[0])(*a[1:]) for a in acts):
                del pool[acts]
                dead.add(acts)
                return
            r = simulate(s2, False, extend_distances(s2, base.dist, acts) if any(a[0] in TOPO_ACTS for a in acts) else base.dist)
            vc = DETOUR_VCOST * sum(a[0] == "do_reroute" and len(a[3]) > len(a[2]) for a in acts)  # longer loops: virtual cost
            cost = st.resources - s2.resources + vc
            value = INF if cost < 0 and r.score >= base.score else (r.score - base.score) * months_left / max(cost, 1)
            if c[1] < 0 and c[4] > 0 and cost > 0:
                ratios.append(value / c[4])
            c[:4], c[5] = (value, ver, s2, r), vc
            heappush(evald, (-value, acts))

        months_left = MONTHS - self.month
        base = simulate(st, True)
        # pool: actions -> [value, version, state, result, estimate, virtual cost]
        # the turn's main run always generates (it starts early and re-measures gen0 every turn); secondary runs start only if a full
        # candidate generation still fits
        pool, dead, ver, ratios, alpha, fresh, evald, sig, bad, tops = {}, set(), 0, [], 2., [], [], {}, set(), []
        gen_time = 0. if st is self.st else self.gen0
        st.legal_ok = set()
        self.bal_bad = set()  # tubes only get added within one greedy run, so its illegal tubes stay illegal
        while self.elapsed() + gen_time < budget:
            t0 = self.elapsed()
            for acts, est in self.candidates(st, base, sig).items():
                if (c := pool.get(acts)) is not None:
                    if REFRESH and 0 <= c[1] < ver:
                        c[4] = est * months_left
                        heappush(fresh, (-c[4], acts))
                elif acts not in dead and not any(a[0] == "do_tube" and (min(a[1], a[2]), max(a[1], a[2])) in bad for a in acts):
                    pool[acts] = [0, -1, None, None, est * months_left, 0]
                    heappush(fresh, (-est * months_left, acts))
            gen_time = self.elapsed() - t0
            if not ver:
                self.gen0 = max(self.gen0 * 0.8, gen_time)  # cost of a full (unsigned) candidate generation, to avoid overshooting
            if len(ratios) >= 20:
                ratios.sort()
                alpha = max(ratios[min(int(len(ratios) * QUANTILE), len(ratios) - 1)], 0.05)
            spent = self.st.resources - st.resources
            steps = min(max(st.resources * len(chosen) / spent if spent > 0 else st.resources / 1500, 1), MAX_STEPS)
            step_end = self.elapsed() + max(STEP_FRAC * (budget - self.elapsed()) / steps, STEP_MIN)
            u0 = self.util(st.resources)
            thr = (u0 - self.util(st.resources - POD_COST)) / POD_COST  # value needed by a pod-sized bundle to be accepted
            n_fresh = 0
            while self.elapsed() < budget:
                if evald:
                    c = pool.get(acts := evald[0][1])
                    if c is None or -evald[0][0] != c[0]:
                        heappop(evald)
                        continue
                    if c[1] != ver:
                        heappop(evald)
                        evaluate(acts)
                        continue
                if fresh:
                    c = pool.get(acts := fresh[0][1])
                    if c is None or c[1] == ver:
                        heappop(fresh)
                        continue
                    if n_fresh < FRESH_MIN or self.elapsed() < step_end or -fresh[0][0] * alpha > (top := -evald[0][0] if evald else 0) \
                            or FILL and top < thr:
                        heappop(fresh)
                        evaluate(acts)
                        n_fresh += 1
                        continue
                break
            ok = [acts for acts, c in pool.items()
                  if c[1] == ver and (c[0] == INF or c[0] > 0 and (c[3].score - base.score) * months_left >= u0 - self.util(c[2].resources - c[5]))]
            best = max(ok, key=lambda a: pool[a][0], default=None)
            if ver == 0:
                tops = sorted(ok, key=lambda a: -pool[a][0])
            if best is None:
                break
            c = pool.pop(best)
            st = c[2]
            st.legal_ok = set()
            chosen.append(best)
            evald = [e for e in evald if e[1] != best]
            heapify(evald)
            base = simulate(st, True, c[3].dist)
            ver += 1
        return st, chosen, tops

    def candidates(self, st: State, base: SimResult, sig: dict) -> dict:
        """Generates candidate action bundles with analytic estimates of their monthly gain per cost: pods, tubes, chains and teleporters
        for stuck astronaut groups (plus re-routings of existing pods through the needed tubes), direct links for slow groups, routed
        bundles, speed-ups of served groups, balancing diversions, pod shortenings and destructions, and teleporters from the buildings
        astronauts stand at. Stuck groups and route jobs whose situation is unchanged since their last generation are skipped, since
        their candidates are still in the persistent pool.
        :param st: current state
        :param base: detailed simulation result of st
        :param sig: (building, type) or route job -> situation at the last generation, updated in place
        :return: dict actions (hashable tuple) -> estimated monthly gain per resource spent"""
        def link_gain(b: int, x: int) -> float:
            s = 0
            for t, n in stuck_at[b].items():
                d = base.dist[t]
                if d[x] < d[b]:
                    s += n if d[x] == 0 else n / 2
            return min(s, 100) * 60

        stuck_at = defaultdict(dict)
        for (b, t), n in base.stuck.items():
            stuck_at[b][t] = n
        edge_pods, node_pods = defaultdict(int), defaultdict(int)
        for path in st.pods.values():
            for key in {(min(x, y), max(x, y)) for x, y in zip(path, path[1:])}:
                edge_pods[key] += 1
            for x in set(path):
                node_pods[x] += 1
        cands = {}
        if DET:  # calibrated to simulate's real/virtual ratio (about 0.4) together with the per-regenerated-group charge below
            VCLOCK[0] += 4e-6 * (len(base.stuck) + len(base.group)) + 2.5e-6 * sum(map(len, st.pods.values()))
        pod_at, users = defaultdict(list), defaultdict(list)  # building -> [(looping pod, index of its first visit, its tubes)], tube -> pods using it
        if detours := DETOURS and st.resources < DETOUR_RES:
            for pid, path in st.pods.items():
                for key in (keys := {(min(x, y), max(x, y)) for x, y in zip(path, path[1:])}):
                    users[key].append(pid)
                if path[0] == path[-1] and len(path) + 2 <= DETOUR_LEN:
                    for x in set(path[:-1]):
                        pod_at[x].append((pid, path.index(x), keys))
            for lst in pod_at.values():
                lst.sort(key=lambda e: (base.pod_load.get(e[0], 0), len(st.pods[e[0]])))
            if DET:
                VCLOCK[0] += 1e-5 + 1e-6 * sum(map(len, st.pods.values()))
        for (b, t), n in sorted(base.stuck.items(), key=lambda kv: -kv[1]):
            d = base.dist[t]
            s, old = (n, d, st.adj.get(b), node_pods[b], b in st.tele_used, detours), sig.get((b, t))
            if old is not None and old[0] == s[0] and old[1] is d and old[2] is s[2] and old[3:] == s[3:]:
                continue
            sig[(b, t)] = s
            if DET:
                VCLOCK[0] += 6e-5
            for v in st.adj[b]:
                if d[v] < d[b]:
                    key = (min(b, v), max(b, v))
                    acts, cost = (), POD_COST
                    if edge_pods[key] >= st.tubes[key]:
                        acts, cost = (("do_upgrade", b, v),), cost + st.tube_cost(b, v) * (st.tubes[key] + 1)
                    g = link_gain(b, v) / cost
                    for path in ((b, v, b), (v, b, v)):
                        cands[acts + (("do_pod", path),)] = g
                    if detours:
                        cands.update(self.detours(st, base, pod_at, (b, v), link_gain(b, v) / 60, (), 0, True))
                    if SPLIT and acts and detours:
                        cands.update(self.splits(st, users[key], b, v, link_gain(b, v)))
            order = self.by_dist(b)
            targets = self.targets(b, t)
            near = [x for x in order if d[x] < d[b] - 1][:NN]
            for x in targets + near:
                if (min(b, x), max(b, x)) not in st.tubes and (acts := (("do_tube", b, x), ("do_pod", (b, x, b)))) not in cands:
                    cands[acts] = link_gain(b, x) / (st.tube_cost(b, x) + POD_COST)
                    if detours and DETOUR_NEW:
                        cands.update(self.detours(st, base, pod_at, (b, x), link_gain(b, x) / 60, acts[:1], st.tube_cost(b, x), DETOUR_NB))
            if b in st.pad_astronauts and b not in st.tele_used:
                for m in targets[:2]:
                    if m not in st.tele_used:
                        cands[(("do_teleport", b, m),)] = sum(n for tt, n in stuck_at[b].items() if tt == st.btype[m]) * 70 / TELEPORT_COST
            for m in targets:
                for x in self.mids(b, m):
                    acts = tuple(("do_tube", u, w) for u, w in ((b, x), (x, m)) if (min(u, w), max(u, w)) not in st.tubes) + (("do_pod", (b, x, m, x, b)),)
                    if acts not in cands:
                        cands[acts] = min(n, 50) * 60 / (sum(st.tube_cost(a[1], a[2]) for a in acts[:-1]) + POD_COST)
                        if detours and DETOUR_CHAIN:
                            cands.update(self.detours(st, base, pod_at, (b, x, m), n, acts[:-1], sum(st.tube_cost(a[1], a[2]) for a in acts[:-1]), False))
                    if CH2 and n > CH2_N:
                        acts2 = acts[:-1] + (("do_pod", (b, x, b)), ("do_pod", (x, m, x)))
                        if acts2 not in cands:
                            cands[acts2] = min(n, 100) * 60 / (sum(st.tube_cost(a[1], a[2]) for a in acts[:-1]) + 2 * POD_COST)
        if SLOW:
            for (p, t), (c, sd) in base.group.items():
                if sd > SLOW_DAY * c:
                    gain = min(c, 100) * (sd / c - 1)
                    for m in self.targets(p, t)[:3]:
                        key = (min(p, m), max(p, m))
                        if key not in st.tubes:
                            cands.setdefault((("do_tube", p, m), ("do_pod", (p, m, p))), gain / (st.tube_cost(p, m) + POD_COST))
                        elif edge_pods[key] < st.tubes[key]:
                            cands.setdefault((("do_pod", (p, m, p)),), gain / POD_COST)
                        else:
                            cands.setdefault((("do_upgrade", p, m), ("do_pod", (p, m, p))), gain / (POD_COST + st.tube_cost(p, m) * (st.tubes[key] + 1)))
                        if p not in st.tele_used and m not in st.tele_used:
                            cands.setdefault((("do_teleport", p, m),), c * (sd / c - 0.5) / TELEPORT_COST)
        if self.pairset is not None:
            self.route_candidates(st, base, sig, edge_pods, cands)
        if SERVED:
            self.served_candidates(st, base, edge_pods, cands)
        if BAL:
            self.balance_candidates(st, base, cands, edge_pods)
        for pid, load in base.pod_load.items():
            if not load:
                cands[(("do_destroy", pid),)] = 1
        if detours and SHORTEN:
            cands.update(self.shortenings(st, base))
        if TELE and base.pres and st.resources >= TELEPORT_COST:
            self.tele_candidates(st, base, cands, edge_pods)
        return cands

    def detours(self, st: State, base: SimResult, pod_at: dict, r: tuple, s: float, pre: tuple, pre_cost: int, via_nb: bool) -> dict:
        """Returns re-routing candidates inserting a traversal of route r = (b, ..., v) into looping pods that visit b (round trip
        b ... v ... b), v (v ... b ... v) or, with via_nb, a tube neighbour y of b (y b v y for a single tube b-v with v-y a tube, else
        y b ... v ... b y), least loaded pods first.
        :param st: current state
        :param base: detailed simulation result of st
        :param pod_at: building -> [(looping pod id, index of its first visit, its tubes)], least loaded first
        :param r: route from the building with stuck astronauts b to the closer building v (its tubes exist or are built by pre)
        :param s: estimated number of astronauts per month a new pod over r would serve
        :param pre: actions applied before the re-routing
        :param pre_cost: cost of pre
        :param via_nb: whether to consider pods visiting tube neighbours of b
        :return: dict actions -> estimated monthly gain per resource spent"""
        b, v = r[0], r[-1]
        opts = [(e, r[1:] + r[-2::-1]) for e in pod_at[b][:DETOUR_PODS]] + [(e, r[-2::-1] + r[1:]) for e in pod_at[v][:DETOUR_PODS]]
        if via_nb:
            for y in st.adj[b]:
                if y not in r:
                    opts += [(e, r + (y,) if len(r) == 2 and y in st.adj[v] else r + r[-2::-1] + (y,)) for e in pod_at[y][:DETOUR_PODS]]
        res, done, key = {}, set(), (min(r[-2], v), max(r[-2], v))
        for (pid, i, keys), ins in opts:
            path = st.pods[pid]
            if pid in done or len(path) + len(ins) > DETOUR_LEN or key in keys:
                continue
            done.add(pid)
            npath = (*path[:i + 1], *ins, *path[i + 1:])
            hops = len(npath) - 1
            est = min(s, POD_CAPACITY * DAYS / hops) * 60 - base.pod_load.get(pid, 0) * len(ins) / hops * DETOUR_LOSS
            res[pre + (("do_reroute", pid, tuple(path), npath),)] = est / (pre_cost + (0 if pid in st.fresh else POD_COST - POD_REFUND) + DETOUR_VCOST)
        if DET:
            VCLOCK[0] += 7e-6 + 3e-6 * len(opts) + 2e-6 * len(res)
        return res

    def splits(self, st: State, pids: list, b: int, v: int, gain: float) -> dict:
        """Returns candidates moving a looping pod's back-and-forth over the saturated tube b-v into its own new shuttle (instead of
        upgrading the tube for an extra shuttle).
        :param st: current state
        :param pids: pods using tube b-v
        :param b: building with stuck astronauts
        :param v: closer building
        :param gain: estimated monthly gain of an extra shuttle over b-v
        :return: dict actions -> estimated monthly gain per resource spent"""
        res = {}
        for pid in pids:
            path = st.pods[pid]
            if len(path) > 3 and path[0] == path[-1]:
                for i in range(len(path) - 2):
                    if path[i] == path[i + 2] and path[i + 1] in (b, v) and path[i] in (b, v):
                        cost = POD_COST + (0 if pid in st.fresh else POD_COST - POD_REFUND)
                        for sh in ((b, v, b), (v, b, v)):
                            res[(("do_reroute", pid, tuple(path), (*path[:i + 1], *path[i + 3:])), ("do_pod", sh))] = gain / (cost + DETOUR_VCOST)
                        break
        if DET:
            VCLOCK[0] += 6e-6 + 3e-7 * sum(len(st.pods[pid]) for pid in pids)
        return res

    def by_dist(self, b: int) -> list[int]:
        """Returns the other buildings sorted by Euclidean distance from building b (cached until new buildings appear)."""
        r = self.order_cache.get(b)
        if r is None:
            ids, idx, dm = self.geometry()
            r = self.order_cache[b] = ids[np.argsort(dm[idx[b]], kind="stable")[1:]].tolist()
        return r

    def geometry(self) -> tuple:
        """Returns (building ids, id -> index, pairwise Euclidean distance matrix), cached until new buildings appear."""
        if self.geo is None:
            ids = list(self.st.pos)
            p = np.array([self.st.pos[b] for b in ids], dtype=float)
            self.geo = np.array(ids), {b: i for i, b in enumerate(ids)}, np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        return self.geo

    def targets(self, b: int, t: int) -> list[int]:
        """Returns the NT modules of type t nearest to building b (cached like by_dist)."""
        r = self.tgt_cache.get((b, t))
        if r is None:
            btype = self.st.btype
            r = self.tgt_cache[(b, t)] = [m for m in self.by_dist(b) if btype[m] == t][:NT]
        return r

    def mids(self, b: int, m: int) -> list[int]:
        """Returns up to NM intermediate buildings x with the smallest detour |bx| + |xm| (at most MLIM |bm|), cached like by_dist."""
        r = self.mid_cache.get((b, m))
        if r is None:
            ids, idx, dm = self.geometry()
            i, j = idx[b], idx[m]
            detour = dm[i] + dm[j]
            detour[[i, j]] = INF
            sel = np.flatnonzero(detour <= MLIM * dm[i, j])
            r = self.mid_cache[(b, m)] = ids[sel[np.argsort(detour[sel], kind="stable")[:NM]]].tolist()
        return r

    def route_candidates(self, st: State, base: SimResult, sig: dict, edge_pods: dict, cands: dict):
        """Adds routed candidates: for the largest stuck groups a cheap route (existing and legal new tubes) to a module of their type,
        for served groups losing speed points a route with fewer hops, and for flows into crowded modules a route (at most as long) to
        a less crowded module of the same type. Each route becomes one bundle: its missing tubes plus pods on its pod-less edges
        (phase-aligned shuttles, or one back-and-forth pod over the pod-less stretch). Jobs whose situation is unchanged are skipped.
        :param st: current state
        :param base: detailed simulation result of st
        :param sig: situation of each job at its last generation, updated in place
        :param edge_pods: tube -> number of pods using it
        :param cands: candidates (actions -> estimated monthly gain per resource), extended in place"""
        def changed(key: tuple, vals: tuple, objs: tuple) -> bool:
            """Checks whether a job's situation differs from its last generation and records the new one.
            :param key: job key
            :param vals: situation values compared by equality
            :param objs: situation objects compared by identity
            :return: whether the job has to be regenerated"""
            old = sig.get(key)
            if old is not None and old[0] == vals and all(a is b for a, b in zip(old[1], objs)):
                return False
            sig[key] = (vals, objs)
            return True

        def add_route(path: list[int], n: float, per: float):
            """Adds the bundles of a route: its missing tubes plus phase-aligned shuttles on its pod-less edges, and the same tubes plus one
            back-and-forth pod over the pod-less stretch; routes whose new tubes cross each other are skipped.
            :param path: route (building ids, teleporter hops allowed)
            :param n: astronauts served
            :param per: estimated monthly gain per astronaut served"""
            edges = [(x, y) for x, y in zip(path, path[1:]) if tele.get(x) != y]
            new = [(x, y) for x, y in edges if (min(x, y), max(x, y)) not in st.tubes]
            for i, (x, y) in enumerate(new):
                for u, w in new[i + 1:]:
                    if len({x, y, u, w}) == 4 and segments_intersect(st.pos[x], st.pos[y], st.pos[u], st.pos[w]):
                        return
            need = [i for i, (x, y) in enumerate(edges) if not edge_pods.get((min(x, y), max(x, y)))]
            if not need:
                return
            tubes = tuple(("do_tube", x, y) for x, y in new)
            tc = sum(st.tube_cost(x, y) for x, y in new)
            acts = tubes + tuple(("do_pod", (x, y, x) if i % 2 == 0 else (y, x, y)) for i in need for x, y in (edges[i],))
            if acts not in cands:
                cands[acts] = min(n, 100) * per / (tc + POD_COST * len(need))
            if len(need) >= 2 and all(edges[i][1] == edges[i + 1][0] for i in range(need[0], need[-1])):
                sub = [edges[need[0]][0]] + [edges[i][1] for i in range(need[0], need[-1] + 1)]
                if (acts := tubes + (("do_pod", tuple(sub + sub[-2::-1])),)) not in cands:
                    cands[acts] = min(n, 100 / (len(sub) - 1)) * per / (tc + POD_COST)

        tele, cnt = st.teleports, base.mod_cnt
        jobs = []  # (kind r/s/b, start, goal type or module set, max hops, astronauts, hop penalty, current mean arrival day / crowded module)
        for (b, t), n in sorted(base.stuck.items(), key=lambda kv: -kv[1])[:ROUTE_GROUPS]:
            if changed((b, t, "r"), (n,), (base.dist[t], st.adj.get(b))):
                jobs += [("r", b, t, INF, n, HOP_PENALTY, None)] + ([("r", b, t, INF, n, LOW_PENALTY, None)] if LOW_PENALTY else [])
                if RB and (alts := {x for x, bt in st.btype.items() if bt == t and cnt.get(x, 0) < ROUTE_BAL_MIN}) and \
                        any(bt == t and x not in alts for x, bt in st.btype.items()):
                    jobs.append(("r", b, alts, INF, n, HOP_PENALTY, None))
        if SPEED_GROUPS:
            slow = sorted(((sd - c, p, t) for (p, t), (c, sd) in base.group.items() if sd > c and 1 < base.dist[t][p] < INF), reverse=True)
            for _, p, t in slow[:SPEED_GROUPS]:
                c, sd = base.group[(p, t)]
                if changed((p, t, "s"), (c, sd), (base.dist[t],)):
                    jobs.append(("s", p, t, base.dist[t][p] - 1, c, HOP_PENALTY, sd / c))
        if BAL_GROUPS:
            crowd = sorted(((n, p, t, m) for (p, t, m), n in base.flow.items() if cnt[m] > ROUTE_BAL_MIN and n >= 5), reverse=True)
            for n, p, t, m in crowd[:BAL_GROUPS]:
                if changed((p, t, m, "b"), (n, cnt[m]), (base.dist[t],)):
                    if alts := {x for x, bt in st.btype.items() if bt == t and cnt.get(x, 0) < cnt[m] - 20}:
                        jobs.append(("b", p, alts, base.dist[t][p], n, HOP_PENALTY, m))
        if DET:
            VCLOCK[0] += 2e-5 + 1e-6 * (len(base.group) + len(base.flow)) + 1e-5 * len(jobs)
        newadj = self.pairset.legal(st) if jobs else None
        for kind, b, goal, max_hops, n, penalty, extra in jobs:
            if (path := self.route(st, b, goal, newadj, edge_pods, max_hops, penalty)) is None:
                continue
            hops = sum(tele.get(x) != y for x, y in zip(path, path[1:]))
            if kind == "r":
                k = min(n, 100)
                add_route(path, n, 50 - hops + balance_pts(cnt.get(path[-1], 0), k) / k if REST else max(10, 60 - 3 * hops))
            elif kind == "s":
                add_route(path, n, max(extra - hops, 0.5) * ROUTE_IMPROVE_K)
            else:
                c, c2 = cnt[extra], cnt.get(path[-1], 0)
                k = max(min(n, (c - c2) // 2), 1)
                add_route(path, k, max(balance_pts(c2, k) - balance_pts(c - k, k), 1) / k * ROUTE_IMPROVE_K)

    def route(self, st: State, b: int, goal: int | set, newadj: dict, edge_pods: dict, max_hops: int = INF, penalty: int = HOP_PENALTY) -> list[int] | None:
        """Finds a cheap route from building b to a goal building over existing tubes, teleporters and legal new tubes (Dijkstra with
        edge weight penalty per hop plus POD_COST on pod-less existing tubes, POD_COST plus the tube cost on new tubes).
        :param st: state
        :param b: start building
        :param goal: target module type (int) or set of target buildings
        :param newadj: legal new tubes as adjacency lists of (neighbour, tube cost)
        :param edge_pods: tube -> number of pods using it
        :param max_hops: maximum number of tube hops (paths reaching the limit are not extended)
        :param penalty: penalty per hop
        :return: list of buildings from b to the goal, or None"""
        btype, adj, tele = st.btype, st.adj, st.teleports
        is_set = isinstance(goal, set)
        best, prev, heap, pops = {b: 0}, {}, [(0, 0, b)], 0
        while heap:
            c, h, u = heappop(heap)
            if c > best[u]:
                continue
            pops += 1
            if (u in goal) if is_set else btype[u] == goal:
                if DET:
                    VCLOCK[0] += 4e-6 + 1.5e-6 * pops
                path = [u]
                while path[-1] != b:
                    path.append(prev[path[-1]])
                return path[::-1]
            if (x := tele.get(u)) is not None and c < best.get(x, INF):
                best[x], prev[x] = c, u
                heappush(heap, (c, h, x))
            if h >= max_hops:
                continue
            for v in adj[u]:
                w = c + penalty + (0 if edge_pods.get((u, v) if u < v else (v, u)) else POD_COST)
                if w < best.get(v, INF):
                    best[v], prev[v] = w, u
                    heappush(heap, (w, h + 1, v))
            for v, tc in newadj.get(u, ()):
                w = c + penalty + POD_COST + tc
                if w < best.get(v, INF):
                    best[v], prev[v] = w, u
                    heappush(heap, (w, h + 1, v))
        if DET:
            VCLOCK[0] += 4e-6 + 1.5e-6 * pops
        return None

    def served_candidates(self, st: State, base: SimResult, edge_pods: dict, cands: dict):
        """Adds candidates speeding up astronauts that arrive but wait at buildings on the way (queues for full pods) for the W_TOP
        (building, type) pairs with the most waiting astronaut-days: an extra shuttle in both phases on each shortest-path tube leaving
        the building (upgraded when already used by capacity pods), phase-aligned shuttles on every hop of its shortest route (lane) and,
        for non-pad buildings, a tube + shuttle to the nearest target modules. Estimates are deliberately low (W_K, LANE_K), so these
        candidates are mostly evaluated when nothing else reaches the acceptance threshold.
        :param st: current state
        :param base: detailed simulation result of st
        :param edge_pods: tube -> number of pods whose path uses it
        :param cands: candidate dict, extended in place (existing entries keep their estimates)"""
        top = sorted(base.wait.items(), key=lambda kv: -kv[1])[:W_TOP]
        if DET:
            VCLOCK[0] += 2e-6 * len(base.wait) + 2e-6 * W_TOP
        ew = defaultdict(int)
        for (b, t), w in top:
            d = base.dist[t]
            for v in st.adj[b]:
                if d[v] < d[b]:
                    ew[(b, v)] = max(ew[(b, v)], w)
        for (b, v), w in sorted(ew.items(), key=lambda kv: -kv[1])[:W_TOP]:
            key = (min(b, v), max(b, v))
            acts, cost = (), POD_COST
            if edge_pods[key] >= st.tubes[key]:
                acts, cost = (("do_upgrade", *key),), cost + st.tube_cost(b, v) * (st.tubes[key] + 1)
            for path in ((b, v, b), (v, b, v)):
                cands.setdefault(acts + (("do_pod", path),), w * W_K / cost)
        for (b, t), w in top:
            d = base.dist[t]
            if W_MIDS and b not in st.pad_astronauts and d[b] > 1:
                for m in self.targets(b, t)[:W_MIDS_N]:
                    if (min(b, m), max(b, m)) not in st.tubes:
                        cands.setdefault((("do_tube", b, m), ("do_pod", (b, m, b))), w * W_K / (st.tube_cost(b, m) + POD_COST))
            if not LANE or d[b] < 2:
                continue
            hops, x = [], b
            while d[x] > 0 and len(hops) < LANE_MAX:
                if (e := st.teleports.get(x)) is not None and d[e] <= d[x]:
                    x = e
                    continue
                v = max((v for v in st.adj[x] if d[v] == d[x] - 1), key=lambda v: (edge_pods[(min(x, v), max(x, v))], -v))
                hops.append((x, v))
                x = v
            if len(hops) < 2:
                continue
            ups = tuple(("do_upgrade", *key) for x, v in hops if edge_pods[key := (min(x, v), max(x, v))] >= st.tubes[key])
            cost = POD_COST * len(hops) + sum(st.tube_cost(a[1], a[2]) * (st.tubes[a[1:]] + 1) for a in ups)
            gain = sum(base.wait.get((x, t), 0) for x, _ in hops) * LANE_K
            for ph in (0, 1):
                cands.setdefault(ups + tuple(("do_pod", (x, v, x) if (i + ph) % 2 == 0 else (v, x, v)) for i, (x, v) in enumerate(hops)), gain / cost)

    def balance_candidates(self, st: State, base: SimResult, cands: dict, edge_pods: dict):
        """Adds candidates that divert arrivals from crowded modules to less crowded modules of the same type: for the largest arrival
        flows (any departure building b on the route -> crowded module m), a shuttle pod (with a new tube or an upgrade if needed) from b
        to a nearby less crowded module, and a re-creation of the pods feeding m from b with higher ids, so that rival pods fill first
        (teleporter diversions come from tele_candidates).
        :param st: current state
        :param base: detailed simulation result of st
        :param cands: actions -> estimated monthly gain per resource spent, extended in place
        :param edge_pods: number of pods using each tube"""
        def bal_gain(c: int, cx: int, mv: int) -> int:
            return balance_pts(0, c - mv) + balance_pts(0, cx + mv) - balance_pts(0, c) - balance_pts(0, cx)

        def add(acts: tuple, g: float):
            if g > cands.get(acts, 0):
                cands[acts] = g

        cnt, btype = base.mod_cnt, st.btype
        legs = defaultdict(set)  # building -> (next stop, pod id) over all pod legs
        if BAL_PRIO:
            for p, path in st.pods.items():
                for u, v in zip(path, path[1:]):
                    legs[u].add((v, p))
        flows = sorted(((f, b, m) for (b, m), f in base.pass_flow.items() if f >= BAL_F and cnt[m] >= BAL_C), reverse=True)
        done = set()
        for f, b, m in flows:
            if (b, m) in done or len(done) >= BAL_FLOWS:
                continue
            done.add((b, m))
            c, t = cnt[m], btype[m]
            d = base.dist[t][b]
            if DET:
                VCLOCK[0] += BAL_VF
            alts = [x for x in self.type_order(b, t) if cnt.get(x, 0) < c - BAL_D]
            k = 0
            for x in alts:
                if k >= BAL_K:
                    break
                key = (min(b, x), max(b, x))
                if key in st.tubes:
                    acts, cost = (), POD_COST
                    if edge_pods[key] >= st.tubes[key]:
                        acts, cost = (("do_upgrade", b, x),), cost + st.tube_cost(b, x) * (st.tubes[key] + 1)
                elif BAL_LEGAL and (key in self.bal_bad or not st.tube_legal(b, x)):
                    self.bal_bad.add(key)
                    continue
                else:
                    acts, cost = (("do_tube", b, x),), st.tube_cost(b, x) + POD_COST
                k += 1
                cx = cnt.get(x, 0)
                if (g := bal_gain(c, cx, min(f // 2, (c - cx) // 2)) if d <= 1 else bal_gain(c, cx, min(f, 100)) + min(f, 100) * (d - 1)) > 0:
                    for path in ((b, x, b), (x, b, x)):
                        add(acts + (("do_pod", path),), g / cost)
            if BAL_PRIO and (b, m) in base.hop_flow and max(st.pods, default=0) < 500 - 8:
                feeders = {p for v, p in legs[b] if v == m}
                rivals = {p for v, p in legs[b] if cnt.get(v, 0) < c - BAL_D and btype[v] == t}
                if feeders and rivals and min(feeders) < max(rivals) and len(feeders) <= 4:
                    top = max(st.pods)
                    acts = tuple(a for k, p in enumerate(sorted(feeders)) for a in (("do_destroy", p), ("do_pod", tuple(st.pods[p]), top + 1 + k)))
                    add(acts, bal_gain(c, 0, min(f // 2, c // 2)) / 2 / (250 * len(feeders)))

    def type_order(self, b: int, t: int) -> list[int]:
        """Returns all modules of type t sorted by Euclidean distance from building b (cached like by_dist)."""
        r = self.tord_cache.get((b, t))
        if r is None:
            btype = self.st.btype
            r = self.tord_cache[(b, t)] = [m for m in self.by_dist(b) if btype[m] == t]
        return r

    def shortenings(self, st: State, base: SimResult) -> dict:
        """Returns re-routing candidates removing round trips x y x from looping pods on which (almost) nobody boards.
        :param st: current state
        :param base: detailed simulation result of st
        :return: dict actions -> estimated monthly gain per resource spent"""
        res = {}
        for pid, path in st.pods.items():
            load, legs = base.pod_load.get(pid, 0), base.leg_load.get(pid)
            if len(path) > 3 and path[0] == path[-1] and load:
                for i in range(len(path) - 2):
                    if path[i] == path[i + 2] and legs[i] + legs[i + 1] <= SHORTEN_MAX:
                        res[(("do_reroute", pid, tuple(path), (*path[:i + 1], *path[i + 3:])),)] = \
                            (load * 2 / (len(path) - 3) * DETOUR_LOSS - (legs[i] + legs[i + 1]) * 60) / ((0 if pid in st.fresh else POD_COST - POD_REFUND) + 1)
        if DET:
            VCLOCK[0] += 7e-6 + 2e-7 * sum(map(len, st.pods.values()))
        return res

    def tele_candidates(self, st: State, base: SimResult, cands: dict, edge_pods: dict):
        """Adds teleporter candidates for every (building, type) with astronauts standing there during the month: an entrance at the
        building itself or at a tube neighbour fed by shuttle pods (upgrading the tube when needed), with an exit at a free module of
        the type. The estimate assumes that these astronauts take it the day after reaching the building (a day later via a hub, whose
        pods divert at most their capacity) and counts the balancing change at the modules.
        :param st: current state
        :param base: detailed simulation result of st
        :param cands: candidates (actions -> estimated monthly gain per resource), extended in place
        :param edge_pods: tube -> number of pods using it"""
        if self.tele_geo is None or self.tele_geo[0] != len(st.pos):
            _, idx, dm = self.geometry()
            pads = [idx[p] for p in st.pad_astronauts]
            mods = defaultdict(list)
            for m, t in st.btype.items():
                if t:
                    mods[t].append(m)
            self.tele_geo = len(st.pos), {m: dm[idx[m], pads].min() for ms in mods.values() for m in ms}, mods
        _, far, mods = self.tele_geo
        cnt, used, adj, btype = base.mod_cnt, st.tele_used, st.adj, st.btype
        pod_dir = defaultdict(int)
        for path in st.pods.values():
            for e in set(zip(path, path[1:])):
                pod_dir[e] += 1
        if DET:
            VCLOCK[0] += 8e-6 * len(base.pres) + 1e-6 * sum(len(mods[t]) for _, t in base.pres)
        for (b, t), (n, g, mv) in base.pres.items():
            want = min(HUB_PODS, (n + 9) // 10)
            sh = min(n, POD_CAPACITY * want)
            for s, pen, ents in ((n, 0, [] if b in used else [b]), (sh, sh, [h for h in adj[b] if h not in used and btype[h] != t])):
                if not ents:
                    continue
                kk = {m: round(k * s / n) for m, k in mv.items()}  # a hub with limited pod throughput diverts only a share s
                g0 = g * s / n - pen - sum(balance_pts(cnt[m] - k, k) for m, k in kk.items())
                opts = sorted((-g0 - balance_pts(cnt[m] - kk.get(m, 0), s), len(adj[m]) > 0, -far[m], m) for m in mods[t] if m not in used)
                opts = [o for i, o in enumerate(opts) if o[0] < 0 and (i == 0 or o[0] != opts[i - 1][0])][:TELE_PER_GROUP]
                for ng, _, _, m in opts:
                    for h in ents:
                        acts, cost = (), TELEPORT_COST
                        if h != b:
                            key = (min(b, h), max(b, h))
                            cap, e = st.tubes[key], edge_pods[key]
                            for j in range(e, e + max(0, want - pod_dir[(b, h)])):
                                if j >= cap:
                                    cap += 1
                                    acts, cost = acts + (("do_upgrade", b, h),), cost + st.tube_cost(b, h) * cap
                                acts, cost = acts + (("do_pod", (b, h, b)),), cost + POD_COST
                        acts += (("do_teleport", h, m),)
                        if (est := -ng * TELE_K / cost) > cands.get(acts, 0):
                            cands[acts] = est

    def util(self, r: float) -> float:
        """Returns the utility (in points over the rest of the game) of holding resources: TH_HI per resource inside the reserve for future
        buildings plus, above it, the integral of the marginal value of money min(TH_MAX, TH_K / r), scaled down linearly over the last
        months to 0 in the last month (unused resources are worthless then).
        A bundle is bought when its gain over the remaining months covers the utility of the resources it spends.
        :param r: resources
        :return: utility"""
        r0, r1 = TH_K / TH_MAX, max(0, r - self.res)
        return TH_HI * min(r, self.res) + min(1, (MONTHS - self.month - 1) / TH_M) * (TH_MAX * r1 if r1 <= r0 else TH_K * (1 + math.log(r1 / r0)))

    def reserve(self) -> float:
        """Returns the resources to keep for buildings expected in future months: the present value of the mean monthly deficit
        (new buildings per month since month 1 times CB minus mean inferred income) over the remaining months (0 in month 0).
        :return: reserve"""
        if self.month == 0:
            return 0
        need = CB * sum(self.hist_new[1:]) / self.month - sum(self.hist_inc[1:]) / self.month
        return max(0, need) * (1 - 1.1 ** (self.month + 1 - MONTHS)) * 10


def log(msg: str):
    """Writes one diagnostic line to stderr."""
    print(f"[S] {msg}", file=sys.stderr, flush=True)


def log_blob(tag: str, text: str):
    """Writes text zlib-compressed and base64-encoded to stderr, split into lines of at most LOG_CHUNK characters.
    :param tag: line tag, e.g. M03 for the buildings of month 3
    :param text: text to encode"""
    b = base64.b64encode(zlib.compress(text.encode(), 9)).decode()
    parts = [b[i:i + LOG_CHUNK] for i in range(0, len(b), LOG_CHUNK)] or [""]
    for i, part in enumerate(parts):
        log(f"{tag} {i + 1}/{len(parts)} {part}")


def main():
    bot = Bot()
    rd = sys.stdin.readline
    while True:
        w = time.perf_counter()
        first = rd()
        if not first:
            break
        if not bot.month and CLOCK is time.perf_counter and time.perf_counter() - w < 0.01:
            bot.t0 = T_START  # the first input was already waiting, so its time limit may have started during the start-up
        lines = [first]
        n = int(rd())
        lines.append(str(n))
        lines += [rd() for _ in range(n)]
        n = int(rd())
        lines.append(str(n))
        lines += [rd() for _ in range(n)]
        n = int(rd())
        lines.append(str(n))
        lines += [rd() for _ in range(n)]
        print(bot.play_turn(lines), flush=True)


if __name__ == "__main__":
    main()
