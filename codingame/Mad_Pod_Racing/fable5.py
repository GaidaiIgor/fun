import sys
import math
import time


def rd():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def ang_of(dx, dy):
    return math.degrees(math.atan2(dy, dx)) % 360.0


def sdiff(a, b):
    # signed smallest difference a-b in [-180, 180]
    return (a - b + 180.0) % 360.0 - 180.0


def seg_pt_d2(px, py, qx, qy, cx, cy):
    # squared min distance from segment p->q to point c
    dx = qx - px
    dy = qy - py
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        ex = px - cx
        ey = py - cy
        return ex * ex + ey * ey
    t = ((cx - px) * dx + (cy - py) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = px + t * dx - cx
    ey = py + t * dy - cy
    return ex * ex + ey * ey


# ---------------- init ----------------
laps = int(rd())
N = int(rd())
cps = [tuple(map(int, rd().split())) for _ in range(N)]

seg_len = [math.hypot(cps[i][0] - cps[i - 1][0], cps[i][1] - cps[i - 1][1])
           for i in range(N)]
longest_to = max(range(N), key=lambda i: seg_len[i])

CP_HIT2 = 585.0 * 585.0      # slight safety margin vs the real 600
CONTACT2 = 815.0 * 815.0     # pod contact distance 800 + small margin

boost_used = False
shield_cd = [0, 0]           # my pods: turns of engine-blocked acceleration left
passed = [0, 0, 0, 0]
prev_ncp = [1, 1, 1, 1]
prev_racer = -1
prev_target = -1
turn = 0
pods = []
my_timeout = 100
prog_hist = []               # (turn, racer prog) for pin detection
pinner = -1                  # enemy pod currently pinning my racer, if any


def min_approach(px, py, rvx, rvy):
    # closest squared distance of p + t*rv for t in [0,1]
    rv2 = rvx * rvx + rvy * rvy
    if rv2 < 1e-9:
        return px * px + py * py
    t = -(px * rvx + py * rvy) / rv2
    t = clamp(t, 0.0, 1.0)
    ex = px + t * rvx
    ey = py + t * rvy
    return ex * ex + ey * ey


def facing_after(i, tx, ty):
    x, y, vx, vy, ang, nc = pods[i]
    des = ang_of(tx - x, ty - y)
    if turn == 1 or ang < 0:
        return des
    return (ang + clamp(sdiff(des, ang), -18.0, 18.0)) % 360.0


def planned_vel(i, tx, ty, act):
    x, y, vx, vy, ang, nc = pods[i]
    th = 0.0
    if shield_cd[i] == 0 and act != 'SHIELD':
        if act == 'BOOST':
            th = 200.0 if boost_used else 650.0
        else:
            th = float(act)
    ar = math.radians(facing_after(i, tx, ty))
    return vx + th * math.cos(ar), vy + th * math.sin(ar)


def nearest_enemy_dist(i):
    x, y = pods[i][0], pods[i][1]
    return min(math.hypot(pods[j][0] - x, pods[j][1] - y) for j in (2, 3))


def prog(i):
    x, y = pods[i][0], pods[i][1]
    nc = pods[i][5]
    return passed[i] * 50000.0 - math.hypot(cps[nc][0] - x, cps[nc][1] - y)


# ---------------- racer: short-horizon rollout search ----------------
H = 8
TWO_PLY = True
OFFS = (-18.0, -13.0, -8.0, -4.0, 0.0, 4.0, 8.0, 13.0, 18.0)
THS = (0.0, 90.0, 200.0)
RAD = math.radians
COS = math.cos
SIN = math.sin


def other_tracks(i):
    # predicted positions of the other three pods for H turns: racing enemies
    # rotate + thrust 200 toward their checkpoint, the teammate coasts.
    # Enemies charging AT me are returned separately as pursuers and get
    # re-simulated inside each rollout (they react to my candidate path).
    x, y = float(pods[i][0]), float(pods[i][1])
    tracks = []
    pursuers = []
    for j in range(4):
        if j == i:
            continue
        px_, py_ = float(pods[j][0]), float(pods[j][1])
        tvx, tvy = float(pods[j][2]), float(pods[j][3])
        is_enemy = j >= 2
        spd2 = tvx * tvx + tvy * tvy
        if is_enemy and spd2 >= 22500.0:
            dx, dy = x - px_, y - py_
            dist = math.hypot(dx, dy)
            spd = math.sqrt(spd2)
            closing = (tvx * dx + tvy * dy) / (dist * spd) if dist > 1.0 else 0.0
            ea = float(pods[j][4])
            if ea < 0:
                ea = ang_of(cps[pods[j][5]][0] - px_,
                            cps[pods[j][5]][1] - py_)
            if dist < 4500.0 and closing > 0.6:
                # heading at me: treat as a pursuer
                pursuers.append((px_, py_, tvx, tvy, ea))
                continue
            enc = pods[j][5]
            tr = []
            for _ in range(H):
                ccx, ccy = cps[enc]
                ea = (ea + clamp(sdiff(ang_of(ccx - px_, ccy - py_), ea),
                                 -18.0, 18.0)) % 360.0
                ar = RAD(ea)
                tvx += 200.0 * COS(ar)
                tvy += 200.0 * SIN(ar)
                nx2, ny2 = px_ + tvx, py_ + tvy
                if seg_pt_d2(px_, py_, nx2, ny2, ccx, ccy) < 360000.0:
                    enc = (enc + 1) % N
                px_, py_ = nx2, ny2
                tvx *= 0.85
                tvy *= 0.85
                tr.append((px_, py_))
            tracks.append((tr, True))
        else:
            tr = []
            for _ in range(H):
                px_ += tvx
                py_ += tvy
                tvx *= 0.85
                tvy *= 0.85
                tr.append((px_, py_))
            tracks.append((tr, is_enemy))
    return tracks, pursuers


def rollout(x, y, vx, vy, ang, nc, acts, tracks, pursuers, s0, pmul=1.0):
    # continue from step s0 applying acts, then the chase policy; also
    # returns the state right after the first applied action. Pursuers are
    # simulated reactively: they rotate + thrust 200 toward my position.
    score = 0.0
    cur = nc
    cx, cy = cps[cur]
    after_first = None
    na = 0.0
    ps = list(pursuers)
    for s in range(s0, H):
        k = s - s0
        if k < len(acts):
            na, th = acts[k]
        else:
            # follow-up policy: chase the compensated checkpoint at full power
            des = ang_of(cx - 3.0 * vx - x, cy - 3.0 * vy - y)
            d = sdiff(des, ang)
            na = (ang + clamp(d, -18.0, 18.0)) % 360.0
            th = 200.0 if abs(d) < 90.0 else 0.0
        ar = RAD(na)
        vx += th * COS(ar)
        vy += th * SIN(ar)
        nx = x + vx
        ny = y + vy
        if seg_pt_d2(x, y, nx, ny, cx, cy) < CP_HIT2:
            score += 50000.0 + (H - s) * 4500.0
            cur = (cur + 1) % N
            cx, cy = cps[cur]
        for tr, is_enemy in tracks:
            ex, ey = tr[s]
            ddx = nx - ex
            ddy = ny - ey
            if ddx * ddx + ddy * ddy < 640000.0:  # 800^2
                if is_enemy:
                    score -= (12000.0 if s <= 1 else 6000.0) * pmul
                else:
                    score -= 2500.0 if s <= 1 else 1200.0
        for pi in range(len(ps)):
            px_, py_, pvx, pvy, pa = ps[pi]
            pa = (pa + clamp(sdiff(ang_of(nx - px_, ny - py_), pa),
                             -18.0, 18.0)) % 360.0
            par = RAD(pa)
            pvx += 200.0 * COS(par)
            pvy += 200.0 * SIN(par)
            px_ += pvx
            py_ += pvy
            pvx *= 0.85
            pvy *= 0.85
            ps[pi] = (px_, py_, pvx, pvy, pa)
            ddx = nx - px_
            ddy = ny - py_
            if ddx * ddx + ddy * ddy < 640000.0:
                score -= (12000.0 if s <= 1 else 6000.0) * pmul
        x, y = nx, ny
        ang = na
        vx *= 0.85
        vy *= 0.85
        if k == 0:
            after_first = (x, y, vx, vy, ang, cur, score, tuple(ps))
    # final heading term keeps a useful gradient when the crossing lies
    # beyond the horizon (e.g. after being spun around in a melee)
    score -= abs(sdiff(ang_of(cx - x, cy - y), ang)) * 12.0
    return score - math.hypot(cx - x, cy - y), after_first


def racer_cmd(i):
    x, y, vx, vy, ang, nc = pods[i]
    cx, cy = cps[nc]
    dcp = math.hypot(cx - x, cy - y)
    if turn == 1 or ang < 0:
        # opening: free rotation, perfect alignment - boost for the early
        # lead (the first melee decides who blocks whom all game)
        return (cx, cy, 'BOOST' if dcp > 3000.0 else 200, 'RUN')

    tracks, pursuers = other_tracks(i)
    fx, fy = float(x), float(y)
    fvx, fvy = float(vx), float(vy)
    fang = float(ang)
    # breakout: when pinned, value clearance over grinding into the wall -
    # back off, build speed, re-attack the checkpoint rim from a new angle
    pmul = 2.5 if pinner >= 0 else 1.0
    deadline = t0 + 0.025
    best = None
    for off in OFFS:
        na0 = (fang + off) % 360.0
        for th0 in THS:
            sc1, mid = rollout(fx, fy, fvx, fvy, fang, nc,
                               ((na0, th0),), tracks, pursuers, 0, pmul)
            if best is None or sc1 > best[0]:
                best = (sc1, na0, th0, off)
            # second ply from the post-action state
            mx, my, mvx2, mvy2, mang, mnc, base, mps = mid
            if not TWO_PLY or time.perf_counter() > deadline:
                continue
            for off2 in OFFS:
                na1 = (mang + off2) % 360.0
                if time.perf_counter() > deadline:
                    break
                for th1 in THS:
                    sc2, _ = rollout(mx, my, mvx2, mvy2, mang, mnc,
                                     ((na1, th1),), tracks, mps, 1, pmul)
                    sc2 += base
                    if sc2 > best[0]:
                        best = (sc2, na0, th0, off)
    _, na0, th0, off = best
    tx = int(x + 8000.0 * COS(RAD(na0)))
    ty = int(y + 8000.0 * SIN(RAD(na0)))
    act = int(th0)
    ad = abs(sdiff(ang_of(cx - x, cy - y), fang))
    # after the real boost is spent, BOOST legally falls back to max thrust,
    # so keep asking on good straights in case the arena grants more boosts
    if (shield_cd[i] == 0 and turn > 2
            and off == 0.0 and th0 == 200.0 and ad < 5.0
            and nearest_enemy_dist(i) > 1200.0
            and ((nc == longest_to and dcp > 3000.0) or dcp > 4200.0
                 or (passed[i] >= N and dcp > 3000.0))):
        act = 'BOOST'
    return (tx, ty, act, 'SRC')


def blocker_cmd(i, j):
    x, y, vx, vy, ang, nc = pods[i]
    ex, ey, evx, evy, eang, enc = pods[j]
    free = (turn == 1 or ang < 0)

    # timer rescue: if the team clock runs low, grab my own next checkpoint
    cx, cy = cps[nc]
    dmy = math.hypot(cx - x, cy - y)
    if my_timeout <= max(32.0, dmy / 700.0 + 10.0):
        axp = cx - 3.0 * vx
        ayp = cy - 3.0 * vy
        d = 0.0 if free else sdiff(ang_of(axp - x, ayp - y), ang)
        ad = abs(d)
        thrust = 0 if ad >= 90.0 else int(200.0 * math.cos(math.radians(ad)))
        return (int(axp), int(ayp), thrust, 'TMR')

    # yield: never squat on my own racer's next checkpoint while it comes in
    ri = 1 - i
    rnc = pods[ri][5]
    rcx, rcy = cps[rnc]
    d_cp = math.hypot(rcx - x, rcy - y)
    racer_d = math.hypot(rcx - pods[ri][0], rcy - pods[ri][1])
    if d_cp < 1400.0 and racer_d < 3500.0:
        ux, uy = x - rcx, y - rcy
        L = math.hypot(ux, uy) or 1.0
        txp = rcx + ux / L * 2400.0
        typ = rcy + uy / L * 2400.0
        d = 0.0 if free else sdiff(ang_of(txp - x, typ - y), ang)
        ad = abs(d)
        thrust = 0 if ad >= 90.0 else int(160.0 * math.cos(math.radians(ad)))
        return (int(txp), int(typ), thrust, 'YLD')

    # simulate the enemy leader racing ahead
    H = 30
    sx, sy = float(ex), float(ey)
    svx, svy = float(evx), float(evy)
    sa = float(eang) if eang >= 0 else ang_of(cps[enc][0] - ex, cps[enc][1] - ey)
    snc = enc
    epath = []
    for _ in range(H):
        ccx, ccy = cps[snc]
        sa = (sa + clamp(sdiff(ang_of(ccx - sx, ccy - sy), sa), -18.0, 18.0)) % 360.0
        ar = math.radians(sa)
        svx += 200.0 * math.cos(ar)
        svy += 200.0 * math.sin(ar)
        nx2, ny2 = sx + svx, sy + svy
        if seg_pt_d2(sx, sy, nx2, ny2, ccx, ccy) < 360000.0:
            snc = (snc + 1) % N
        sx, sy = nx2, ny2
        svx *= 0.85
        svy *= 0.85
        epath.append((sx, sy))

    # can I intercept the enemy along that path?
    s = math.hypot(vx, vy)
    dc = 0.0
    inter = None
    for k in range(H):
        s += 200.0
        dc += s
        s *= 0.85
        pxk, pyk = epath[k]
        if dc >= math.hypot(pxk - x, pyk - y) - 650.0:
            inter = (k, pxk, pyk)
            break
    if inter is not None and inter[0] <= 10:
        _, tx, ty = inter
        axp = tx - 2.0 * vx
        ayp = ty - 2.0 * vy
        d = 0.0 if free else sdiff(ang_of(axp - x, ayp - y), ang)
        ad = abs(d)
        thrust = 0 if ad >= 90.0 else int(200.0 * math.cos(math.radians(ad)))
        return (int(axp), int(ayp), thrust, 'INT')

    # otherwise camp at the checkpoint after the enemy's next one
    cA = cps[enc]
    nn = (enc + 1) % N
    cB = cps[nn]
    ddx, ddy = cA[0] - cB[0], cA[1] - cB[1]
    L = math.hypot(ddx, ddy) or 1.0
    campx = cB[0] + ddx / L * 500.0
    campy = cB[1] + ddy / L * 500.0
    dcamp = math.hypot(campx - x, campy - y)
    denemy = math.hypot(ex - x, ey - y)
    if dcamp > 1000.0:
        axp = campx - 3.0 * vx
        ayp = campy - 3.0 * vy
        d = 0.0 if free else sdiff(ang_of(axp - x, ayp - y), ang)
        ad = abs(d)
        thrust = 0 if ad >= 90.0 else int(200.0 * math.cos(math.radians(ad)))
        if dcamp < 2200.0 and math.hypot(vx, vy) > 420.0:
            thrust = min(thrust, 60)
        return (int(axp), int(ayp), thrust, 'CMP')
    if denemy < 3200.0:
        tx = ex + 3 * evx
        ty = ey + 3 * evy
        d = 0.0 if free else sdiff(ang_of(tx - x, ty - y), ang)
        thrust = 200 if abs(d) < 90.0 else 0
        return (int(tx), int(ty), thrust, 'CHG')
    if math.hypot(vx, vy) > 120.0:
        return (int(ex), int(ey), 0, 'HLD')
    return (int(campx), int(campy), 40, 'HLD')


def enemy_contact(i, mvx, mvy):
    # (best_rel_speed, rel_vx, rel_vy, enemy_idx) for any enemy contacting me
    # this turn, using the enemy's ACTUAL velocity (no speculative accel)
    x, y = pods[i][0], pods[i][1]
    best = None
    for j in (2, 3):
        ex, ey, evx, evy = pods[j][0], pods[j][1], pods[j][2], pods[j][3]
        rvx, rvy = evx - mvx, evy - mvy
        if min_approach(ex - x, ey - y, rvx, rvy) < CONTACT2:
            rel = math.hypot(rvx, rvy)
            if best is None or rel > best[0]:
                best = (rel, rvx, rvy, j)
    return best


# ---------------- main loop ----------------
while True:
    first_line = rd()
    pods = [list(map(int, first_line.split()))]
    for _ in range(3):
        pods.append(list(map(int, rd().split())))
    t0 = time.perf_counter()
    turn += 1
    shield_cd = [max(0, c - 1) for c in shield_cd]

    mine_passed = False
    for i in range(4):
        if pods[i][5] != prev_ncp[i]:
            passed[i] += 1
            prev_ncp[i] = pods[i][5]
            if i < 2:
                mine_passed = True
    my_timeout = 100 if mine_passed else my_timeout - 1

    s0, s1 = prog(0), prog(1)
    if prev_racer == 0:
        s0 += 3000.0
    elif prev_racer == 1:
        s1 += 3000.0
    racer = 0 if s0 >= s1 else 1
    blocker = 1 - racer

    # pin detection: a racer that stops gaining progress while an enemy sits
    # on it needs its blocker to come knock the pinner away (PROTECT mode)
    if racer != prev_racer:
        prog_hist = []
    prog_hist.append((turn, prog(racer)))
    while prog_hist and prog_hist[0][0] < turn - 20:
        prog_hist.pop(0)
    pinner = -1
    if (turn > 25 and len(prog_hist) > 15
            and prog_hist[-1][1] - prog_hist[0][1] < 600.0):
        rx, ry = pods[racer][0], pods[racer][1]
        dists = [(math.hypot(pods[j][0] - rx, pods[j][1] - ry), j)
                 for j in (2, 3)]
        dmin, jmin = min(dists)
        if dmin < 1600.0:
            pinner = jmin
    prev_racer = racer
    p2, p3 = prog(2), prog(3)
    if prev_target == 2:
        p2 += 25000.0
    elif prev_target == 3:
        p3 += 25000.0
    e_leader = 2 if p2 >= p3 else 3
    prev_target = e_leader

    cmds = [None, None]
    cmds[racer] = racer_cmd(racer)
    cmds[blocker] = blocker_cmd(blocker, e_leader)

    # SHIELD decisions
    for idx, role in ((racer, 'R'), (blocker, 'B')):
        tx, ty, act, tag = cmds[idx]
        if shield_cd[idx] > 0 or act == 'SHIELD':
            continue
        mvx, mvy = planned_vel(idx, tx, ty, act)
        hit = enemy_contact(idx, mvx, mvy)
        if hit is None:
            continue
        rel, rvx, rvy, ej = hit
        if role == 'B':
            # shield only for hard hits on the pod we are actually hunting;
            # reload turns are too expensive to spend on the enemy blocker
            if rel > 300.0 and ej == e_leader:
                cmds[idx] = (tx, ty, 'SHIELD', tag)
        else:
            # racer: shields cost 4 acceleration turns - only for truly hard,
            # adverse hits, and never right at my checkpoint (momentum through
            # the crossing is worth more than the bounce)
            nc = pods[idx][5]
            dx = cps[nc][0] - pods[idx][0]
            dy = cps[nc][1] - pods[idx][1]
            dcp_ = math.hypot(dx, dy)
            L = dcp_ or 1.0
            along = (rvx * dx + rvy * dy) / L
            if rel > 520.0 and along < 0.25 * rel and dcp_ > 1000.0:
                cmds[idx] = (tx, ty, 'SHIELD', tag)

    # avoid ramming my own racer
    rtx, rty, ract, _ = cmds[racer]
    btx, bty, bact, btag = cmds[blocker]
    if bact != 'SHIELD':
        rvx_, rvy_ = planned_vel(racer, rtx, rty, ract)
        bvx_, bvy_ = planned_vel(blocker, btx, bty, bact)
        px = pods[racer][0] - pods[blocker][0]
        py = pods[racer][1] - pods[blocker][1]
        if min_approach(px, py, rvx_ - bvx_, rvy_ - bvy_) < CONTACT2:
            cmds[blocker] = (btx, bty, 0, 'AVD')

    out = []
    for i in (0, 1):
        tx, ty, act, tag = cmds[i]
        if act == 'BOOST':
            boost_used = True
        if act == 'SHIELD':
            shield_cd[i] = 4
        out.append("%d %d %s" % (int(tx), int(ty), act))

    sys.stdout.write(out[0] + "\n" + out[1] + "\n")
    sys.stdout.flush()

    ms = (time.perf_counter() - t0) * 1000.0
    sys.stderr.write("T%d r%d %s|%s %.1fms\n"
                     % (turn, racer, cmds[0][3], cmds[1][3], ms))
    sys.stderr.flush()
