"""Lands Mars Lander on the flat ground zone spending as little fuel as possible. A closed-loop guidance policy with 16 real parameters is
simulated from the exactly tracked lander state until touchdown, and its parameters are re-optimized every turn by random search seeded with
the previous turn's elite and a set of offline-tuned parameter vectors."""
import gc
from bisect import bisect_left, bisect_right, insort
from itertools import pairwise
from math import atan2, cos, degrees, radians, sin, sqrt
from random import Random
from time import perf_counter

G, FIRST_BUDGET, BUDGET = 3.711, 0.01, 0.04
SIN, COS = [sin(radians(a)) for a in range(-90, 91)], [cos(radians(a)) for a in range(-90, 91)]
# Search bounds of the policy parameters: vx_max (cruise horizontal speed cap), acc_x (deceleration assumed by the braking-distance horizontal
# speed law), acc_y (deceleration of the descent speed profile), h_slow (altitude above the zone where the profile reaches v_land), tilt (tilt
# cap), kx, ky (horizontal and vertical speed error gains), dead (horizontal acceleration dead band while coasting), v_land (touchdown descent
# speed), margin (target inset from the zone edges), vx_land (touchdown horizontal speed cap), ka (gain on the constant horizontal acceleration
# that reaches the target in time), look (terrain look-ahead time), clear (terrain clearance), vy_cruise (descent speed cap while cruising),
# acc_max (largest constant horizontal acceleration that is still followed).
LO = [0, 0.2, 0.02, -30, 5, 0.05, 0.05, 0, 20, 20, 3, 0.5, 0, 0, 0, 0.05]
HI = [200, 4, 0.35, 400, 90, 3, 3, 4, 39, 400, 19, 4, 20, 400, 60, 4]
SEEDS = [[63.71, 0.2, 0.35, -30, 51.7, 0.1454, 0.118, 2.557, 30.56, 153.9, 3.349, 2.033, 17.71, 308.9, 0, 3.027],
         [133.5, 1.745, 0.0829, 295.3, 47.83, 2.991, 1.82, 1.725, 32.83, 130, 6.665, 2.084, 16.8, 266.4, 13.79, 1.081],
         [200, 0.4593, 0.2639, -0.5674, 29.21, 1.426, 0.4609, 1.581, 36.22, 27.72, 12.6, 2.387, 18.59, 10.72, 0, 4],
         [85.54, 1.132, 0.2909, 400, 49.15, 0.5834, 1.389, 0.3601, 37.65, 87.95, 12.59, 0.7954, 19.1, 400, 40.77, 0.2805],
         [46.83, 0.3149, 0.1539, 29.2, 50.3, 0.3241, 0.2526, 1.017, 37.14, 125.4, 15.71, 2.764, 12.89, 320.9, 10.39, 1.277],
         [62.27, 0.6893, 0.1353, 53.33, 74.04, 0.7808, 2.987, 3.933, 39, 69.14, 13.41, 3.29, 17.27, 57.66, 2.327, 2.106],
         [171, 1.114, 0.1412, 389.1, 56.55, 0.1239, 2.109, 1.835, 35.81, 23.14, 9.622, 2.374, 16.27, 108.9, 2.162, 3.376],
         [121.9, 1.054, 0.2283, 136.4, 25.15, 0.05, 0.1211, 1.142, 34.49, 82.14, 6.952, 1.707, 19.78, 3.647, 60, 2.889],
         [122.6, 0.8827, 0.136, 108.6, 65.93, 1.732, 1.012, 0.7345, 38.45, 89.54, 18.15, 2.674, 16.44, 105.9, 17.35, 2.552],
         [121.9, 2.062, 0.3137, 102.5, 43.09, 1.321, 0.9295, 3.99, 37.09, 226.4, 19, 1.465, 19.96, 211.3, 12.05, 0.8099],
         [107.4, 1.082, 0.1868, 138.2, 35.46, 1.218, 0.4102, 0.1999, 34.71, 146.9, 13.67, 1.234, 18.79, 260.8, 54.49, 2.652],
         [75.81, 1.665, 0.1613, 239.6, 68.11, 1.596, 2.986, 0.7486, 33.29, 239.6, 10.24, 3.492, 4.723, 0, 0, 3.292],
         [53.91, 3.942, 0.198, 13.49, 58.76, 0.05, 0.1338, 2.186, 39, 20, 16.8, 3.965, 17.21, 326.2, 55.88, 4],
         [36.97, 2.49, 0.35, 134.8, 67, 0.1606, 0.1292, 3.583, 35.66, 20, 9.865, 2.282, 13.68, 400, 60, 3.99],
         [77.81, 1.348, 0.1331, 53.78, 67.67, 0.6607, 1.898, 2.929, 38.73, 107.5, 18.72, 4, 17.93, 6.278, 0, 3.743],
         [92.41, 1.518, 0.02, 45.99, 48.18, 2.089, 2.185, 2.433, 38.18, 364.2, 16.96, 1.123, 6.043, 241.2, 33.63, 1.458],
         [192.2, 1.103, 0.08996, 278.9, 49.01, 1.033, 0.2755, 2.975, 27.16, 287, 5.285, 2.171, 17.51, 287.9, 38.01, 1.725],
         [36.27, 0.9268, 0.1226, 127.3, 50.97, 2.775, 2.02, 1.362, 38.32, 125.5, 12.24, 2.59, 17.99, 105.6, 37.38, 4],
         [86.79, 1.275, 0.2486, 186.2, 72.27, 0.7272, 3, 0, 35.14, 207.8, 10.3, 1.578, 5.119, 0, 0.7349, 4],
         [117.9, 1.994, 0.2576, 8.78, 44.51, 3, 0.8494, 0.9409, 34.48, 242.5, 15.75, 2.145, 20, 299.6, 34.97, 3.219],
         [73.73, 1.014, 0.03826, 300.6, 73.46, 2.311, 2.114, 0.4596, 37.19, 20, 17.46, 3.128, 8.428, 50.77, 0, 2.1],
         [8.808, 3.941, 0.1302, 19.38, 37.63, 0.3658, 0.1758, 1.964, 38.91, 70.55, 17.4, 1.685, 11.49, 226.3, 30.96, 3.913],
         [46.7, 3.793, 0.3194, 309.3, 68.09, 0.1113, 0.4996, 2.635, 32.79, 356.6, 8.478, 2.705, 11.78, 30.26, 47.76, 2.957],
         [6.549, 0.7106, 0.247, 60.69, 31.93, 1.139, 1.553, 0.1128, 29.59, 228.5, 14.35, 0.8804, 17.06, 241.5, 8.504, 2.568],
         [76.92, 0.5367, 0.2251, 127.2, 32.17, 1.89, 2.668, 1.57, 24.28, 67.54, 16.25, 3, 4.579, 96.81, 27.74, 3.023],
         [121.9, 0.2, 0.07439, 26.98, 56.07, 0.767, 0.7209, 2.931, 33.7, 282.9, 17.18, 1.078, 14.08, 391.4, 0, 3.523],
         [70.14, 0.6347, 0.241, 161.2, 56.84, 0.2191, 2.287, 3.189, 22.65, 213.4, 8.098, 3.794, 15.7, 68.4, 0, 3.03],
         [90.99, 0.6743, 0.3097, 107.4, 32.87, 3, 1.162, 2.648, 31.25, 79.85, 4.444, 1.006, 18.28, 226.3, 2.602, 1.818],
         [193.4, 0.738, 0.1201, -28.42, 90, 3, 1.29, 1.212, 34.2, 101.1, 12.7, 0.9612, 16.46, 376.7, 28.4, 4],
         [79, 0.3641, 0.0809, 40.68, 51.32, 0.4505, 0.5607, 2.995, 39, 114.1, 19, 2.491, 7.564, 176.1, 4.42, 2.014],
         [190.1, 0.7237, 0.2959, 323.8, 54.32, 0.4658, 0.6614, 3.671, 33.68, 76.87, 11.87, 3.634, 19.22, 94.47, 24.33, 3.55],
         [164.6, 2.644, 0.08117, 83.7, 44.21, 1.803, 0.8715, 0.934, 37.61, 20, 10.27, 3.356, 5.4, 23, 35.56, 2.987],
         [174.2979, 3.1263, 0.0612, 226.7702, 22.6482, 0.2313, 2.0158, 1.3477, 28.8878, 205.7589, 16.6738, 2.3313, 13.8784, 153.9844, 0, 3.2567],
         [131.3718, 3.3529, 0.124, 308.9669, 40.6118, 0.2622, 2.2842, 0, 30.4472, 97.6682, 3.8064, 3.8553, 6.4466, 122.6622, 6.6614, 3.0211],
         [129.1884, 3.0625, 0.0677, 300.1743, 28.6115, 0.2099, 1.2503, 0.2904, 31.3806, 47.348, 7.5801, 3.1707, 9.491, 57.6262, 3.9522, 3.1578],
         [139.0189, 2.2698, 0.1446, 165.048, 39.2263, 1.5534, 0.4049, 0.5359, 25.7052, 392.9136, 5.0812, 0.796, 13.589, 186.8908, 0, 0.1794]]


def main():
    """Reads the surface, then every turn re-optimizes the policy parameters from the tracked lander state and prints the best policy's command."""
    def rollout(p: list[float], x: float, y: float, vx: float, vy: float, fuel: int, rot: int, pw: int, cap: float) -> tuple[float, int, int]:
        """Simulates the guidance policy until the lander touches the ground or leaves the zone.
        :param p: policy parameters (described next to LO)
        :param x: horizontal position
        :param y: altitude
        :param vx: horizontal speed
        :param vy: vertical speed
        :param fuel: remaining fuel
        :param rot: current tilt angle
        :param pw: current thrust power
        :param cap: fuel level at which the rollout is abandoned because it can no longer beat the elite
        :return: cost (negated remaining fuel after a safe landing, 0.5 when abandoned, a positive penalty otherwise), first tilt command,
            first power command"""
        vx_max, acc_x, acc_y, h_slow, tilt, kx, ky, dead, v_land, margin, vx_land, ka, look, clear, vy_cruise, acc_max = p
        lb, rb = fx1 + margin, fx2 - margin
        t_slow = max(h_slow, 0) / v_land
        first = None
        for _ in range(500):
            h = y - fy
            c = v_land * v_land + 2 * acc_y * max(0, h - h_slow)
            vyd = -sqrt(c)
            t1 = 0 if vy <= vyd else (vy + sqrt((acc_y * vy * vy + G * c) / (G + acc_y))) / G
            v1 = vy - G * t1
            th = max(1, t1 + t_slow - 1.5 + ((-v1 - v_land) / acc_y if v1 < -v_land else 0))
            q = x + vx * th
            lo, hi = max(2 * (lb - q) / (th * th), (-vx_land - vx) / th), min(2 * (rb - q) / (th * th), (vx_land - vx) / th)
            t = max(lo, min(hi, 0))
            if lo <= hi and abs(t) <= acc_max:
                tx = ka * t
            else:
                vxd = min(sqrt(2 * acc_x * (rb - x)), vx_max) if x < lb else \
                    max(-sqrt(2 * acc_x * (x - lb)), -vx_max if x > rb else min(vx, sqrt(2 * acc_x * (rb - x))))
                tx = kx * (vxd - vx)
                vyd = max(vyd, -vy_cruise)
            if not lb <= x <= rb:
                xa, xb = sorted((x, x + vx * look))
                xa, xb = (max(xa, 0), min(xb, lb)) if x < lb else (max(xa, rb), min(xb, 6999))
                vyd = max(vyd, 0.15 * (max(hts[int(xa)], hts[int(xb)], fy, *vys[nxt[int(xa)]:bisect_left(vxs, xb)]) + clear - y))
            ty = ky * (vyd - vy) + G
            if ty > 0:
                ang = -degrees(atan2(tx, ty))
            elif abs(tx) > dead:
                ty, ang = 0, -tilt if tx > 0 else tilt
            else:
                ty, tx, ang = 0, 0, -degrees(atan2(tx, G))
            if fx1 < x < fx2 and vy < -1 and h < -vy * (abs(rot) / 15 + 1.5):
                ang = 0
            rot = max(rot - 15, min(rot + 15, round(max(-tilt, min(tilt, ang)))))
            i = rot + 90
            pw = min(4, pw + 1, max(pw - 1, round(ty * COS[i] - tx * SIN[i]), 0))
            first = first or (rot, pw)
            pw = min(pw, fuel)
            fuel -= pw
            if fuel <= cap:
                return 0.5, *first
            ax, ay = -pw * SIN[i], pw * COS[i] - G
            nx, ny = x + vx + 0.5 * ax, y + vy + 0.5 * ay
            vx, vy = vx + ax, vy + ay
            if not 0 <= nx < 6999.5 or ny > 2990:
                return 20000 + abs(vx) + abs(vy), *first
            k = int(nx)
            if ny < hts[k] + (hts[k + 1] - hts[k]) * (nx - k) or nxt[int(x)] != nxt[k] and \
                    any(vys[j] > y + (ny - y) * (vxs[j] - x) / (nx - x) for j in range(*sorted((nxt[int(x)], nxt[k])))):
                ex, ey = max(abs(vx), abs(vx - ax)) - 19, max(abs(vy), abs(vy - ay)) - 39
                return (0 if lx <= x <= rx and lx <= nx <= rx else 1000 + max(lx - nx, nx - rx, 0)) + (100 + abs(rot) if rot else 0) \
                    + (100 + 10 * ex if ex > 0 else 0) + (100 + 50 * ey if ey > 0 else 0) or -fuel, *first
            x, y = nx, ny
        return 30000, *first
    gc.disable()
    pts = [tuple(map(int, input().split())) for _ in range(int(input()))]
    fx1, fx2, fy = next((x0, x1, y0) for (x0, y0), (x1, y1) in pairwise(pts) if y0 == y1 and x1 - x0 >= 1000)
    hts = [y0 + (y1 - y0) * (x - x0) / (x1 - x0) for (x0, y0), (x1, y1) in pairwise(pts) for x in range(x0, x1)] + [pts[-1][1]] * 2
    vxs, vys = zip(*pts)
    nxt = [bisect_right(vxs, x) for x in range(7001)]
    lx, rx = fx1 + 20, fx2 - 20
    rng, st, t0, budget, elite = Random(12345), [*map(int, input().split())], perf_counter(), FIRST_BUDGET, SEEDS
    while True:
        queue, scored, fallback = elite[:], [], elite is not SEEDS
        while not scored or perf_counter() - t0 < budget:
            if not queue and fallback and scored[0][0] > 0:
                queue, fallback = SEEDS[:], False
            if queue:
                p = queue.pop(0)
            elif rng.random() < 0.1 or scored[0][0] > 500 and rng.random() < 0.5:
                p = [*map(rng.uniform, LO, HI)]
            else:
                p = scored[min(int(rng.expovariate(0.7)), len(scored) - 1)][3][:]
                for i in rng.choices(range(16), k=rng.randint(1, 3)):
                    p[i] = min(HI[i], max(LO[i], p[i] + rng.gauss(0, rng.choice((0.1, 0.03, 0.005))) * (HI[i] - LO[i])))
            insort(scored, (*rollout(p, *st, -scored[19][0] if len(scored) == 20 else -1), p))
            del scored[20:]
        elite = [e[3] for e in scored[:10]] + queue
        _, rot, pw, _ = scored[0]
        print(rot, pw, flush=True)
        x, y, vx, vy, fuel = st[:5]
        pw = min(pw, fuel)
        ax, ay = -pw * SIN[rot + 90], pw * COS[rot + 90] - G
        st = [x + vx + ax / 2, y + vy + ay / 2, vx + ax, vy + ay, fuel - pw, rot, pw]
        line = [*map(int, input().split())]
        t0, budget = perf_counter(), BUDGET
        if line[4:] != st[4:] or max(abs(a - b) for a, b in zip(st, line[:4])) > 1.01:
            st = line


if __name__ == "__main__":
    main()
