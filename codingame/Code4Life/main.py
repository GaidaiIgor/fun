import sys
from itertools import combinations

MTYPES = ['A', 'B', 'C', 'D', 'E']
TERMINAL_MAX = 5

def debug(*args):
    print(*args, file=sys.stderr, flush=True)

class Sample:
    def __init__(self, sid, carried_by, rank, gain, health, cost):
        self.id = sid
        self.carried_by = carried_by
        self.rank = rank
        self.gain = gain
        self.health = health
        self.cost = cost

    @property
    def diagnosed(self):
        return self.cost['A'] >= 0

    def net_cost(self, exp):
        return {t: max(0, self.cost[t] - exp[t]) for t in MTYPES}

    def total_net_cost(self, exp):
        return sum(self.net_cost(exp).values())

    def can_produce(self, storage, exp):
        nc = self.net_cost(exp)
        return all(storage[t] >= nc[t] for t in MTYPES)

    def efficiency(self, exp):
        return self.health / max(1, self.total_net_cost(exp))

    def __repr__(self):
        return f"S({self.id},r{self.rank},hp{self.health})"


def needed_for_set(samples, storage, exp):
    """Additional molecules needed beyond current storage to produce all samples."""
    result = {t: 0 for t in MTYPES}
    temp = dict(storage)
    for s in samples:
        nc = s.net_cost(exp)
        for t in MTYPES:
            extra = max(0, nc[t] - temp[t])
            result[t] += extra
            temp[t] = max(0, temp[t] - nc[t])
    return result


def best_subset(samples, storage, exp, max_mol=10):
    """Highest-HP subset of <=3 samples fitting carry cap and per-type terminal cap."""
    total_stored = sum(storage.values())
    remaining = max(0, max_mol - total_stored)
    best_combo, best_hp = [], -1
    for r in range(min(3, len(samples)), 0, -1):
        for combo in combinations(samples, r):
            lst = list(combo)
            needed = needed_for_set(lst, storage, exp)
            if (sum(needed.values()) <= remaining and
                    all(needed[t] <= TERMINAL_MAX for t in MTYPES)):
                hp = sum(s.health for s in lst)
                if hp > best_hp:
                    best_hp, best_combo = hp, lst
        if best_hp >= 0:
            break
    return best_combo


def _best_cloud_fit(cloud_diag, held_diag, storage, exp, available):
    """
    Highest-efficiency cloud sample that:
      1. Fits within remaining carry capacity
      2. Requires no more than TERMINAL_MAX extra of any single type
      3. Every type still needed (beyond storage) has available > 0
         (prevents re-downloading a sample we just dropped due to shortage)
    """
    best, best_eff = None, -1
    total_stored = sum(storage.values())
    for cs in cloud_diag:
        needed = needed_for_set(held_diag + [cs], storage, exp)
        addl = sum(needed.values())
        if (addl <= (10 - total_stored) and
                all(needed[t] <= TERMINAL_MAX for t in MTYPES) and
                all(needed[t] == 0 or available[t] > 0 for t in MTYPES)):
            eff = cs.efficiency(exp)
            if eff > best_eff:
                best_eff, best = eff, cs
    return best


def desired_rank(total_exp):
    if total_exp >= 6:
        return 3
    if total_exp >= 3:
        return 2
    return 1


class Game:
    def __init__(self):
        self.projects = []
        self.stuck_turns = 0
        self.force_drop = False


def decide(game, me, opp, available, samples):
    target = me['target']
    eta = me['eta']
    storage = me['storage']
    exp = me['expertise']

    if eta > 0:
        return "WAIT"

    mine = [s for s in samples if s.carried_by == 0]
    cloud = [s for s in samples if s.carried_by == -1]
    undiag = [s for s in mine if not s.diagnosed]
    diag = [s for s in mine if s.diagnosed]
    cloud_diag = [s for s in cloud if s.diagnosed]
    producible = [s for s in diag if s.can_produce(storage, exp)]
    total_mol = sum(storage.values())
    total_exp = sum(exp.values())
    rank = desired_rank(total_exp)

    debug(f"[{target}] mine={len(mine)}({len(undiag)}u) prod={len(producible)} "
          f"mol={total_mol} exp={total_exp} rank={rank} stuck={game.stuck_turns} fd={game.force_drop}")
    debug(f"  storage={storage}  avail={available}")

    # ── LABORATORY ────────────────────────────────────────────────────
    if target == 'LABORATORY':
        if producible:
            best = max(producible, key=lambda s: s.health)
            game.stuck_turns = 0
            game.force_drop = False
            debug(f"  -> Producing {best.id} ({best.health} HP)")
            return f"CONNECT {best.id}"

        if game.force_drop:
            return "GOTO DIAGNOSIS"

        if undiag:
            return "GOTO DIAGNOSIS"

        if diag:
            needed = needed_for_set(diag, storage, exp)
            if sum(needed.values()) > 0:
                if any(needed[t] > 0 and available[t] > 0 for t in MTYPES):
                    game.stuck_turns = 0
                    return "GOTO MOLECULES"
                game.stuck_turns += 1
                debug(f"  -> Stuck at LAB ({game.stuck_turns}/4)")
                if game.stuck_turns > 4:
                    game.stuck_turns = 0
                    game.force_drop = True
                    return "GOTO DIAGNOSIS"
                return "WAIT"

        best_cloud = _best_cloud_fit(cloud_diag, diag, storage, exp, available)
        if best_cloud and len(mine) < 3 and best_cloud.efficiency(exp) > 5:
            return "GOTO DIAGNOSIS"
        if len(mine) < 3:
            return "GOTO SAMPLES"
        return "GOTO MOLECULES"

    # ── MOLECULES ─────────────────────────────────────────────────────
    if target == 'MOLECULES':
        if not diag:
            return "GOTO DIAGNOSIS" if undiag else "GOTO SAMPLES"

        if producible:
            game.stuck_turns = 0
            return "GOTO LABORATORY"

        needed = needed_for_set(diag, storage, exp)
        total_need = sum(needed.values())

        if total_need == 0 or total_mol >= 10:
            game.stuck_turns = 0
            return "GOTO LABORATORY"

        for t in MTYPES:
            if needed[t] > 0 and available[t] > 0 and total_mol < 10:
                game.stuck_turns = 0
                return f"CONNECT {t}"

        # All needed types at 0 — wait or escalate to drop
        game.stuck_turns += 1
        debug(f"  -> Stuck at MOLECULES ({game.stuck_turns}/4)")
        if game.stuck_turns <= 4:
            return "WAIT"
        game.stuck_turns = 0
        game.force_drop = True
        return "GOTO DIAGNOSIS"

    # ── DIAGNOSIS ─────────────────────────────────────────────────────
    if target == 'DIAGNOSIS':

        # 1. Diagnose undiagnosed samples
        if undiag:
            return f"CONNECT {undiag[0].id}"

        # 2. Force-drop the blocking sample
        if game.force_drop and diag:
            game.force_drop = False
            def block_score(s):
                ns = needed_for_set([s], storage, exp)
                return sum(ns[t] for t in MTYPES if available[t] == 0)
            worst = max(diag, key=lambda s: (block_score(s), -s.efficiency(exp)))
            debug(f"  -> Force-dropping {worst.id}")
            return f"CONNECT {worst.id}"

        # 3. Drop samples outside budget or terminal-cap feasibility
        if diag:
            keep = best_subset(diag, storage, exp)
            drop = [s for s in diag if s not in keep]
            if drop:
                worst = min(drop, key=lambda s: s.efficiency(exp))
                debug(f"  -> Dropping {worst.id} (infeasible)")
                return f"CONNECT {worst.id}"

        # 4. Already producible → skip MOLECULES
        if producible:
            return "GOTO LABORATORY"

        # 5. Try to fill up to 3 from cloud (availability-gated to prevent re-download loops)
        if len(mine) < 3:
            best_cloud = _best_cloud_fit(cloud_diag, diag, storage, exp, available)
            if best_cloud:
                debug(f"  -> Cloud {best_cloud.id} (eff={best_cloud.efficiency(exp):.2f})")
                return f"CONNECT {best_cloud.id}"

        # 6. Go to MOLECULES with any diagnosed sample (even 1 is fine)
        if diag:
            return "GOTO MOLECULES"

        return "GOTO SAMPLES"

    # ── SAMPLES ───────────────────────────────────────────────────────
    if target == 'SAMPLES':
        if len(mine) >= 3:
            return "GOTO DIAGNOSIS"
        return f"CONNECT {rank}"

    # ── START / FALLBACK ──────────────────────────────────────────────
    if cloud_diag and len(mine) < 3:
        best_cloud = _best_cloud_fit(cloud_diag, [], storage, exp, available)
        if best_cloud and best_cloud.efficiency(exp) > 5:
            return "GOTO DIAGNOSIS"
    return "GOTO SAMPLES"


def parse_turn():
    players = []
    for _ in range(2):
        line = input().split()
        target = line[0]
        eta = int(line[1])
        score = int(line[2])
        storage = {t: int(line[3 + i]) for i, t in enumerate(MTYPES)}
        expertise = {t: int(line[8 + i]) for i, t in enumerate(MTYPES)}
        players.append({'target': target, 'eta': eta, 'score': score,
                        'storage': storage, 'expertise': expertise})
    avail_raw = list(map(int, input().split()))
    available = {t: avail_raw[i] for i, t in enumerate(MTYPES)}
    n = int(input())
    samples = []
    for _ in range(n):
        parts = input().split()
        sid = int(parts[0])
        carried = int(parts[1])
        rank = int(parts[2])
        gain = parts[3]
        health = int(parts[4])
        cost = {t: int(parts[5 + i]) for i, t in enumerate(MTYPES)}
        samples.append(Sample(sid, carried, rank, gain, health, cost))
    return players, available, samples


def main():
    game = Game()
    n = int(input())
    for _ in range(n):
        parts = list(map(int, input().split()))
        game.projects.append({t: parts[i] for i, t in enumerate(MTYPES)})
    debug(f"Projects: {game.projects}")
    while True:
        players, available, samples = parse_turn()
        action = decide(game, players[0], players[1], available, samples)
        debug(f"  => {action}\n")
        print(action)


main()