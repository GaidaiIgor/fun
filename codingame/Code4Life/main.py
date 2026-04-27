import sys
from itertools import combinations

MTYPES = ['A', 'B', 'C', 'D', 'E']
TERMINAL_MAX = 5   # terminal holds at most 5 of each molecule type

def debug(*args):
    print(*args, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Sample:
    def __init__(self, sid, carried_by, rank, gain, health, cost):
        self.id         = sid
        self.carried_by = carried_by
        self.rank       = rank
        self.gain       = gain
        self.health     = health
        self.cost       = cost      # {A:n, B:n, C:n, D:n, E:n}  (-1 = undiagnosed)

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
        tc = self.total_net_cost(exp)
        return self.health / max(1, tc)

    def __repr__(self):
        return f"S(id={self.id},rank={self.rank},hp={self.health})"


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------

def needed_for_set(samples, storage, exp):
    """
    Return how many ADDITIONAL molecules of each type are needed to
    produce every sample in `samples`, given current `storage`.
    Storage is consumed sequentially across samples.
    """
    result = {t: 0 for t in MTYPES}
    temp   = dict(storage)
    for s in samples:
        nc = s.net_cost(exp)
        for t in MTYPES:
            extra      = max(0, nc[t] - temp[t])
            result[t] += extra
            temp[t]    = max(0, temp[t] - nc[t])
    return result


def best_subset(samples, storage, exp, max_mol=10):
    """
    Find the highest-total-HP subset of up to 3 samples such that:
      1. additional molecules needed <= remaining carry capacity
      2. no single type requires more than TERMINAL_MAX (terminal cap)
    Returns [] if no feasible subset exists.
    """
    total_stored = sum(storage.values())
    remaining    = max(0, max_mol - total_stored)
    n            = min(3, len(samples))
    best_combo   = []
    best_hp      = -1

    for r in range(n, 0, -1):
        for combo in combinations(samples, r):
            lst    = list(combo)
            needed = needed_for_set(lst, storage, exp)
            addl   = sum(needed.values())
            if addl <= remaining and all(needed[t] <= TERMINAL_MAX for t in MTYPES):
                hp = sum(s.health for s in lst)
                if hp > best_hp:
                    best_hp    = hp
                    best_combo = lst
        if best_hp >= 0:
            break

    return best_combo


# ---------------------------------------------------------------------------
# Per-game persistent state
# ---------------------------------------------------------------------------

class Game:
    def __init__(self):
        self.projects    = []
        self.stuck_turns = 0     # consecutive turns unable to collect any needed molecule
        self.force_drop  = False # flag: go to DIAGNOSIS to drop the blocking sample


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def decide(game, me, opp, available, samples):
    target   = me['target']
    eta      = me['eta']
    storage  = me['storage']
    exp      = me['expertise']

    if eta > 0:
        return "WAIT"

    mine       = [s for s in samples if s.carried_by ==  0]
    cloud      = [s for s in samples if s.carried_by == -1]
    undiag     = [s for s in mine  if not s.diagnosed]
    diag       = [s for s in mine  if s.diagnosed]
    cloud_diag = [s for s in cloud if s.diagnosed]
    producible = [s for s in diag  if s.can_produce(storage, exp)]
    total_mol  = sum(storage.values())
    total_exp  = sum(exp.values())

    debug(f"[{target}] mine={len(mine)}({len(undiag)}u) prod={len(producible)} "
          f"mol={total_mol} exp={total_exp} stuck={game.stuck_turns} fd={game.force_drop}")
    debug(f"  storage={storage}  avail={available}")

    # ==================================================================
    # LABORATORY
    # ==================================================================
    if target == 'LABORATORY':
        if producible:
            best = max(producible, key=lambda s: s.health)
            game.stuck_turns = 0
            game.force_drop  = False
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
                # Needed molecules are all at 0 right now — wait rather than bounce
                game.stuck_turns += 1
                debug(f"  -> Stuck at LAB ({game.stuck_turns}/5)")
                if game.stuck_turns > 5:
                    game.stuck_turns = 0
                    game.force_drop  = True
                    return "GOTO DIAGNOSIS"
                return "WAIT"

        if len(mine) < 3:
            return "GOTO DIAGNOSIS" if cloud_diag else "GOTO SAMPLES"
        return "GOTO MOLECULES"

    # ==================================================================
    # MOLECULES
    # ==================================================================
    if target == 'MOLECULES':
        if not diag:
            return "GOTO DIAGNOSIS" if undiag else "GOTO SAMPLES"

        needed     = needed_for_set(diag, storage, exp)
        total_need = sum(needed.values())

        if total_need == 0 or total_mol >= 10:
            game.stuck_turns = 0
            return "GOTO LABORATORY"

        for t in MTYPES:
            if needed[t] > 0 and available[t] > 0 and total_mol < 10:
                game.stuck_turns = 0
                return f"CONNECT {t}"

        # All needed molecule types are at 0 in the terminal
        game.stuck_turns += 1
        debug(f"  -> Stuck at MOLECULES ({game.stuck_turns}/5), needed={needed}")
        if game.stuck_turns <= 5:
            return "WAIT"
        # Give up: drop the blocking sample
        game.stuck_turns = 0
        game.force_drop  = True
        return "GOTO DIAGNOSIS"

    # ==================================================================
    # DIAGNOSIS
    # ==================================================================
    if target == 'DIAGNOSIS':

        # 1. Diagnose any undiagnosed sample we carry
        if undiag:
            return f"CONNECT {undiag[0].id}"

        # 2. Force-drop the sample that caused a deadlock
        if game.force_drop and diag:
            game.force_drop = False
            def block_score(s):
                ns = needed_for_set([s], storage, exp)
                return sum(ns[t] for t in MTYPES if available[t] == 0)
            worst = max(diag, key=lambda s: (block_score(s), -s.efficiency(exp)))
            debug(f"  -> Force-dropping {worst.id}")
            return f"CONNECT {worst.id}"

        # 3. Drop samples that exceed budget or hit terminal cap
        if diag:
            keep = best_subset(diag, storage, exp)
            drop = [s for s in diag if s not in keep]
            if drop:
                worst = min(drop, key=lambda s: s.efficiency(exp))
                debug(f"  -> Budget/cap drop {worst.id}")
                return f"CONNECT {worst.id}"

        # 4. If samples already producible, head straight to lab
        if producible:
            return "GOTO LABORATORY"

        # 5. Ensure we have at least 2 samples before heading to MOLECULES
        #    (batching 2-3 per cycle avoids excessive movement overhead)
        if len(mine) < 2:
            for cs in cloud_diag:
                test_needed = needed_for_set(diag + [cs], storage, exp)
                addl = sum(test_needed.values())
                if (addl <= (10 - total_mol) and
                        all(test_needed[t] <= TERMINAL_MAX for t in MTYPES)):
                    debug(f"  -> Cloud sample {cs.id} (filling to 2)")
                    return f"CONNECT {cs.id}"
            return "GOTO SAMPLES"

        # 6. Opportunistically grab a 3rd sample from the cloud
        if len(mine) < 3:
            best_cloud, best_eff = None, -1
            for cs in cloud_diag:
                test_needed = needed_for_set(diag + [cs], storage, exp)
                addl = sum(test_needed.values())
                if (addl <= (10 - total_mol) and
                        all(test_needed[t] <= TERMINAL_MAX for t in MTYPES)):
                    eff = cs.efficiency(exp)
                    if eff > best_eff:
                        best_eff   = eff
                        best_cloud = cs
            if best_cloud:
                debug(f"  -> Cloud sample {best_cloud.id} (eff={best_eff:.2f})")
                return f"CONNECT {best_cloud.id}"

        if diag:
            return "GOTO MOLECULES"

        return "GOTO SAMPLES"

    # ==================================================================
    # SAMPLES
    # ==================================================================
    if target == 'SAMPLES':
        if len(mine) >= 3:
            return "GOTO DIAGNOSIS"

        rank = 3 if total_exp >= 5 else 2
        return f"CONNECT {rank}"

    # ==================================================================
    # START (first turn / any unrecognised state)
    # ==================================================================
    if cloud_diag and len(mine) < 3:
        return "GOTO DIAGNOSIS"
    return "GOTO SAMPLES"


# ---------------------------------------------------------------------------
# I/O loop
# ---------------------------------------------------------------------------

def parse_turn():
    players = []
    for _ in range(2):
        line      = input().split()
        target    = line[0]
        eta       = int(line[1])
        score     = int(line[2])
        storage   = {t: int(line[3+i]) for i, t in enumerate(MTYPES)}
        expertise = {t: int(line[8+i]) for i, t in enumerate(MTYPES)}
        players.append({'target': target, 'eta': eta, 'score': score,
                        'storage': storage, 'expertise': expertise})

    avail_raw = list(map(int, input().split()))
    available = {t: avail_raw[i] for i, t in enumerate(MTYPES)}

    n = int(input())
    samples = []
    for _ in range(n):
        parts   = input().split()
        sid     = int(parts[0])
        carried = int(parts[1])
        rank    = int(parts[2])
        gain    = parts[3]
        health  = int(parts[4])
        cost    = {t: int(parts[5+i]) for i, t in enumerate(MTYPES)}
        samples.append(Sample(sid, carried, rank, gain, health, cost))

    return players, available, samples


def main():
    game = Game()
    n    = int(input())
    for _ in range(n):
        parts = list(map(int, input().split()))
        game.projects.append({t: parts[i] for i, t in enumerate(MTYPES)})
    debug(f"Science projects: {game.projects}")

    while True:
        players, available, samples = parse_turn()
        me  = players[0]
        opp = players[1]

        action = decide(game, me, opp, available, samples)
        debug(f"  => {action}\n")
        print(action)


main()