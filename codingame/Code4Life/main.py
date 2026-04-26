import sys
from itertools import combinations

MTYPES = ['A', 'B', 'C', 'D', 'E']

def debug(*args):
    print(*args, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Sample:
    def __init__(self, sid, carried_by, rank, gain, health, cost):
        self.id = sid
        self.carried_by = carried_by
        self.rank = rank
        self.gain = gain      # molecule type of expertise gained on research
        self.health = health
        self.cost = cost      # dict {A:n, B:n, C:n, D:n, E:n}

    @property
    def diagnosed(self):
        # Undiagnosed samples have cost fields of -1
        return self.cost['A'] >= 0

    def net_cost(self, exp):
        """Cost after subtracting expertise."""
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
        return f"Sample(id={self.id}, rank={self.rank}, hp={self.health}, cost={self.cost})"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def needed_for_set(samples, storage, exp):
    """Extra molecules needed (beyond current storage) to produce all samples."""
    result = {t: 0 for t in MTYPES}
    temp = dict(storage)
    for s in samples:
        nc = s.net_cost(exp)
        for t in MTYPES:
            extra = max(0, nc[t] - temp[t])
            result[t] += extra
            temp[t] = max(0, temp[t] - nc[t])
    return result


def best_subset(samples, exp, max_mol=10):
    """
    Return the highest-total-HP subset of ≤3 samples whose combined
    net molecule cost fits within max_mol.
    Falls back to smaller subsets if needed.
    Returns [] if nothing fits.
    """
    zero_storage = {t: 0 for t in MTYPES}
    n = min(3, len(samples))
    best_combo, best_hp = [], -1

    for r in range(n, 0, -1):
        for combo in combinations(samples, r):
            needed = needed_for_set(list(combo), zero_storage, exp)
            if sum(needed.values()) <= max_mol:
                hp = sum(s.health for s in combo)
                if hp > best_hp:
                    best_hp = hp
                    best_combo = list(combo)
        if best_hp >= 0:
            break

    return best_combo


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def decide(me, opp, available, samples, projects):
    target   = me['target']
    eta      = me['eta']
    storage  = me['storage']
    exp      = me['expertise']

    if eta > 0:
        return "WAIT"

    # Partition samples
    mine        = [s for s in samples if s.carried_by == 0]
    cloud       = [s for s in samples if s.carried_by == -1]

    undiag      = [s for s in mine  if not s.diagnosed]
    diag        = [s for s in mine  if s.diagnosed]
    cloud_diag  = [s for s in cloud if s.diagnosed]

    producible  = [s for s in diag if s.can_produce(storage, exp)]
    total_mol   = sum(storage.values())
    total_exp   = sum(exp.values())

    debug(f"[{target}] eta={eta} mine={len(mine)} undiag={len(undiag)} "
          f"diag={len(diag)} producible={len(producible)} "
          f"mol={total_mol} exp_total={total_exp}")
    debug(f"  storage={storage}  expertise={exp}  available={available}")

    # ------------------------------------------------------------------
    # LABORATORY
    # ------------------------------------------------------------------
    if target == 'LABORATORY':
        # Produce as many medicines as possible
        if producible:
            best = max(producible, key=lambda s: s.health)
            debug(f"  -> Producing sample {best.id} ({best.health} HP)")
            return f"CONNECT {best.id}"

        # Still have undiagnosed samples → diagnose first
        if undiag:
            return "GOTO DIAGNOSIS"

        # Diagnosed samples but missing molecules
        if diag:
            needed = needed_for_set(diag, storage, exp)
            if sum(needed.values()) > 0:
                return "GOTO MOLECULES"

        # Need more samples
        if len(mine) < 3:
            if cloud_diag:          # free pick-up, skip the SAMPLES trip
                return "GOTO DIAGNOSIS"
            return "GOTO SAMPLES"

        return "GOTO MOLECULES"

    # ------------------------------------------------------------------
    # MOLECULES
    # ------------------------------------------------------------------
    if target == 'MOLECULES':
        if not diag:
            return "GOTO DIAGNOSIS" if undiag else "GOTO SAMPLES"

        needed      = needed_for_set(diag, storage, exp)
        total_need  = sum(needed.values())

        if total_need == 0 or total_mol >= 10:
            return "GOTO LABORATORY"

        # Collect one molecule we still need
        for t in MTYPES:
            if needed[t] > 0 and available[t] > 0 and total_mol < 10:
                return f"CONNECT {t}"

        # Required molecules temporarily unavailable → go produce what we can
        debug("  -> Some molecules unavailable, heading to lab")
        return "GOTO LABORATORY"

    # ------------------------------------------------------------------
    # DIAGNOSIS
    # ------------------------------------------------------------------
    if target == 'DIAGNOSIS':
        # 1. Diagnose any undiagnosed sample we carry
        if undiag:
            return f"CONNECT {undiag[0].id}"

        # 2. Drop samples that cannot fit in our 10-molecule budget
        if diag:
            keep = best_subset(diag, exp)
            drop = [s for s in diag if s not in keep]
            if drop:
                worst = min(drop, key=lambda s: s.efficiency(exp))
                debug(f"  -> Uploading sample {worst.id} to cloud (doesn't fit budget)")
                return f"CONNECT {worst.id}"

        # 3. Opportunistically grab a good cloud sample if we have room
        if len(mine) < 3:
            best_cloud   = None
            best_eff     = -1
            for cs in cloud_diag:
                # Would we still fit in 10 molecules if we added this?
                test_needed = needed_for_set(diag + [cs], storage, exp)
                if sum(test_needed.values()) <= 10:
                    eff = cs.efficiency(exp)
                    if eff > best_eff:
                        best_eff   = eff
                        best_cloud = cs
            if best_cloud:
                debug(f"  -> Taking cloud sample {best_cloud.id} (eff={best_eff:.2f})")
                return f"CONNECT {best_cloud.id}"

        if diag:
            return "GOTO MOLECULES"

        return "GOTO SAMPLES"

    # ------------------------------------------------------------------
    # SAMPLES
    # ------------------------------------------------------------------
    if target == 'SAMPLES':
        if len(mine) >= 3:
            return "GOTO DIAGNOSIS"

        # Choose rank based on accumulated expertise
        # More expertise → we can afford rank-3's expensive costs
        if total_exp >= 5:
            rank = 3
        elif total_exp >= 2:
            rank = 2
        else:
            rank = 2        # rank 2 gives 10-30 HP at 5-8 molecules: good ratio early on

        return f"CONNECT {rank}"

    # ------------------------------------------------------------------
    # START (or any other state, e.g. first turn)
    # ------------------------------------------------------------------
    if cloud_diag and len(mine) < 3:
        return "GOTO DIAGNOSIS"
    return "GOTO SAMPLES"


# ---------------------------------------------------------------------------
# I/O loop
# ---------------------------------------------------------------------------

def parse_turn():
    players = []
    for _ in range(2):
        line = input().split()
        target   = line[0]
        eta      = int(line[1])
        score    = int(line[2])
        storage  = {t: int(line[3 + i]) for i, t in enumerate(MTYPES)}
        expertise = {t: int(line[8 + i]) for i, t in enumerate(MTYPES)}
        players.append({
            'target':    target,
            'eta':       eta,
            'score':     score,
            'storage':   storage,
            'expertise': expertise,
        })

    avail_raw = list(map(int, input().split()))
    available = {t: avail_raw[i] for i, t in enumerate(MTYPES)}

    n = int(input())
    samples = []
    for _ in range(n):
        parts     = input().split()
        sid       = int(parts[0])
        carried   = int(parts[1])
        rank      = int(parts[2])
        gain      = parts[3]
        health    = int(parts[4])
        cost      = {t: int(parts[5 + i]) for i, t in enumerate(MTYPES)}
        samples.append(Sample(sid, carried, rank, gain, health, cost))

    return players, available, samples


def main():
    project_count = int(input())
    projects = []
    for _ in range(project_count):
        parts = list(map(int, input().split()))
        projects.append({t: parts[i] for i, t in enumerate(MTYPES)})
    debug(f"Science projects: {projects}")

    while True:
        players, available, samples = parse_turn()
        me  = players[0]
        opp = players[1]

        action = decide(me, opp, available, samples, projects)
        debug(f"  => {action}\n")
        print(action)


main()