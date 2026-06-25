import sys

def debug(msg):
    print(msg, file=sys.stderr)

project_count = int(input())
projects = []
for _ in range(project_count):
    projects.append(list(map(int, input().split())))

TYPES = ['A', 'B', 'C', 'D', 'E']

turn = 0
mol_wait = 0

while True:
    turn += 1
    remaining = 200 - turn

    players = []
    for i in range(2):
        parts = input().split()
        players.append({
            'target': parts[0], 'eta': int(parts[1]), 'score': int(parts[2]),
            'storage': [int(x) for x in parts[3:8]],
            'expertise': [int(x) for x in parts[8:13]],
        })
    me, opp = players[0], players[1]
    avail = list(map(int, input().split()))

    n_samp = int(input())
    samples = []
    for _ in range(n_samp):
        p = input().split()
        samples.append({
            'id': int(p[0]), 'carriedBy': int(p[1]), 'rank': int(p[2]),
            'gain': p[3], 'health': int(p[4]),
            'cost': [int(x) for x in p[5:10]],
        })

    my_all = [s for s in samples if s['carriedBy'] == 0]
    my_undiag = [s for s in my_all if s['health'] < 0]
    my_diag = [s for s in my_all if s['health'] >= 0]
    cloud = [s for s in samples if s['carriedBy'] == -1]

    pos = me['target']
    eta = me['eta']

    # ---- Utility functions ----
    def ecost(s):
        return [max(0, s['cost'][i] - me['expertise'][i]) for i in range(5)]

    def etotal(s):
        return sum(ecost(s))

    def can_make(s):
        c = ecost(s)
        return all(me['storage'][i] >= c[i] for i in range(5))

    def feasible(s):
        return etotal(s) <= 10

    def held():
        return sum(me['storage'])

    def is_blocked(s):
        """Sample needs molecules that aren't available and we don't have enough stored."""
        c = ecost(s)
        for i in range(5):
            need_more = c[i] - me['storage'][i]
            if need_more > 0 and avail[i] < need_more:
                return True
        return False

    def science_bonus(s):
        if s['gain'] == '0':
            return 0
        gi = TYPES.index(s['gain'])
        bonus = 0
        for p in projects:
            rem = [max(0, p[i] - me['expertise'][i]) for i in range(5)]
            tr = sum(rem)
            if tr > 0 and rem[gi] > 0:
                bonus = max(bonus, max(0, 15 - tr) * 3)
        return bonus

    def sample_score(s):
        t = etotal(s)
        base = s['health'] + science_bonus(s)
        if t == 0:
            return 1000 + base
        c = ecost(s)
        penalty = 0
        for i in range(5):
            shortfall = max(0, c[i] - me['storage'][i])
            if shortfall > 0:
                if avail[i] == 0:
                    penalty += shortfall * 50  # Massive penalty for completely blocked
                elif avail[i] < shortfall:
                    penalty += (shortfall - avail[i]) * 10
        return (base - penalty) / t

    def best_subset():
        good = sorted([s for s in my_diag if feasible(s)], key=sample_score, reverse=True)
        # Filter out samples with very negative scores (completely blocked)
        good = [s for s in good if sample_score(s) > -10]
        result, needed_per_type = [], [0] * 5
        for s in good:
            c = ecost(s)
            new_needed = [needed_per_type[i] + c[i] for i in range(5)]
            peak = sum(max(new_needed[i], me['storage'][i]) for i in range(5))
            if peak <= 10:
                result.append(s)
                needed_per_type = new_needed
        return result

    def mol_need(sub):
        total = [0] * 5
        for s in sub:
            c = ecost(s)
            for i in range(5):
                total[i] += c[i]
        return [max(0, total[i] - me['storage'][i]) for i in range(5)]

    def pick_rank():
        te = sum(me['expertise'])
        if remaining < 30:
            return 2 if te >= 6 else 1
        if te >= 8: return 3
        if te >= 3: return 2
        return 1

    # ---- Compute state ----
    producible = [s for s in my_diag if can_make(s)]
    subset = best_subset()
    needed = mol_need(subset)
    tot_need = sum(needed)
    prod_value = sum(s['health'] for s in producible)

    # Blocked: diagnosed samples that need unavailable molecules
    blocked = [s for s in my_diag if is_blocked(s) and not can_make(s)]

    debug(f"T{turn} @{pos} eta={eta} sc={me['score']} opp={opp['score']} "
          f"st={me['storage']} ex={me['expertise']} avl={avail} held={held()}")
    debug(f"  #all={len(my_all)} #ud={len(my_undiag)} #dg={len(my_diag)} "
          f"#prod={len(producible)}(v={prod_value}) #blocked={len(blocked)} rem={remaining}")
    if subset:
        debug(f"  sub={[(s['id'],s['health'],etotal(s)) for s in subset]} "
              f"need={needed} tot={tot_need}")
    if blocked:
        debug(f"  blocked={[(s['id'],s['health']) for s in blocked]}")

    if eta > 0:
        print("WAIT")
        continue

    def should_get_more():
        if len(my_all) >= 3: return False
        if len(my_diag) >= 2 and tot_need > 0 and not all(is_blocked(s) for s in subset):
            return False  # Have 2+ viable diagnosed samples with work
        if len(my_all) == 0: return True
        if remaining < 20: return len(my_diag) == 0 and len(my_undiag) == 0
        return True

    def any_obtainable_needed():
        """At least some needed molecules are available."""
        return any(needed[i] > 0 and avail[i] > 0 for i in range(5))

    def act():
        global mol_wait

        # ---- LABORATORY ----
        if pos == "LABORATORY":
            if producible:
                best = max(producible, key=lambda s: s['health'])
                return f"CONNECT {best['id']}"
            return where_next()

        # ---- MOLECULES ----
        if pos == "MOLECULES":
            if producible and (tot_need == 0 or held() >= 10 or
                              (prod_value >= 10 and tot_need > 3)):
                mol_wait = 0
                return "GOTO LABORATORY"
            if not subset and not producible:
                mol_wait = 0
                return where_next()
            if held() >= 10:
                mol_wait = 0
                return "GOTO LABORATORY" if producible else "GOTO DIAGNOSIS"
            if tot_need == 0:
                mol_wait = 0
                return "GOTO LABORATORY"

            # Try to collect
            for i in range(5):
                if needed[i] > 0 and avail[i] > 0:
                    mol_wait = 0
                    return f"CONNECT {TYPES[i]}"

            # Can't collect
            mol_wait += 1
            if producible:
                mol_wait = 0
                return "GOTO LABORATORY"
            if mol_wait <= 2 and remaining > 30 and any_obtainable_needed():
                return "WAIT"
            mol_wait = 0
            return "GOTO DIAGNOSIS"

        # ---- DIAGNOSIS ----
        if pos == "DIAGNOSIS":
            if my_undiag:
                return f"CONNECT {my_undiag[0]['id']}"

            # Drop infeasible (cost > 10)
            infeas = [s for s in my_diag if not feasible(s)]
            if infeas:
                debug(f"  Drop infeasible {infeas[0]['id']}")
                return f"CONNECT {infeas[0]['id']}"

            # Drop blocked samples (need molecules with 0 availability)
            # But only if ALL subset samples are blocked or we have no viable path
            if blocked and (not subset or all(is_blocked(s) for s in subset)):
                # Drop the lowest-value blocked sample
                worst = min(blocked, key=lambda s: s['health'])
                debug(f"  Drop blocked {worst['id']} (needs unavail mols)")
                return f"CONNECT {worst['id']}"

            # If we have room and should get more
            if len(my_all) < 3 and should_get_more():
                good_cloud = sorted([s for s in cloud if feasible(s) and not is_blocked(s)],
                                   key=sample_score, reverse=True)
                if good_cloud:
                    return f"CONNECT {good_cloud[0]['id']}"
                # Also try blocked cloud samples if nothing else
                any_cloud = sorted([s for s in cloud if feasible(s)],
                                  key=sample_score, reverse=True)
                if any_cloud:
                    return f"CONNECT {any_cloud[0]['id']}"
                return "GOTO SAMPLES"

            # Figure out where to go - but NEVER return to DIAGNOSIS
            dest = where_next()
            if dest == "GOTO DIAGNOSIS":
                # We're stuck at DIAGNOSIS - emergency drop worst sample
                if my_diag:
                    worst = min(my_diag, key=lambda s: sample_score(s))
                    debug(f"  Emergency drop {worst['id']} to break loop")
                    return f"CONNECT {worst['id']}"
                return "GOTO SAMPLES"
            return dest

        # ---- SAMPLES ----
        if pos == "SAMPLES":
            if len(my_all) < 3 and should_get_more():
                return f"CONNECT {pick_rank()}"
            if my_undiag:
                return "GOTO DIAGNOSIS"
            dest = where_next()
            if dest == "GOTO SAMPLES":
                # Already here - get a sample or leave
                if len(my_all) < 3:
                    return f"CONNECT {pick_rank()}"
                return "GOTO DIAGNOSIS"
            return dest

        # ---- START ----
        good_cloud = [s for s in cloud if feasible(s) and sample_score(s) > 3]
        if good_cloud:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES"

    def where_next():
        """Decide next destination. May return current pos - caller must handle."""
        if my_undiag:
            return "GOTO DIAGNOSIS"
        if producible:
            return "GOTO LABORATORY"
        if any(not feasible(s) for s in my_diag):
            return "GOTO DIAGNOSIS"

        # All subset samples blocked?
        if subset and all(is_blocked(s) for s in subset):
            return "GOTO DIAGNOSIS"  # Go drop blocked samples

        if not my_all:
            good_cloud = [s for s in cloud if feasible(s)]
            return "GOTO DIAGNOSIS" if good_cloud else "GOTO SAMPLES"

        if subset and tot_need > 0:
            if any_obtainable_needed():
                return "GOTO MOLECULES"
            return "GOTO DIAGNOSIS"  # Can't get molecules, swap samples

        if subset:
            return "GOTO LABORATORY"

        if should_get_more():
            good_cloud = [s for s in cloud if feasible(s)]
            return "GOTO DIAGNOSIS" if good_cloud else "GOTO SAMPLES"

        if my_diag:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES"

    action = act()
    debug(f"  -> {action}")
    print(action)
