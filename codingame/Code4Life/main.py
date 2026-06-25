import sys

def debug(msg):
    print(msg, file=sys.stderr)

# ---------- Initialization ----------
project_count = int(input())
projects = []
for _ in range(project_count):
    projects.append(list(map(int, input().split())))

TYPES = ['A', 'B', 'C', 'D', 'E']

turn = 0
mol_wait = 0  # consecutive waits at MOLECULES

# ---------- Main Loop ----------
while True:
    turn += 1
    remaining_turns = 200 - turn

    # --- Read state ---
    players = []
    for i in range(2):
        parts = input().split()
        players.append({
            'target': parts[0],
            'eta': int(parts[1]),
            'score': int(parts[2]),
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
            'id': int(p[0]),
            'carriedBy': int(p[1]),
            'rank': int(p[2]),
            'gain': p[3],
            'health': int(p[4]),
            'cost': [int(x) for x in p[5:10]],
        })

    # --- Categorize samples ---
    my_all = [s for s in samples if s['carriedBy'] == 0]
    my_undiag = [s for s in my_all if s['health'] < 0]
    my_diag = [s for s in my_all if s['health'] >= 0]
    cloud = [s for s in samples if s['carriedBy'] == -1]

    pos = me['target']
    eta = me['eta']

    # --- Utility functions ---
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

    def science_bonus(s):
        """Extra score for samples whose gain type helps toward a science project."""
        if s['gain'] == '0':
            return 0
        gi = TYPES.index(s['gain'])
        bonus = 0
        for p in projects:
            remaining = [max(0, p[i] - me['expertise'][i]) for i in range(5)]
            total_remaining = sum(remaining)
            if total_remaining > 0 and remaining[gi] > 0:
                # Closer projects get more bonus
                closeness = max(0, 15 - total_remaining)
                bonus = max(bonus, closeness * 3)
        return bonus

    def sample_score(s):
        """Value metric for ranking samples. Higher = better."""
        t = etotal(s)
        base = s['health'] + science_bonus(s)
        if t == 0:
            return 1000 + base
        # Penalize samples needing unavailable molecules
        c = ecost(s)
        penalty = 0
        for i in range(5):
            shortfall = max(0, c[i] - me['storage'][i])
            if shortfall > 0 and avail[i] < shortfall:
                penalty += (shortfall - avail[i]) * 5  # Heavy penalty
        return (base - penalty) / t

    def best_subset():
        """Pick best subset of diagnosed samples fitting in 10 molecules,
        considering current storage waste."""
        good = sorted([s for s in my_diag if feasible(s)], key=sample_score, reverse=True)
        result, needed_per_type = [], [0] * 5
        for s in good:
            c = ecost(s)
            new_needed = [needed_per_type[i] + c[i] for i in range(5)]
            peak = sum(max(new_needed[i], me['storage'][i]) for i in range(5))
            if peak <= 10:
                result.append(s)
                needed_per_type = new_needed
        return result

    def mol_need(subset):
        total = [0] * 5
        for s in subset:
            c = ecost(s)
            for i in range(5):
                total[i] += c[i]
        return [max(0, total[i] - me['storage'][i]) for i in range(5)]

    def can_obtain_molecules(needed):
        """Check if all needed molecules are currently available."""
        return all(needed[i] <= avail[i] for i in range(5))

    def pick_rank():
        te = sum(me['expertise'])
        if remaining_turns < 30:
            if te >= 6:
                return 2
            return 1
        if te >= 8:
            return 3
        if te >= 3:
            return 2
        return 1

    # --- Compute key state ---
    producible = [s for s in my_diag if can_make(s)]
    subset = best_subset()
    needed = mol_need(subset)
    tot_need = sum(needed)

    # Value of producible samples
    prod_value = sum(s['health'] for s in producible)

    # --- Debug output ---
    debug(f"T{turn} @{pos} eta={eta} sc={me['score']} opp={opp['score']} "
          f"st={me['storage']} ex={me['expertise']} avl={avail} held={held()}")
    debug(f"  #all={len(my_all)} #ud={len(my_undiag)} #dg={len(my_diag)} "
          f"#prod={len(producible)}(val={prod_value}) remain={remaining_turns}")
    if subset:
        debug(f"  sub={[(s['id'],s['health'],etotal(s),s['gain']) for s in subset]} "
              f"need={needed} tot={tot_need}")
    sci = [sum(max(0, p[i]-me['expertise'][i]) for i in range(5)) for p in projects]
    debug(f"  sci_remain={sci}")

    if eta > 0:
        print("WAIT")
        continue

    # ========== DECISION LOGIC ==========

    # Check: should we even bother going to SAMPLES for a 3rd sample?
    def should_get_more_samples():
        """Decide if it's worth getting more samples vs proceeding with what we have."""
        if len(my_all) >= 3:
            return False
        if len(my_diag) >= 2 and tot_need > 0:
            # We have 2+ diagnosed samples with work to do - just proceed
            return False
        if len(my_all) == 0:
            return True
        if remaining_turns < 25:
            # Late game: work with what we have if we have anything
            return len(my_diag) == 0 and len(my_undiag) == 0
        return True

    def act():
        global mol_wait

        # ---------- LABORATORY ----------
        if pos == "LABORATORY":
            if producible:
                best = max(producible, key=lambda s: s['health'])
                return f"CONNECT {best['id']}"
            return decide_next()

        # ---------- MOLECULES ----------
        if pos == "MOLECULES":
            # If we have high-value producible, go produce
            if producible and (tot_need == 0 or held() >= 10 or
                              (prod_value >= 10 and tot_need > 3)):
                mol_wait = 0
                return "GOTO LABORATORY"

            if not subset and not producible:
                mol_wait = 0
                return decide_next()

            if held() >= 10:
                mol_wait = 0
                if producible:
                    return "GOTO LABORATORY"
                return "GOTO DIAGNOSIS"  # swap samples

            if tot_need == 0:
                mol_wait = 0
                return "GOTO LABORATORY"

            # Try to collect
            for i in range(5):
                if needed[i] > 0 and avail[i] > 0:
                    mol_wait = 0
                    return f"CONNECT {TYPES[i]}"

            # Can't collect what we need
            mol_wait += 1
            if producible:
                mol_wait = 0
                return "GOTO LABORATORY"

            # Only wait 2 turns max, then bail
            if mol_wait <= 2 and remaining_turns > 30:
                return "WAIT"

            # Give up - go swap samples
            mol_wait = 0
            return "GOTO DIAGNOSIS"

        # ---------- DIAGNOSIS ----------
        if pos == "DIAGNOSIS":
            if my_undiag:
                return f"CONNECT {my_undiag[0]['id']}"

            # Drop infeasible samples
            infeasible = [s for s in my_diag if not feasible(s)]
            if infeasible:
                debug(f"  Dropping infeasible {infeasible[0]['id']}")
                return f"CONNECT {infeasible[0]['id']}"

            # If we have room and should get more, try cloud first
            if len(my_all) < 3 and should_get_more_samples():
                good_cloud = [s for s in cloud if feasible(s)]
                if good_cloud:
                    good_cloud.sort(key=sample_score, reverse=True)
                    return f"CONNECT {good_cloud[0]['id']}"
                return "GOTO SAMPLES"

            # We have diagnosed samples - figure out where to go
            return decide_next()

        # ---------- SAMPLES ----------
        if pos == "SAMPLES":
            if len(my_all) < 3 and should_get_more_samples():
                return f"CONNECT {pick_rank()}"
            if my_undiag:
                return "GOTO DIAGNOSIS"
            return decide_next()

        # ---------- START ----------
        # Check if cloud has good samples right away
        good_cloud = [s for s in cloud if feasible(s) and sample_score(s) > 3]
        if good_cloud:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES"

    def decide_next():
        """Where to go next."""
        if my_undiag:
            return "GOTO DIAGNOSIS"

        # Producible? Go produce
        if producible:
            return "GOTO LABORATORY"

        # Infeasible to drop?
        if any(not feasible(s) for s in my_diag):
            return "GOTO DIAGNOSIS"

        # No samples at all
        if not my_all:
            # Check cloud
            good_cloud = [s for s in cloud if feasible(s)]
            if good_cloud:
                return "GOTO DIAGNOSIS"
            return "GOTO SAMPLES"

        # Have a valid subset needing molecules
        if subset and tot_need > 0:
            # Check if molecules are obtainable
            if can_obtain_molecules(needed) or any(avail[i] > 0 and needed[i] > 0 for i in range(5)):
                return "GOTO MOLECULES"
            else:
                # None of the needed molecules are available at all
                # Go swap samples at diagnosis
                return "GOTO DIAGNOSIS"

        # Subset ready
        if subset:
            return "GOTO LABORATORY"

        # Need different/more samples
        if should_get_more_samples():
            good_cloud = [s for s in cloud if feasible(s)]
            if good_cloud:
                return "GOTO DIAGNOSIS"
            return "GOTO SAMPLES"

        # Have diagnosed but no valid subset (all infeasible?)
        if my_diag:
            return "GOTO DIAGNOSIS"

        return "GOTO SAMPLES"

    action = act()
    debug(f"  -> {action}")
    print(action)
