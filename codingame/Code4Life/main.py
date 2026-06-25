import sys

def debug(msg):
    print(msg, file=sys.stderr)

# ---------- Initialization ----------
project_count = int(input())
projects = []
for _ in range(project_count):
    projects.append(list(map(int, input().split())))

TYPES = ['A', 'B', 'C', 'D', 'E']
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}

turn = 0
wait_at_molecules = 0  # Track consecutive waits at MOLECULES

# ---------- Main Loop ----------
while True:
    turn += 1

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

    # --- Utility functions ---
    def ecost(s, exp=None):
        """Effective cost after expertise reduction."""
        if exp is None:
            exp = me['expertise']
        return [max(0, s['cost'][i] - exp[i]) for i in range(5)]

    def etotal(s):
        return sum(ecost(s))

    def can_make(s):
        c = ecost(s)
        return all(me['storage'][i] >= c[i] for i in range(5))

    def feasible(s):
        """Can this sample ever be completed with 10 molecule slots?"""
        return etotal(s) <= 10

    def held():
        return sum(me['storage'])

    def sample_score(s):
        """Value metric for ranking samples. Higher = better."""
        t = etotal(s)
        if t == 0:
            return 1000 + s['health']  # Free samples are amazing
        return s['health'] / t

    def best_subset():
        """Pick the best subset of diagnosed samples whose combined cost fits in 10 molecules.
        Accounts for existing molecules in storage that take up carrying capacity."""
        good = sorted([s for s in my_diag if feasible(s)], key=sample_score, reverse=True)
        result, needed_per_type = [], [0] * 5
        for s in good:
            c = ecost(s)
            new_needed = [needed_per_type[i] + c[i] for i in range(5)]
            # Peak storage = max(what we need, what we already have) for each type
            peak = sum(max(new_needed[i], me['storage'][i]) for i in range(5))
            if peak <= 10:
                result.append(s)
                needed_per_type = new_needed
        return result

    def mol_need(subset):
        """Additional molecules needed beyond what we currently hold."""
        total = [0] * 5
        for s in subset:
            c = ecost(s)
            for i in range(5):
                total[i] += c[i]
        return [max(0, total[i] - me['storage'][i]) for i in range(5)]

    def molecules_can_be_obtained(subset):
        """Check if the molecules we need are obtainable (available or already held)."""
        total = [0] * 5
        for s in subset:
            c = ecost(s)
            for i in range(5):
                total[i] += c[i]
        for i in range(5):
            if total[i] > me['storage'][i] + avail[i]:
                return False
        return True

    def science_progress():
        """How close are we to completing each science project?"""
        exp = me['expertise']
        results = []
        for p in projects:
            remaining = sum(max(0, p[i] - exp[i]) for i in range(5))
            results.append(remaining)
        return results

    def pick_rank():
        """Choose sample rank based on current expertise."""
        te = sum(me['expertise'])
        remaining_turns = 200 - turn
        if remaining_turns < 40:
            # Late game: pick what we can process quickly
            if te >= 8:
                return 2
            return 1
        if te >= 12:
            return 3
        if te >= 5:
            return 2
        return 1

    # --- Compute key state ---
    producible = [s for s in my_diag if can_make(s)]
    subset = best_subset()
    needed = mol_need(subset)
    tot_need = sum(needed)

    pos = me['target']
    eta = me['eta']

    # --- Debug output ---
    debug(f"T{turn} @{pos} eta={eta} sc={me['score']} opp_sc={opp['score']} "
          f"st={me['storage']} ex={me['expertise']} avl={avail}")
    debug(f"  #samp={len(my_all)} #undiag={len(my_undiag)} #diag={len(my_diag)} "
          f"#prod={len(producible)} held={held()}")
    if subset:
        debug(f"  subset={[(s['id'], s['health'], etotal(s)) for s in subset]} "
              f"needed={needed} tot={tot_need}")
    sci = science_progress()
    debug(f"  science_remaining={sci} projects={projects}")

    if eta > 0:
        print("WAIT")
        continue

    # ========== DECISION LOGIC PER MODULE ==========

    def act():
        """Return the action string."""

        # ---------- LABORATORY ----------
        if pos == "LABORATORY":
            if producible:
                # Produce highest-value sample first
                best = max(producible, key=lambda s: s['health'])
                return f"CONNECT {best['id']}"
            # Nothing to produce here, move on
            return decide_next_destination()

        # ---------- MOLECULES ----------
        if pos == "MOLECULES":
            # If we have something producible, consider going to LAB first
            # (producing gains expertise which may reduce costs, and frees capacity)
            if producible:
                if tot_need == 0 or held() >= 10 or tot_need > 3:
                    return "GOTO LABORATORY"

            # If no valid subset and nothing producible, leave
            if not subset and not producible:
                return decide_next_destination()

            # If full on molecules, go produce or swap
            if held() >= 10:
                if producible:
                    return "GOTO LABORATORY"
                # Can't collect more, go diagnose/swap
                return "GOTO DIAGNOSIS"

            # All molecules collected for subset
            if tot_need == 0:
                if producible or subset:
                    return "GOTO LABORATORY"
                return decide_next_destination()

            # Try to collect a needed molecule
            for i in range(5):
                if needed[i] > 0 and avail[i] > 0:
                    return f"CONNECT {TYPES[i]}"

            # Needed molecules not available
            if producible:
                return "GOTO LABORATORY"

            # Wait a few turns for molecules to become available
            if wait_at_molecules < 5 and turn < 185:
                any_hope = any(needed[i] > 0 for i in range(5))
                if any_hope:
                    return "WAIT_AT_MOL"

            # Give up on current subset, try swapping
            return "GOTO DIAGNOSIS"

        # ---------- DIAGNOSIS ----------
        if pos == "DIAGNOSIS":
            # First: diagnose any undiagnosed samples
            if my_undiag:
                return f"CONNECT {my_undiag[0]['id']}"

            # Drop infeasible samples back to cloud
            infeasible = [s for s in my_diag if not feasible(s)]
            if infeasible:
                debug(f"  Dropping infeasible sample {infeasible[0]['id']} "
                      f"(cost={etotal(infeasible[0])})")
                return f"CONNECT {infeasible[0]['id']}"

            # If we have room, try to grab from cloud
            if len(my_all) < 3:
                # Score cloud samples
                good_cloud = [s for s in cloud if feasible(s)]
                if good_cloud:
                    good_cloud.sort(key=sample_score, reverse=True)
                    return f"CONNECT {good_cloud[0]['id']}"
                # No good cloud samples, go to SAMPLES
                return "GOTO SAMPLES"

            # All samples diagnosed and feasible, figure out next step
            return decide_next_destination()

        # ---------- SAMPLES ----------
        if pos == "SAMPLES":
            if len(my_all) < 3:
                return f"CONNECT {pick_rank()}"
            # Full on samples, move on
            if my_undiag:
                return "GOTO DIAGNOSIS"
            return decide_next_destination()

        # ---------- START / Unknown ----------
        return "GOTO SAMPLES"

    def decide_next_destination():
        """When done at current module, figure out where to go."""
        # Priority 1: diagnose undiagnosed samples
        if my_undiag:
            return "GOTO DIAGNOSIS"

        # Priority 2: produce what we can
        if producible:
            return "GOTO LABORATORY"

        # Priority 3: need to drop infeasible or grab cloud samples
        infeasible = [s for s in my_diag if not feasible(s)]
        if infeasible:
            return "GOTO DIAGNOSIS"

        # Priority 4: need more samples
        if len(my_all) == 0:
            return "GOTO SAMPLES"

        # Priority 5: collect molecules for our subset
        if subset and tot_need > 0:
            return "GOTO MOLECULES"

        # Priority 6: subset ready, go produce
        if subset:
            return "GOTO LABORATORY"

        # Priority 7: need more or different samples
        if len(my_all) < 3:
            # Decide: cloud (DIAGNOSIS) or fresh (SAMPLES)?
            good_cloud = [s for s in cloud if feasible(s)]
            if good_cloud:
                return "GOTO DIAGNOSIS"
            return "GOTO SAMPLES"

        # Fallback: get new samples
        return "GOTO SAMPLES"

    action = act()

    # Track waiting at molecules
    if action == "WAIT_AT_MOL":
        wait_at_molecules += 1
        action = "WAIT"
    elif pos == "MOLECULES" and not action.startswith("CONNECT"):
        # Leaving molecules, reset counter
        wait_at_molecules = 0
    elif pos != "MOLECULES":
        wait_at_molecules = 0

    debug(f"  -> {action}")
    print(action)