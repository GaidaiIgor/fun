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
recently_dropped = set()  # Track IDs dropped this visit to DIAGNOSIS

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
    te = sum(me['expertise'])

    # Clear recently_dropped when we leave DIAGNOSIS
    if pos != "DIAGNOSIS" and eta == 0:
        recently_dropped.clear()

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
        best = 0
        for p in projects:
            rem = [max(0, p[i] - me['expertise'][i]) for i in range(5)]
            tr = sum(rem)
            if tr > 0 and rem[gi] > 0:
                best = max(best, max(0, 15 - tr) * 3)
        return best

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
                    penalty += shortfall * 50
                elif avail[i] < shortfall:
                    penalty += (shortfall - avail[i]) * 10
        return (base - penalty) / t

    def best_subset():
        good = sorted([s for s in my_diag if feasible(s)], key=sample_score, reverse=True)
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
        if remaining < 25:
            return 2 if te >= 6 else 1
        if te >= 8: return 3
        return 2

    producible = [s for s in my_diag if can_make(s)]
    subset = best_subset()
    needed = mol_need(subset)
    tot_need = sum(needed)
    prod_value = sum(s['health'] for s in producible)
    blocked = [s for s in my_diag if is_blocked(s) and not can_make(s)]

    debug(f"T{turn} @{pos} eta={eta} sc={me['score']} opp={opp['score']} "
          f"st={me['storage']} ex={me['expertise']} avl={avail} h={held()}")
    debug(f"  #a={len(my_all)} #u={len(my_undiag)} #d={len(my_diag)} "
          f"#p={len(producible)}(v={prod_value}) #b={len(blocked)} rem={remaining}")
    if subset:
        debug(f"  sub={[(s['id'],s['health'],etotal(s)) for s in subset]} "
              f"need={needed} tot={tot_need}")

    if eta > 0:
        print("WAIT")
        continue

    def should_get_more():
        if len(my_all) >= 3: return False
        viable = [s for s in subset if not is_blocked(s)]
        if len(viable) >= 2 and tot_need > 0:
            return False
        if len(my_all) == 0: return True
        if remaining < 20: return len(my_diag) == 0 and len(my_undiag) == 0
        return True

    def any_needed_avail():
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
            if tot_need == 0 and (producible or subset):
                mol_wait = 0
                return "GOTO LABORATORY"
            if held() >= 10:
                mol_wait = 0
                if producible:
                    return "GOTO LABORATORY"
                return "GOTO DIAGNOSIS"
            if not subset and not producible:
                mol_wait = 0
                return where_next()
            if remaining < 15 and producible:
                mol_wait = 0
                return "GOTO LABORATORY"

            # Try to collect
            for i in range(5):
                if needed[i] > 0 and avail[i] > 0:
                    mol_wait = 0
                    return f"CONNECT {TYPES[i]}"

            # Can't collect - wait up to 2 turns
            mol_wait += 1
            if mol_wait <= 2 and remaining > 25:
                return "WAIT"

            # Gave up waiting
            mol_wait = 0
            if producible:
                return "GOTO LABORATORY"
            return "GOTO DIAGNOSIS"

        # ---- DIAGNOSIS ----
        if pos == "DIAGNOSIS":
            if my_undiag:
                return f"CONNECT {my_undiag[0]['id']}"

            # Drop infeasible
            infeas = [s for s in my_diag if not feasible(s)]
            if infeas:
                recently_dropped.add(infeas[0]['id'])
                return f"CONNECT {infeas[0]['id']}"

            # Drop low-value samples when full
            if len(my_all) >= 3:
                low_val = [s for s in my_diag if s['health'] <= 1 and etotal(s) > 2]
                if low_val:
                    recently_dropped.add(low_val[0]['id'])
                    debug(f"  Drop low-val {low_val[0]['id']}")
                    return f"CONNECT {low_val[0]['id']}"

            # Drop blocked samples when all subset samples are blocked
            if blocked and (not subset or all(is_blocked(s) for s in subset)):
                worst = min(blocked, key=lambda s: s['health'])
                recently_dropped.add(worst['id'])
                debug(f"  Drop blocked {worst['id']}")
                return f"CONNECT {worst['id']}"

            # If room and should get more - ONLY pick unblocked, non-recently-dropped cloud samples
            if len(my_all) < 3 and should_get_more():
                good_cloud = sorted(
                    [s for s in cloud if feasible(s) and not is_blocked(s)
                     and s['id'] not in recently_dropped],
                    key=sample_score, reverse=True)
                if good_cloud:
                    return f"CONNECT {good_cloud[0]['id']}"
                # No good cloud samples - go to SAMPLES
                return "GOTO SAMPLES"

            # Figure out where to go - prevent self-loop
            dest = where_next()
            if dest == "GOTO DIAGNOSIS":
                if my_diag:
                    worst = min(my_diag, key=lambda s: sample_score(s))
                    recently_dropped.add(worst['id'])
                    debug(f"  Emergency drop {worst['id']}")
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
                if len(my_all) < 3:
                    return f"CONNECT {pick_rank()}"
                return "GOTO DIAGNOSIS"
            return dest

        # ---- START ----
        good_cloud = [s for s in cloud if feasible(s) and not is_blocked(s)
                      and sample_score(s) > 5]
        if good_cloud:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES"

    def where_next():
        if my_undiag:
            return "GOTO DIAGNOSIS"
        if producible:
            return "GOTO LABORATORY"
        if any(not feasible(s) for s in my_diag):
            return "GOTO DIAGNOSIS"
        if len(my_all) >= 3 and any(s['health'] <= 1 and etotal(s) > 2 for s in my_diag):
            return "GOTO DIAGNOSIS"
        if subset and all(is_blocked(s) for s in subset):
            return "GOTO DIAGNOSIS"
        if not my_all:
            good_cloud = [s for s in cloud if feasible(s) and not is_blocked(s)]
            return "GOTO DIAGNOSIS" if good_cloud else "GOTO SAMPLES"
        if subset and tot_need > 0:
            if any_needed_avail():
                return "GOTO MOLECULES"
            return "GOTO DIAGNOSIS"
        if subset:
            return "GOTO LABORATORY"
        if should_get_more():
            good_cloud = [s for s in cloud if feasible(s) and not is_blocked(s)]
            return "GOTO DIAGNOSIS" if good_cloud else "GOTO SAMPLES"
        if my_diag:
            return "GOTO DIAGNOSIS"
        return "GOTO SAMPLES"

    action = act()
    debug(f"  -> {action}")
    print(action)
