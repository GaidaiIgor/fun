import sys

def log(*args):
    print(*args, file=sys.stderr, flush=True)

MOL_NAMES = ['A', 'B', 'C', 'D', 'E']

# --- Helpers ---

def net_cost(sample, expertise):
    return [max(0, sample['cost'][i] - expertise[i]) for i in range(5)]

def can_produce(sample, storage, expertise):
    return all(sample['cost'][i] - expertise[i] <= storage[i] for i in range(5))

def pick_rank(expertise):
    """Skip rank 1 (low HP, unreliable). Rank 2 until enough expertise for rank 3."""
    if sum(expertise) >= 8:
        return 3
    return 2

def compute_best_plan(diagnosed, storage, expertise):
    """
    Highest-HP subset of diagnosed samples whose peak inventory fits in 10 mols.
    NOTE: deliberately does NOT require availability to currently cover the need
    -- we plan, then gather what we can and wait at MOLECULES for the rest.
    Tiebreak: prefer lower total cost (frees room/turns for later).
    """
    n = len(diagnosed)
    best_plan = []
    best_health = -1
    best_cost = 1000
    for mask in range(1 << n):
        subset = [diagnosed[i] for i in range(n) if mask & (1 << i)]
        total_needed = [0]*5
        for s in subset:
            nc = net_cost(s, expertise)
            for i in range(5):
                total_needed[i] += nc[i]
        peak = sum(max(storage[i], total_needed[i]) for i in range(5))
        if peak > 10:
            continue
        total_health = sum(s['health'] for s in subset)
        total_c = sum(total_needed)
        if total_health > best_health or (total_health == best_health and total_c < best_cost):
            best_health = total_health
            best_cost = total_c
            best_plan = subset
    total_needed = [0]*5
    for s in best_plan:
        nc = net_cost(s, expertise)
        for i in range(5):
            total_needed[i] += nc[i]
    return best_plan, total_needed

# --- I/O ---

def read_robot():
    parts = input().split()
    return {
        'target': parts[0],
        'eta': int(parts[1]),
        'score': int(parts[2]),
        'storage': list(map(int, parts[3:8])),
        'expertise': list(map(int, parts[8:13])),
    }

def read_sample():
    parts = input().split()
    return {
        'id': int(parts[0]),
        'carried_by': int(parts[1]),
        'rank': int(parts[2]),
        'gain': parts[3],
        'health': int(parts[4]),
        'cost': list(map(int, parts[5:10])),
    }

# --- Decision logic ---

def decide(me, opp, available, samples):
    if me['eta'] > 0:
        return "WAIT"

    my_samples = [s for s in samples if s['carried_by'] == 0]
    undiagnosed = [s for s in my_samples if s['gain'] == '0']
    diagnosed   = [s for s in my_samples if s['gain'] != '0']
    target = me['target']

    # Drop diagnosed samples with net cost > 10 (can never make them).
    for s in diagnosed:
        if sum(net_cost(s, me['expertise'])) > 10:
            log(f"  Drop impossible sample {s['id']} net_cost={sum(net_cost(s, me['expertise']))}")
            if target == 'DIAGNOSIS':
                return f"CONNECT {s['id']}"
            return "GOTO DIAGNOSIS"

    plan, total_needed = compute_best_plan(diagnosed, me['storage'], me['expertise'])
    still_need = [max(0, total_needed[i] - me['storage'][i]) for i in range(5)]
    producible_in_plan = [s for s in plan if can_produce(s, me['storage'], me['expertise'])]

    log(f"  diag={len(diagnosed)} undiag={len(undiagnosed)} "
        f"plan={[s['id'] for s in plan]} hp={sum(s['health'] for s in plan)} "
        f"still_need={still_need} can_produce={[s['id'] for s in producible_in_plan]}")

    # --- Module-specific actions (when AT a module) ---
    if target == 'LABORATORY' and producible_in_plan:
        producible_in_plan.sort(key=lambda s: -s['health'])
        return f"CONNECT {producible_in_plan[0]['id']}"

    if target == 'DIAGNOSIS' and undiagnosed:
        return f"CONNECT {undiagnosed[0]['id']}"

    if target == 'MOLECULES':
        # Gather scarce needed molecule first so opp doesn't race us for it.
        cands = [(available[i], i) for i in range(5) if still_need[i] > 0 and available[i] > 0]
        if cands:
            cands.sort()  # least-available first
            return f"CONNECT {MOL_NAMES[cands[0][1]]}"
        # Nothing useful to gather right now.
        if producible_in_plan and sum(still_need) == 0:
            return "GOTO LABORATORY"
        if sum(still_need) > 0:
            # KEY FIX: wait HERE for needed mols to free up.
            # Don't wander off and lose position.
            return "WAIT"
        # plan empty & nothing to do here -> fall through to defaults

    if target == 'SAMPLES' and len(my_samples) < 3:
        return f"CONNECT {pick_rank(me['expertise'])}"

    # --- Decide next destination ---

    # Diagnose first so we know real costs before planning gather.
    if undiagnosed:
        return "GOTO DIAGNOSIS"

    # Got everything for plan -> produce.
    if producible_in_plan and sum(still_need) == 0:
        return "GOTO LABORATORY"

    # KEY FIX: go to MOLECULES even if needed mols not currently available.
    # We'll wait there for opp to release them.
    if sum(still_need) > 0:
        return "GOTO MOLECULES"

    # Edge: producible but partial gather blocked -> at least go produce what we can.
    if producible_in_plan:
        return "GOTO LABORATORY"

    # Fill up on samples.
    if len(my_samples) < 3:
        return "GOTO SAMPLES"

    return "WAIT"

# --- Init ---

project_count = int(input())
projects = []
for _ in range(project_count):
    projects.append(list(map(int, input().split())))
log(f"Projects: {projects}")

# --- Main loop ---

turn = 0
while True:
    turn += 1
    me = read_robot()
    opp = read_robot()
    available = list(map(int, input().split()))
    sample_count = int(input())
    samples = [read_sample() for _ in range(sample_count)]

    log(f"=== T{turn} | tgt={me['target']} eta={me['eta']} "
        f"score={me['score']} (opp:{opp['score']}) | "
        f"stor={me['storage']} exp={me['expertise']} | avail={available}")

    try:
        action = decide(me, opp, available, samples)
    except Exception as e:
        log(f"ERROR: {e}")
        action = "WAIT"

    log(f"  -> {action}")
    print(action)