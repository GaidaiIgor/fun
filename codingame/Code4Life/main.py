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
    if sum(expertise) >= 6:
        return 3
    return 2

def compute_best_plan(diagnosed, storage, expertise, available):
    """Best subset of diagnosed.
    Score: (HP, gatherable, -cost).
    - Higher HP wins.
    - Tiebreak: prefer plans whose full need is covered by current storage+avail (so we
      can complete them without waiting for refills).
    - Tiebreak: lower total cost.
    Constraint: peak inventory <= 10.

    CRITICAL: HP must come first in the tuple. v4 had (gatherable, HP, -cost) and the
    empty subset is trivially "gatherable" (no needs, no shortfall), so it scored (1,0,0)
    and beat every non-empty blocked plan -> bot did nothing for 95 turns.
    """
    n = len(diagnosed)
    best_plan = []
    best_score = (-1, -1, -10000)
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
        gatherable = all(storage[i] + available[i] >= total_needed[i] for i in range(5))
        score = (total_health, 1 if gatherable else 0, -total_c)
        if score > best_score:
            best_score = score
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

    # Drop diagnosed samples whose net cost > 10 (literally cannot fit in inventory).
    for s in diagnosed:
        if sum(net_cost(s, me['expertise'])) > 10:
            log(f"  Drop impossible sample {s['id']}")
            if target == 'DIAGNOSIS':
                return f"CONNECT {s['id']}"
            return "GOTO DIAGNOSIS"

    plan, total_needed = compute_best_plan(diagnosed, me['storage'], me['expertise'], available)
    still_need = [max(0, total_needed[i] - me['storage'][i]) for i in range(5)]
    producible_in_plan = [s for s in plan if can_produce(s, me['storage'], me['expertise'])]
    can_gather_something = any(still_need[i] > 0 and available[i] > 0 for i in range(5))
    plan_gatherable = all(me['storage'][i] + available[i] >= total_needed[i] for i in range(5))

    log(f"  diag={len(diagnosed)} undiag={len(undiagnosed)} "
        f"plan={[s['id'] for s in plan]} hp={sum(s['health'] for s in plan)} "
        f"still_need={still_need} producible={[s['id'] for s in producible_in_plan]} "
        f"plan_gatherable={plan_gatherable}")

    # --- Module-specific actions ---

    if target == 'LABORATORY' and producible_in_plan:
        producible_in_plan.sort(key=lambda s: -s['health'])
        return f"CONNECT {producible_in_plan[0]['id']}"

    if target == 'DIAGNOSIS' and undiagnosed:
        return f"CONNECT {undiagnosed[0]['id']}"

    # NOTE: No drop-blocker logic. v3 sent sample 19 to cloud as a "blocker" and opp
    # downloaded it and scored the deciding 40 HP. A diagnosed sample on the cloud is
    # a partial deal the opponent can finish with their different expertise. Just hold
    # blockers - if we can't make them, neither can opp, since they're in our hand.

    if target == 'MOLECULES':
        cands = [(available[i], i) for i in range(5) if still_need[i] > 0 and available[i] > 0]
        if cands:
            cands.sort()  # least-available first (race opp for scarce mols)
            return f"CONNECT {MOL_NAMES[cands[0][1]]}"
        # Nothing to gather right now.
        if producible_in_plan:
            return "GOTO LABORATORY"
        if sum(still_need) > 0:
            if len(my_samples) < 3:
                return "GOTO SAMPLES"  # find new options
            return "WAIT"  # 3 samples and fully stuck; wait for opp to release

    if target == 'SAMPLES' and len(my_samples) < 3:
        return f"CONNECT {pick_rank(me['expertise'])}"

    # --- Decide next destination ---

    if undiagnosed:
        return "GOTO DIAGNOSIS"

    if producible_in_plan and sum(still_need) == 0:
        return "GOTO LABORATORY"

    if sum(still_need) > 0:
        if can_gather_something:
            return "GOTO MOLECULES"
        if producible_in_plan:
            return "GOTO LABORATORY"
        if len(my_samples) < 3:
            return "GOTO SAMPLES"
        return "GOTO MOLECULES"  # park at MOL waiting for refills

    if producible_in_plan:
        return "GOTO LABORATORY"

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
    sample_brief = [
        (s['id'],
         'M' if s['carried_by']==0 else ('O' if s['carried_by']==1 else 'C'),
         s['rank'], s['gain'], s['health'], s['cost'])
        for s in samples
    ]
    log(f"  samples: {sample_brief}")

    try:
        action = decide(me, opp, available, samples)
    except Exception as e:
        log(f"ERROR: {e}")
        action = "WAIT"

    log(f"  -> {action}")
    print(action)