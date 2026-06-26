import sys
import traceback

SAMPLES = "SAMPLES"
DIAGNOSIS = "DIAGNOSIS"
MOLECULES = "MOLECULES"
LABORATORY = "LABORATORY"
MODULES = (SAMPLES, DIAGNOSIS, MOLECULES, LABORATORY)
MOLS = ['A', 'B', 'C', 'D', 'E']


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# --- Initialization ----------------------------------------------------------
project_count = int(input())
projects = []
for _ in range(project_count):
    projects.append(list(map(int, input().split())))
log(f"Init: projects={projects}")


# --- Helpers -----------------------------------------------------------------
def effective_cost(sample, expertise):
    """Cost vector after expertise reduction (clamped to 0)."""
    return [max(0, sample['cost'][i] - expertise[i]) for i in range(5)]


def need_for_samples(samples, expertise):
    """Sum of effective costs across all diagnosed samples, per type."""
    need = [0] * 5
    for s in samples:
        if not s['diagnosed']:
            continue
        ec = effective_cost(s, expertise)
        for i in range(5):
            need[i] += ec[i]
    return need


def is_feasible(subset, expertise, storage):
    """Can we carry all needed molecules + existing storage within capacity 10?"""
    if not subset:
        return True
    need = need_for_samples(subset, expertise)
    # Total carry after fetching: each type ends up at max(need[i], storage[i]).
    return sum(max(need[i], storage[i]) for i in range(5)) <= 10


def can_complete(sample, expertise, storage):
    """Do we have enough molecules to submit this sample to LAB right now?"""
    if not sample['diagnosed']:
        return False
    ec = effective_cost(sample, expertise)
    return all(storage[i] >= ec[i] for i in range(5))


def choose_rank(expertise):
    """Pick which rank of sample to grab based on total expertise."""
    te = sum(expertise)
    if te < 4:
        return 1
    elif te < 9:
        return 2
    return 3


def best_subset(samples, expertise, storage):
    """Enumerate all subsets; pick the feasible one with the highest total health.
    Tie-break: prefer smaller subset (faster cycle)."""
    n = len(samples)
    best = None
    best_health = -1
    best_size = 999
    for mask in range(1, 1 << n):
        subset = [samples[i] for i in range(n) if mask & (1 << i)]
        if not is_feasible(subset, expertise, storage):
            continue
        health = sum(s['health'] for s in subset)
        size = len(subset)
        if health > best_health or (health == best_health and size < best_size):
            best_health = health
            best_size = size
            best = subset
    return best


# --- Decision logic ----------------------------------------------------------
def decide(my_target, my_eta, my_storage, my_expertise, available,
           my_samples, cloud_samples):
    # In transit — game ignores our action.
    if my_eta > 0:
        return "WAIT"

    # Start area or any unknown location — head to SAMPLES.
    if my_target not in MODULES:
        return f"GOTO {SAMPLES}"

    # ---- SAMPLES module ----
    if my_target == SAMPLES:
        if len(my_samples) < 3:
            rank = choose_rank(my_expertise)
            return f"CONNECT {rank}"
        return f"GOTO {DIAGNOSIS}"

    # ---- DIAGNOSIS module ----
    if my_target == DIAGNOSIS:
        # 1. Diagnose any undiagnosed sample in hand.
        for s in my_samples:
            if not s['diagnosed']:
                return f"CONNECT {s['id']}"
        # 2. No samples? Restart cycle.
        if not my_samples:
            return f"GOTO {SAMPLES}"
        # 3. All diagnosed — choose best feasible subset, drop the rest to cloud.
        chosen = best_subset(my_samples, my_expertise, my_storage) or []
        chosen_ids = {s['id'] for s in chosen}
        for s in my_samples:
            if s['id'] not in chosen_ids:
                log(f"  Dropping infeasible sample {s['id']} (cost={s['cost']} h={s['health']})")
                return f"CONNECT {s['id']}"
        # 4. If nothing was feasible, dropped everything — restart.
        if not chosen:
            return f"GOTO {SAMPLES}"
        log(f"  Subset: ids={chosen_ids} health={sum(s['health'] for s in chosen)}")
        return f"GOTO {MOLECULES}"

    # ---- MOLECULES module ----
    if my_target == MOLECULES:
        if not my_samples:
            return f"GOTO {SAMPLES}"
        need = need_for_samples(my_samples, my_expertise)
        total_carried = sum(my_storage)
        # Fetch a molecule we still need (if available and capacity allows).
        for i in range(5):
            if need[i] > my_storage[i] and available[i] > 0 and total_carried < 10:
                return f"CONNECT {MOLS[i]}"
        # If something is already completable, head to LAB and submit it.
        for s in my_samples:
            if can_complete(s, my_expertise, my_storage):
                return f"GOTO {LABORATORY}"
        # All needs satisfied (no fetch required) — head to LAB.
        if all(my_storage[i] >= need[i] for i in range(5)):
            return f"GOTO {LABORATORY}"
        # Stuck waiting on a molecule that's currently exhausted.
        log("  Stuck at MOLECULES: needed type unavailable, no completable sample.")
        return "WAIT"

    # ---- LABORATORY module ----
    if my_target == LABORATORY:
        for s in my_samples:
            if can_complete(s, my_expertise, my_storage):
                log(f"  Completing sample {s['id']} for {s['health']} HP, gain={s['gain']}")
                return f"CONNECT {s['id']}"
        # No completable samples remaining.
        if my_samples:
            # Still have samples needing more molecules.
            return f"GOTO {MOLECULES}"
        return f"GOTO {SAMPLES}"

    return "WAIT"


# --- Main loop ---------------------------------------------------------------
turn = 0
while True:
    turn += 1
    try:
        parts = input().split()
        my_target = parts[0]
        my_eta = int(parts[1])
        my_score = int(parts[2])
        my_storage = list(map(int, parts[3:8]))
        my_expertise = list(map(int, parts[8:13]))

        parts = input().split()
        opp_target = parts[0]
        opp_eta = int(parts[1])
        opp_score = int(parts[2])
        opp_storage = list(map(int, parts[3:8]))
        opp_expertise = list(map(int, parts[8:13]))

        available = list(map(int, input().split()))

        sample_count = int(input())
        samples = []
        for _ in range(sample_count):
            s_parts = input().split()
            sample = {
                'id': int(s_parts[0]),
                'carried_by': int(s_parts[1]),
                'rank': int(s_parts[2]),
                'gain': s_parts[3],
                'health': int(s_parts[4]),
                'cost': list(map(int, s_parts[5:10])),
            }
            # Undiagnosed samples have cost fields of -1.
            sample['diagnosed'] = sample['cost'][0] != -1
            samples.append(sample)

        my_samples = [s for s in samples if s['carried_by'] == 0]
        cloud_samples = [s for s in samples if s['carried_by'] == -1]

        log(f"--- Turn {turn} ---")
        log(f"Me:  tgt={my_target} eta={my_eta} score={my_score} "
            f"storage={my_storage} expertise={my_expertise}")
        log(f"Available={available}")
        for s in my_samples:
            log(f"  carry id={s['id']} r={s['rank']} diag={s['diagnosed']} "
                f"h={s['health']} cost={s['cost']} gain={s['gain']}")
        log(f"Cloud: {[(s['id'], s['rank'], s['health'], s['cost']) for s in cloud_samples]}")
        log(f"Opp: tgt={opp_target} eta={opp_eta} score={opp_score} "
            f"storage={opp_storage} expertise={opp_expertise}")

        cmd = decide(my_target, my_eta, my_storage, my_expertise, available,
                     my_samples, cloud_samples)
        log(f"=> {cmd}")
        print(cmd, flush=True)
    except EOFError:
        break
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())
        # Fail-safe so we never produce invalid output and instant-lose.
        print("WAIT", flush=True)