import sys

TYPES = ['A', 'B', 'C', 'D', 'E']
MOL_POOL_CAP = 5  # max molecules of each type in circulation
STORAGE_CAP = 10


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def read_player():
    parts = input().split()
    return {
        'target': parts[0],
        'eta': int(parts[1]),
        'score': int(parts[2]),
        'storage': [int(x) for x in parts[3:8]],
        'expertise': [int(x) for x in parts[8:13]],
    }


def read_sample():
    parts = input().split()
    return {
        'id': int(parts[0]),
        'carried_by': int(parts[1]),
        'rank': int(parts[2]),
        'gain': parts[3],
        'health': int(parts[4]),
        'cost': [int(x) for x in parts[5:10]],
    }


def is_diagnosed(s):
    return s['cost'][0] >= 0


def needed_molecules(s, me):
    return [max(0, s['cost'][i] - me['expertise'][i] - me['storage'][i]) for i in range(5)]


def can_complete_now(s, me):
    return sum(needed_molecules(s, me)) == 0


def total_need_after_exp(s, me):
    """Total molecules required given current expertise, ignoring storage."""
    return sum(max(0, s['cost'][i] - me['expertise'][i]) for i in range(5))


def is_impossible(s, me):
    """Per-type cost exceeds pool (5), or total exceeds storage capacity (10).

    The pool of each type has at most 5 molecules in circulation; if a single sample
    needs more than 5 of one type after expertise, it can never be held in storage
    in sufficient quantity to produce. Drop such samples instead of hoarding.
    """
    for i in range(5):
        if s['cost'][i] - me['expertise'][i] > MOL_POOL_CAP:
            return True
    return total_need_after_exp(s, me) > STORAGE_CAP


def choose_rank(me):
    total_exp = sum(me['expertise'])
    if total_exp < 3:
        return 1
    if total_exp < 7:
        return 2
    return 3


def project_value_of_gain(gain, me, projects):
    """Value of gaining 1 expertise in `gain` type, summed across science projects.

    For each project that (a) we still need this type for and (b) is not already
    complete, add 50 / remaining_total_expertise_needed. Projects close to done
    weigh more — a gain that finishes a project is worth the full 50.
    """
    if gain not in TYPES:
        return 0.0
    gi = TYPES.index(gain)
    value = 0.0
    for proj in projects:
        if me['expertise'][gi] >= proj[gi]:
            continue
        rem = sum(max(0, proj[i] - me['expertise'][i]) for i in range(5))
        if rem <= 0:
            continue
        value += 50.0 / rem
    return value


def priority_score(s, me, projects):
    """Higher is better. Combines sample health with project advancement value."""
    return s['health'] + project_value_of_gain(s.get('gain', '0'), me, projects)


def pick_molecule_index(me, sorted_feasible, available):
    """Decide which molecule index to CONNECT for, considering multi-sample reservation.

    For each sample in priority order, reserve its post-expertise cost from storage.
    Pick the first molecule type that the current sample needs and is available.
    """
    if sum(me['storage']) >= STORAGE_CAP:
        return None
    reserved = [0] * 5
    for s in sorted_feasible:
        effective_storage = [max(0, me['storage'][i] - reserved[i]) for i in range(5)]
        need = [max(0, s['cost'][i] - me['expertise'][i] - effective_storage[i]) for i in range(5)]
        for i in range(5):
            if need[i] > 0 and available[i] > 0:
                return i
        # Sample is fully covered. Reserve its post-expertise cost.
        for i in range(5):
            reserved[i] += max(0, s['cost'][i] - me['expertise'][i])
    return None


def decide(me, opp, available, samples, projects):
    if me['eta'] > 0:
        return "WAIT|moving"

    my_samples = [s for s in samples if s['carried_by'] == 0]
    cloud_samples = [s for s in samples if s['carried_by'] == -1]
    diagnosed = [s for s in my_samples if is_diagnosed(s)]
    undiagnosed = [s for s in my_samples if not is_diagnosed(s)]
    n_samples = len(my_samples)

    feasible_diag = [s for s in diagnosed if not is_impossible(s, me)]
    impossible_diag = [s for s in diagnosed if is_impossible(s, me)]
    sorted_feasible = sorted(feasible_diag, key=lambda s: -priority_score(s, me, projects))
    completable_now = [s for s in feasible_diag if can_complete_now(s, me)]

    # Useful cloud samples — diagnosed cloud samples we could feasibly fund
    cloud_useful = [s for s in cloud_samples if not is_impossible(s, me)]
    cloud_useful.sort(key=lambda s: -priority_score(s, me, projects))
    best_cloud_score = priority_score(cloud_useful[0], me, projects) if cloud_useful else 0

    # 1. At LAB and have a producible sample -> produce highest-priority one
    if me['target'] == 'LABORATORY' and completable_now:
        target = max(completable_now, key=lambda s: priority_score(s, me, projects))
        return f"CONNECT {target['id']}|produce {target['id']} h={target['health']} g={target.get('gain','?')} pri={priority_score(target, me, projects):.0f}"

    # 2. At SAMPLES and inventory not full -> grab another
    if me['target'] == 'SAMPLES' and n_samples < 3:
        rank = choose_rank(me)
        return f"CONNECT {rank}|pick r{rank} ({n_samples}/3 exp={sum(me['expertise'])})"

    # 3. At DIAGNOSIS and have undiagnosed -> diagnose one
    if me['target'] == 'DIAGNOSIS' and undiagnosed:
        return f"CONNECT {undiagnosed[0]['id']}|diagnose {undiagnosed[0]['id']}"

    # 3a. At DIAGNOSIS with n<3 and a useful cloud sample -> pull (cheaper than SAMPLES)
    if me['target'] == 'DIAGNOSIS' and n_samples < 3 and cloud_useful:
        best = cloud_useful[0]
        return f"CONNECT {best['id']}|pull cloud {best['id']} h={best['health']} g={best.get('gain','?')} pri={best_cloud_score:.0f}"

    # 3b. At DIAGNOSIS with impossible diagnosed (and no undiagnosed) -> dump
    if me['target'] == 'DIAGNOSIS' and impossible_diag and not undiagnosed:
        return f"CONNECT {impossible_diag[0]['id']}|drop impossible {impossible_diag[0]['id']}"

    # 4. At MOLECULES -> opportunistically pick up a useful molecule
    if me['target'] == 'MOLECULES' and sorted_feasible:
        idx = pick_molecule_index(me, sorted_feasible, available)
        if idx is not None:
            return f"CONNECT {TYPES[idx]}|pickup {TYPES[idx]} for sorted={[s['id'] for s in sorted_feasible]}"

    # 5. Decide where to go next

    # 5a. Producible sample ready -> head to LAB
    if completable_now:
        return f"GOTO LABORATORY|produce ready={[s['id'] for s in completable_now]}"

    # 5b. Inventory not full -> SAMPLES or DIAGNOSIS (whichever yields a better sample)
    if n_samples < 3:
        # A high-priority cloud sample beats a random rank pick. Threshold 25 ≈ a
        # mid-rank-2 sample with no project value, so anything notably better detours.
        if cloud_useful and best_cloud_score > 25:
            return f"GOTO DIAGNOSIS|cloud has pri={best_cloud_score:.0f} {cloud_useful[0]['id']}"
        return f"GOTO SAMPLES|need {3 - n_samples} more"

    # 5c. Have undiagnosed -> DIAGNOSIS
    if undiagnosed:
        return f"GOTO DIAGNOSIS|diagnose {len(undiagnosed)}"

    # 5d. Have feasible diagnosed needing mols -> MOLECULES (or wait if already there)
    if feasible_diag:
        if me['target'] == 'MOLECULES':
            return "WAIT|wait for mols to refill"
        return "GOTO MOLECULES|fund samples"

    # 5e. All diagnosed are impossible -> DIAGNOSIS to dump
    if impossible_diag:
        return "GOTO DIAGNOSIS|drop impossibles"

    # Fallback
    return "WAIT|fallback"


def main():
    project_count = int(input())
    projects = []
    for _ in range(project_count):
        projects.append([int(x) for x in input().split()])
    log(f"projects: {projects}")

    turn = 0
    while True:
        turn += 1
        me = read_player()
        opp = read_player()
        available = [int(x) for x in input().split()]
        sample_count = int(input())
        samples = [read_sample() for _ in range(sample_count)]

        action = decide(me, opp, available, samples, projects)

        if '|' in action:
            cmd, reason = action.split('|', 1)
            cmd = cmd.strip()
        else:
            cmd, reason = action, ''

        log(
            f"t{turn} score={me['score']}/{opp['score']} "
            f"tgt={me['target']} eta={me['eta']} "
            f"stor={me['storage']} exp={me['expertise']} avail={available} "
            f"mine={[(s['id'], s['rank'], 'D' if is_diagnosed(s) else 'U', s['gain'], s['health'], s['cost']) for s in samples if s['carried_by']==0]} "
            f"cloud={[(s['id'], s['rank'], 'D' if is_diagnosed(s) else 'U', s['gain'], s['health']) for s in samples if s['carried_by']==-1]} "
            f"-> {cmd} ({reason})"
        )
        print(cmd)


if __name__ == '__main__':
    main()
