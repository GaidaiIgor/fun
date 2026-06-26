import sys

TYPES = ['A', 'B', 'C', 'D', 'E']


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
    """Sample cost (with expertise reduction) exceeds storage capacity (10)."""
    return total_need_after_exp(s, me) > 10


def choose_rank(me):
    total_exp = sum(me['expertise'])
    if total_exp < 3:
        return 1
    if total_exp < 7:
        return 2
    return 3


def pick_molecule_index(me, sorted_feasible, available):
    """Decide which molecule index to CONNECT for, considering multi-sample reservation.

    For each sample in priority order, reserve its post-expertise cost from storage.
    Pick the first molecule type that the current sample needs and is available.
    """
    if sum(me['storage']) >= 10:
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
    diagnosed = [s for s in my_samples if is_diagnosed(s)]
    undiagnosed = [s for s in my_samples if not is_diagnosed(s)]
    n_samples = len(my_samples)

    feasible_diag = [s for s in diagnosed if not is_impossible(s, me)]
    impossible_diag = [s for s in diagnosed if is_impossible(s, me)]
    sorted_feasible = sorted(feasible_diag, key=lambda s: -s['health'])
    completable_now = [s for s in feasible_diag if can_complete_now(s, me)]

    # 1. At LAB and have a producible sample -> produce highest health
    if me['target'] == 'LABORATORY' and completable_now:
        target = max(completable_now, key=lambda s: s['health'])
        return f"CONNECT {target['id']}|produce {target['id']} h={target['health']}"

    # 2. At SAMPLES and inventory not full -> grab another
    if me['target'] == 'SAMPLES' and n_samples < 3:
        rank = choose_rank(me)
        return f"CONNECT {rank}|pick r{rank} ({n_samples}/3 exp={sum(me['expertise'])})"

    # 3. At DIAGNOSIS and have undiagnosed -> diagnose one
    if me['target'] == 'DIAGNOSIS' and undiagnosed:
        return f"CONNECT {undiagnosed[0]['id']}|diagnose {undiagnosed[0]['id']}"

    # 3b. At DIAGNOSIS with impossible diagnosed samples (and no undiagnosed) -> dump
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

    # 5b. Inventory not full -> SAMPLES (always batch to 3 before progressing)
    if n_samples < 3:
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
            f"mine={[(s['id'], s['rank'], 'D' if is_diagnosed(s) else 'U', s['health'], s['cost']) for s in samples if s['carried_by']==0]} "
            f"cloud={[s['id'] for s in samples if s['carried_by']==-1]} "
            f"-> {cmd} ({reason})"
        )
        print(cmd)


if __name__ == '__main__':
    main()
