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
    # Undiagnosed samples have negative cost fields.
    return s['cost'][0] >= 0


def needed_molecules(s, me):
    return [max(0, s['cost'][i] - me['expertise'][i] - me['storage'][i]) for i in range(5)]


def can_complete_now(s, me):
    return sum(needed_molecules(s, me)) == 0


def can_complete_eventually(s, me, available):
    need = needed_molecules(s, me)
    if sum(me['storage']) + sum(need) > 10:
        return False
    for i in range(5):
        if need[i] > available[i]:
            return False
    return True


def fits_in_storage(s, me):
    """Could this sample fit if we picked up all needed molecules (ignoring availability)?"""
    need = needed_molecules(s, me)
    return sum(me['storage']) + sum(need) <= 10


def choose_rank(me):
    total_exp = sum(me['expertise'])
    if total_exp < 3:
        return 1
    if total_exp < 8:
        return 2
    return 3


def priority_sort(samples, me, available):
    """Order samples we own by usefulness: completable-now first, then by health/cost ratio."""
    def key(s):
        n = needed_molecules(s, me)
        total_need = sum(n)
        now = total_need == 0
        ratio = s['health'] / (total_need + 1)
        # higher health & lower remaining need first; completable_now is the strongest tiebreaker
        return (-int(now), -s['health'], -ratio, total_need)
    return sorted(samples, key=key)


def next_molecule_for(s, me, available):
    """Return index of molecule type to pick up next for this sample, or None."""
    need = needed_molecules(s, me)
    for i in range(5):
        if need[i] > 0 and available[i] > 0:
            return i
    return None


def decide(me, opp, available, samples, projects):
    if me['eta'] > 0:
        return "WAIT|moving"

    my_samples = [s for s in samples if s['carried_by'] == 0]
    diagnosed = [s for s in my_samples if is_diagnosed(s)]
    undiagnosed = [s for s in my_samples if not is_diagnosed(s)]

    # Split diagnosed by feasibility
    completable_eventually = [s for s in diagnosed if can_complete_eventually(s, me, available)]
    completable_now = [s for s in diagnosed if can_complete_now(s, me)]
    impossible = [s for s in diagnosed
                  if not fits_in_storage(s, me)]  # cost > capacity even with expertise

    sorted_diag = priority_sort(completable_eventually, me, available)

    # 1. At LAB and a completable sample exists -> produce highest-health one
    if completable_now:
        target = max(completable_now, key=lambda s: s['health'])
        if me['target'] == 'LABORATORY':
            return f"CONNECT {target['id']}|produce {target['id']} h={target['health']}"
        else:
            return f"GOTO LABORATORY|head to lab for {target['id']}"

    # 2. Have a still-feasible diagnosed sample -> go get molecules
    if sorted_diag:
        focus = sorted_diag[0]
        idx = next_molecule_for(focus, me, available)
        if idx is not None:
            if me['target'] == 'MOLECULES':
                return f"CONNECT {TYPES[idx]}|pickup for sample {focus['id']}"
            else:
                return f"GOTO MOLECULES|need mols for {focus['id']} need={needed_molecules(focus, me)}"
        # focus needs nothing or no available next mol -> fall through; if completable now we'd have caught it above

    # 3. Undiagnosed samples -> diagnose them
    if undiagnosed:
        if me['target'] == 'DIAGNOSIS':
            return f"CONNECT {undiagnosed[0]['id']}|diagnose {undiagnosed[0]['id']}"
        else:
            return f"GOTO DIAGNOSIS|to diagnose {len(undiagnosed)} sample(s)"

    # 4. Dump impossible diagnosed samples back to cloud
    if impossible:
        if me['target'] == 'DIAGNOSIS':
            return f"CONNECT {impossible[0]['id']}|drop impossible {impossible[0]['id']}"
        else:
            return f"GOTO DIAGNOSIS|drop impossible {impossible[0]['id']}"

    # 5. Less than 3 samples -> pick more
    if len(my_samples) < 3:
        if me['target'] == 'SAMPLES':
            rank = choose_rank(me)
            return f"CONNECT {rank}|pick rank {rank} (exp_sum={sum(me['expertise'])})"
        else:
            return f"GOTO SAMPLES|need more samples ({len(my_samples)}/3)"

    # 6. Stuck: have 3 diagnosed samples but waiting for molecules. Wait at MOLECULES.
    if diagnosed and not sorted_diag:
        # All our diagnosed samples are blocked by global availability — wait, opponent may release mols.
        if me['target'] == 'MOLECULES':
            return "WAIT|blocked, wait for molecules"
        else:
            return "GOTO MOLECULES|wait at mol module"

    return "WAIT|fallback"


def main():
    project_count = int(input())
    projects = []
    for _ in range(project_count):
        projects.append([int(x) for x in input().split()])
    log(f"projects: {projects}")

    while True:
        me = read_player()
        opp = read_player()
        available = [int(x) for x in input().split()]
        sample_count = int(input())
        samples = [read_sample() for _ in range(sample_count)]

        action = decide(me, opp, available, samples, projects)

        # Split command from reason for stderr logging
        if '|' in action:
            cmd, reason = action.split('|', 1)
            cmd = cmd.strip()
        else:
            cmd, reason = action, ''

        log(
            f"t? me.score={me['score']} opp.score={opp['score']} "
            f"target={me['target']} eta={me['eta']} "
            f"stor={me['storage']} exp={me['expertise']} avail={available} "
            f"my_samples={[(s['id'], s['rank'], 'D' if is_diagnosed(s) else 'U', s['health'], s['cost']) for s in samples if s['carried_by']==0]} "
            f"cloud={[s['id'] for s in samples if s['carried_by']==-1]} "
            f"-> {cmd} ({reason})"
        )
        print(cmd)


if __name__ == '__main__':
    main()
