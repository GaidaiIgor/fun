import sys


def debug(msg):
    print(f"DEBUG: {msg}", file=sys.stderr, flush=True)


# --- INITIALIZATION INPUT ---
project_count = int(input())
projects = []
for i in range(project_count):
    inputs = input().split()
    projects.append({
        'A': int(inputs[0]), 'B': int(inputs[1]),
        'C': int(inputs[2]), 'D': int(inputs[3]), 'E': int(inputs[4])
    })

# --- GAME LOOP ---
while True:
    # 1. READ PLAYER DATA (Us)
    inputs = input().split()
    my_target = inputs[0]
    my_eta = int(inputs[1])
    my_score = int(inputs[2])
    my_storage = {
        'A': int(inputs[3]), 'B': int(inputs[4]),
        'C': int(inputs[5]), 'D': int(inputs[6]), 'E': int(inputs[7])
    }
    my_expertise = {
        'A': int(inputs[8]), 'B': int(inputs[9]),
        'C': int(inputs[10]), 'D': int(inputs[11]), 'E': int(inputs[12])
    }

    # 2. READ ENEMY DATA
    enemy_inputs = input().split()
    # We read but ignore enemy data to maintain focus on our own strict efficiency

    # 3. READ AVAILABLE MOLECULES
    inputs = input().split()
    available = {
        'A': int(inputs[0]), 'B': int(inputs[1]),
        'C': int(inputs[2]), 'D': int(inputs[3]), 'E': int(inputs[4])
    }

    # 4. READ SAMPLES
    sample_count = int(input())
    samples = []
    for i in range(sample_count):
        inputs = input().split()
        samples.append({
            'id': int(inputs[0]),
            'carried_by': int(inputs[1]),
            'rank': int(inputs[2]),
            'gain': inputs[3],
            'health': int(inputs[4]),
            'costs': {
                'A': int(inputs[5]), 'B': int(inputs[6]),
                'C': int(inputs[7]), 'D': int(inputs[8]), 'E': int(inputs[9])
            }
        })

    # --- BOT LOGIC ---
    my_samples = [s for s in samples if s['carried_by'] == 0]
    undiagnosed = [s for s in my_samples if s['health'] < 0]
    diagnosed = [s for s in my_samples if s['health'] >= 0]

    # Calculate real costs for diagnosed samples (Cost minus Expertise)
    for s in diagnosed:
        s['real_cost'] = {mol: max(0, s['costs'][mol] - my_expertise[mol]) for mol in 'ABCDE'}

    # 1. Identify Ready Samples (We have all required molecules in storage)
    ready_samples = []
    for s in diagnosed:
        if all(my_storage[m] >= s['real_cost'][m] for m in 'ABCDE'):
            ready_samples.append(s)

    # 2. Identify Impossible Samples (Cost > 10 slots)
    impossible_samples = [s for s in diagnosed if sum(s['real_cost'].values()) > 10]
    valid_diagnosed = [s for s in diagnosed if s not in impossible_samples]

    # 3. Science Project Synergy
    valuable_expertise = set()
    for p in projects:
        # Check if this project is still incomplete by us
        if any(my_expertise[m] < p[m] for m in 'ABCDE'):
            for m in 'ABCDE':
                if my_expertise[m] < p[m]:
                    valuable_expertise.add(m)


    def sample_priority(s):
        # We want to minimize missing cost, but give a "discount" to samples that fulfill projects
        missing_cost = sum(max(0, s['real_cost'][m] - my_storage[m]) for m in 'ABCDE')
        bonus = 2 if s['gain'] in valuable_expertise else 0
        return (missing_cost - bonus, -s['health'])


    valid_diagnosed.sort(key=sample_priority)

    # 4. Determine Target Sample & Molecule
    target_gather_sample = None
    target_mol = None

    # Pass 1: Look for a perfect finish
    for s in valid_diagnosed:
        missing = {m: max(0, s['real_cost'][m] - my_storage[m]) for m in 'ABCDE'}
        can_finish = all(available[m] >= missing[m] for m in 'ABCDE')
        has_space = sum(my_storage.values()) + sum(missing.values()) <= 10

        if can_finish and has_space:
            target_gather_sample = s
            for m in 'ABCDE':
                if missing[m] > 0:
                    target_mol = m
                    break
            break

    # Pass 2: If no perfect finish, scan all valid samples for ANY progress
    if not target_gather_sample:
        for s in valid_diagnosed:
            missing = {m: max(0, s['real_cost'][m] - my_storage[m]) for m in 'ABCDE'}
            # Can we make progress on this sample without overflowing inventory?
            if any(missing[m] > 0 and available[m] > 0 for m in 'ABCDE') and sum(my_storage.values()) < 10:
                target_gather_sample = s
                for m in 'ABCDE':
                    if missing[m] > 0 and available[m] > 0:
                        target_mol = m
                        break
                break

    # Anti-Deadlock Recovery
    if not target_mol and not ready_samples and not impossible_samples and len(my_samples) == 3 and valid_diagnosed:
        # We are full, but can't gather a single molecule for ANY sample. The enemy is starving us.
        # Force the hardest sample into the impossible list to dump it to the cloud.
        hardest = max(valid_diagnosed, key=lambda s: sum(max(0, s['real_cost'][m] - my_storage[m]) for m in 'ABCDE'))
        impossible_samples.append(hardest)
        debug(f"Deadlock detected! Forcing dump of sample {hardest['id']}")

    # 5. Dynamic Rank Scaling (Adjusted for Science Project Rush)
    total_exp = sum(my_expertise.values())
    if total_exp < 5:
        target_rank = 1  # Fast early game points/projects
    elif total_exp < 10:
        target_rank = 2  # Mid game scaling
    else:
        target_rank = 3  # Late game bombs

    # --- ACTION DECISION TREE ---
    action = "WAIT"

    if my_eta > 0:
        action = "WAIT"
        debug(f"Moving, ETA: {my_eta}")

    elif impossible_samples:
        target_sample = impossible_samples[0]
        if my_target != 'DIAGNOSIS':
            action = "GOTO DIAGNOSIS"
            debug("Going to dump impossible/deadlocked sample.")
        else:
            action = f"CONNECT {target_sample['id']}"
            debug(f"Dumping sample {target_sample['id']} to cloud.")

    elif ready_samples:
        target_sample = ready_samples[0]
        if my_target != 'LABORATORY':
            action = "GOTO LABORATORY"
            debug("Going to lab to craft.")
        else:
            action = f"CONNECT {target_sample['id']}"
            debug(f"Crafting sample {target_sample['id']}")

    elif undiagnosed:
        if my_target == 'SAMPLES' and len(my_samples) < 3:
            action = f"CONNECT {target_rank}"
            debug(f"At SAMPLES, batching another Rank {target_rank}.")
        else:
            if my_target != 'DIAGNOSIS':
                action = "GOTO DIAGNOSIS"
                debug("Going to diag to diagnose samples.")
            else:
                action = f"CONNECT {undiagnosed[0]['id']}"
                debug(f"Diagnosing sample {undiagnosed[0]['id']}")

    elif target_mol:
        if my_target != 'MOLECULES':
            action = "GOTO MOLECULES"
            debug(f"Going to gather molecule {target_mol}.")
        else:
            action = f"CONNECT {target_mol}"
            debug(f"Gathering molecule {target_mol} for sample {target_gather_sample['id']}.")

    elif len(my_samples) < 3:
        if my_target != 'SAMPLES':
            action = "GOTO SAMPLES"
            debug("Going to get new samples to cycle inventory.")
        else:
            action = f"CONNECT {target_rank}"
            debug(f"Grabbing a rank {target_rank} sample.")

    else:
        action = "WAIT"
        debug("Idle.")

    # Execute action
    print(action)