import sys


def debug(msg):
    print(f"DEBUG: {msg}", file=sys.stderr, flush=True)


# --- INITIALIZATION INPUT ---
project_count = int(input())
for i in range(project_count):
    # Ignoring science projects for Version 3
    input()

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
    input()

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
        sample = {
            'id': int(inputs[0]),
            'carried_by': int(inputs[1]),
            'rank': int(inputs[2]),
            'gain': inputs[3],
            'health': int(inputs[4]),
            'costs': {
                'A': int(inputs[5]), 'B': int(inputs[6]),
                'C': int(inputs[7]), 'D': int(inputs[8]), 'E': int(inputs[9])
            }
        }
        samples.append(sample)

    # --- BOT LOGIC ---
    my_samples = [s for s in samples if s['carried_by'] == 0]
    undiagnosed = [s for s in my_samples if s['health'] < 0]
    diagnosed = [s for s in my_samples if s['health'] >= 0]

    # Calculate real costs for diagnosed samples (Cost minus Expertise)
    for s in diagnosed:
        s['real_cost'] = {}
        for mol in 'ABCDE':
            s['real_cost'][mol] = max(0, s['costs'][mol] - my_expertise[mol])

    # 1. Identify Ready Samples (We have all required molecules in storage)
    ready_samples = []
    for s in diagnosed:
        is_ready = True
        for mol in 'ABCDE':
            if my_storage[mol] < s['real_cost'][mol]:
                is_ready = False
                break
        if is_ready:
            ready_samples.append(s)

    # 2. Identify Impossible Samples (Cost > 10 slots)
    impossible_samples = [s for s in diagnosed if sum(s['real_cost'].values()) > 10]

    # 3. Determine Target Sample for Gathering
    target_gather_sample = None
    target_mol = None

    if not ready_samples and not impossible_samples and diagnosed:
        # First, try to find a sample we can completely finish right now without getting stuck
        for s in diagnosed:
            missing = {m: max(0, s['real_cost'][m] - my_storage[m]) for m in 'ABCDE'}
            can_finish = all(available[m] >= missing[m] for m in 'ABCDE')
            has_space = sum(my_storage.values()) + sum(missing.values()) <= 10

            if can_finish and has_space:
                target_gather_sample = s
                break

        # If no sample is perfectly available, strictly lock onto the first one to avoid deadlocks
        if not target_gather_sample:
            target_gather_sample = diagnosed[0]

        # Find the first missing molecule for the locked target sample that is available
        missing = {m: max(0, target_gather_sample['real_cost'][m] - my_storage[m]) for m in 'ABCDE'}
        for mol in 'ABCDE':
            if missing[mol] > 0 and available[mol] > 0:
                if sum(my_storage.values()) < 10:
                    target_mol = mol
                    break

    # --- ACTION DECISION ---
    action = "WAIT"

    if my_eta > 0:
        action = "WAIT"
        debug("Moving, ETA: " + str(my_eta))

    elif impossible_samples:
        target_sample = impossible_samples[0]
        if my_target != 'DIAGNOSIS':
            action = "GOTO DIAGNOSIS"
            debug("Going to diag to drop impossible sample.")
        else:
            action = f"CONNECT {target_sample['id']}"
            debug(f"Dropping impossible sample {target_sample['id']}")

    elif ready_samples:
        target_sample = ready_samples[0]
        if my_target != 'LABORATORY':
            action = "GOTO LABORATORY"
            debug("Going to lab to craft.")
        else:
            action = f"CONNECT {target_sample['id']}"
            debug(f"Crafting sample {target_sample['id']}")

    elif undiagnosed:
        # Batch samples: if at SAMPLES and have < 3, grab another before leaving
        if my_target == 'SAMPLES' and len(my_samples) < 3:
            action = "CONNECT 2"
            debug("At SAMPLES, grabbing another to batch.")
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
            debug("Going to get new samples.")
        else:
            action = "CONNECT 2"
            debug("Grabbing a rank 2 sample.")

    else:
        action = "WAIT"
        debug("Idle - target molecules unavailable. Waiting.")

    # Execute action
    print(action)