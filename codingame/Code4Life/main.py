import sys


def debug(msg):
    print(f"DEBUG: {msg}", file=sys.stderr, flush=True)


# --- INITIALIZATION INPUT ---
project_count = int(input())
for i in range(project_count):
    # Ignoring science projects for Version 2
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

    # 1. Simulate allocation and expertise
    simulated_expertise = my_expertise.copy()
    allocated_storage = my_storage.copy()
    missing_per_sample = {}

    for s in diagnosed:
        missing = {}
        for mol in 'ABCDE':
            needed = max(0, s['costs'][mol] - simulated_expertise[mol])
            available_for_s = min(needed, allocated_storage[mol])
            lack = needed - available_for_s
            if lack > 0:
                missing[mol] = lack
            allocated_storage[mol] -= available_for_s

        missing_per_sample[s['id']] = missing

        # Simulate gaining expertise ONLY if this sample is fully ready right now.
        # This prevents locking ourselves out if a prior sample's molecules are unavailable.
        if sum(missing.values()) == 0:
            if s['gain'] in simulated_expertise:
                simulated_expertise[s['gain']] += 1

    # 2. Identify ready samples
    ready_samples = [s for s in diagnosed if sum(missing_per_sample[s['id']].values()) == 0]

    # 3. Identify impossible samples (cost > 10)
    impossible_samples = []
    for s in diagnosed:
        real_cost = sum(max(0, s['costs'][mol] - my_expertise[mol]) for mol in 'ABCDE')
        if real_cost > 10:
            impossible_samples.append(s)

    # 4. Find target molecule
    target_mol = None
    for s in diagnosed:
        # Ignore impossible samples when gathering
        if s in impossible_samples:
            continue
        missing = missing_per_sample[s['id']]
        for mol, amount in missing.items():
            if amount > 0 and available[mol] > 0 and sum(my_storage.values()) < 10:
                target_mol = mol
                break
        if target_mol:
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

    elif undiagnosed:
        # Batch samples! If we are already at SAMPLES, stay and grab up to 3.
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

    elif ready_samples:
        # Batch gathering! If we have a ready sample but can still gather for others, stay!
        if my_target == 'MOLECULES' and target_mol is not None:
            action = f"CONNECT {target_mol}"
            debug(f"Sample ready, but staying to batch gather {target_mol}.")
        else:
            target_sample = ready_samples[0]
            if my_target != 'LABORATORY':
                action = "GOTO LABORATORY"
                debug("Going to lab to craft.")
            else:
                action = f"CONNECT {target_sample['id']}"
                debug(f"Crafting sample {target_sample['id']}")

    elif target_mol is not None:
        if my_target != 'MOLECULES':
            action = "GOTO MOLECULES"
            debug(f"Going to gather molecule {target_mol}.")
        else:
            action = f"CONNECT {target_mol}"
            debug(f"Gathering molecule {target_mol}.")

    elif len(my_samples) < 3:
        if my_target != 'SAMPLES':
            action = "GOTO SAMPLES"
            debug("Going to get new samples.")
        else:
            action = "CONNECT 2"
            debug("Grabbing a rank 2 sample.")

    else:
        action = "WAIT"
        debug("Idle - waiting for enemy to release molecules.")

    # Execute action
    print(action)