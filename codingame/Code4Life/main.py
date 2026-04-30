import sys


def debug(msg):
    print(f"DEBUG: {msg}", file=sys.stderr, flush=True)


# --- INITIALIZATION INPUT ---
project_count = int(input())
for i in range(project_count):
    # Ignoring science projects for Version 1
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
    input()  # Skipping enemy data for Version 1

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

    # Identify undiagnosed samples (usually represented by negative costs or health)
    undiagnosed = [s for s in my_samples if s['costs']['A'] < 0 or s['health'] < 0]
    diagnosed = [s for s in my_samples if s['costs']['A'] >= 0 and s['health'] >= 0]


    def get_missing_molecules(sample):
        """Returns a dict of molecules needed to craft this sample."""
        missing = {}
        for mol in 'ABCDE':
            needed_total = max(0, sample['costs'][mol] - my_expertise[mol])
            lack = max(0, needed_total - my_storage[mol])
            if lack > 0:
                missing[mol] = lack
        return missing


    def get_total_real_cost(sample):
        """Returns the total number of molecules required after expertise."""
        return sum(max(0, sample['costs'][mol] - my_expertise[mol]) for mol in 'ABCDE')


    action = "WAIT"

    if my_eta > 0:
        # We are currently moving, any command is ignored, so we just wait.
        debug("Moving, ETA: " + str(my_eta))
        action = "WAIT"
    else:
        # Priority 1: Do we have any fully prepared samples to produce?
        ready_samples = [s for s in diagnosed if sum(get_missing_molecules(s).values()) == 0]

        # Priority 2: Do we have impossible samples to drop?
        impossible_samples = [s for s in diagnosed if get_total_real_cost(s) > 10]

        if ready_samples:
            target_sample = ready_samples[0]
            if my_target != 'LABORATORY':
                action = "GOTO LABORATORY"
                debug("Going to lab to craft.")
            else:
                action = f"CONNECT {target_sample['id']}"
                debug(f"Crafting sample {target_sample['id']}")

        elif impossible_samples:
            target_sample = impossible_samples[0]
            if my_target != 'DIAGNOSIS':
                action = "GOTO DIAGNOSIS"
                debug("Going to diag to drop impossible sample.")
            else:
                action = f"CONNECT {target_sample['id']}"
                debug(f"Dropping sample {target_sample['id']} to cloud.")

        elif undiagnosed:
            target_sample = undiagnosed[0]
            if my_target != 'DIAGNOSIS':
                action = "GOTO DIAGNOSIS"
                debug("Going to diag to diagnose sample.")
            else:
                action = f"CONNECT {target_sample['id']}"
                debug(f"Diagnosing sample {target_sample['id']}")

        elif diagnosed and sum(my_storage.values()) < 10:
            # We need molecules, find what's missing and available
            target_mol = None
            for s in diagnosed:
                missing = get_missing_molecules(s)
                for mol, amount in missing.items():
                    if available[mol] > 0:
                        target_mol = mol
                        break
                if target_mol:
                    break

            if target_mol:
                if my_target != 'MOLECULES':
                    action = "GOTO MOLECULES"
                    debug(f"Going to gather molecule {target_mol}.")
                else:
                    action = f"CONNECT {target_mol}"
                    debug(f"Gathering molecule {target_mol}.")
            else:
                # We need molecules, but the ones we need are currently unavailable
                if len(my_samples) < 3:
                    if my_target != 'SAMPLES':
                        action = "GOTO SAMPLES"
                    else:
                        action = "CONNECT 2"
                        debug("Missing molecules unavailable. Getting another rank 2 sample.")
                else:
                    action = "WAIT"
                    debug("Stuck! Waiting for molecules to become available.")

        elif len(my_samples) < 3:
            if my_target != 'SAMPLES':
                action = "GOTO SAMPLES"
                debug("Going to get initial samples.")
            else:
                action = "CONNECT 2"
                debug("Grabbing a rank 2 sample.")
        else:
            action = "WAIT"
            debug("Idle state reached.")

    # Execute action
    print(action)