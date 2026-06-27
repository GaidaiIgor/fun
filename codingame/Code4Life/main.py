import sys

def debug(*args):
    print(*args, file=sys.stderr)

project_count = int(input())
for _ in range(project_count):
    input()

def total_needed(sample, expertise):
    return [
        max(0, sample["cost"][i] - expertise[i])
        for i in range(5)
    ]

def can_complete(sample, storage, expertise):
    need = total_needed(sample, expertise)
    return all(storage[i] >= need[i] for i in range(5))

def is_possible(sample, storage, expertise, available):
    need = total_needed(sample, expertise)
    return all(storage[i] + available[i] >= need[i] for i in range(5))

def remaining_cost(sample, storage, expertise):
    need = total_needed(sample, expertise)
    return sum(max(0, need[i] - storage[i]) for i in range(5))

def pick_target(diagnosed, storage, expertise, available):
    feasible = [
        s for s in diagnosed
        if is_possible(s, storage, expertise, available)
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda s: remaining_cost(s, storage, expertise))

while True:
    players = []
    for _ in range(2):
        inputs = input().split()
        target = inputs[0]
        eta = int(inputs[1])
        score = int(inputs[2])
        storage = list(map(int, inputs[3:8]))
        expertise = list(map(int, inputs[8:13]))
        players.append({
            "target": target,
            "eta": eta,
            "score": score,
            "storage": storage,
            "expertise": expertise
        })

    me = players[0]

    available = list(map(int, input().split()))
    sample_count = int(input())

    samples = []
    for _ in range(sample_count):
        inputs = input().split()
        samples.append({
            "id": int(inputs[0]),
            "carried_by": int(inputs[1]),
            "rank": int(inputs[2]),
            "gain": inputs[3],
            "health": int(inputs[4]),
            "cost": list(map(int, inputs[5:10]))
        })

    if me["eta"] > 0:
        print("WAIT")
        continue

    my_samples = [s for s in samples if s["carried_by"] == 0]
    undiagnosed = [s for s in my_samples if s["cost"][0] == -1]
    diagnosed = [s for s in my_samples if s["cost"][0] != -1]

    # 1. Get samples
    if len(my_samples) < 3:
        if me["target"] != "SAMPLES":
            print("GOTO SAMPLES")
        else:
            print("CONNECT 2")
        continue

    # 2. Diagnose
    if undiagnosed:
        if me["target"] != "DIAGNOSIS":
            print("GOTO DIAGNOSIS")
        else:
            print(f"CONNECT {undiagnosed[0]['id']}")
        continue

    # 3. Produce if possible
    doable = [
        s for s in diagnosed
        if can_complete(s, me["storage"], me["expertise"])
    ]

    if doable:
        if me["target"] != "LABORATORY":
            print("GOTO LABORATORY")
        else:
            print(f"CONNECT {doable[0]['id']}")
        continue

    # 4. Pick target sample
    target_sample = pick_target(diagnosed, me["storage"], me["expertise"], available)

    # 5. Drop only if NOTHING is possible
    if target_sample is None:
        if me["target"] != "DIAGNOSIS":
            print("GOTO DIAGNOSIS")
        else:
            worst = max(diagnosed, key=lambda s: sum(s["cost"]))
            print(f"CONNECT {worst['id']}")
        continue

    # 6. Go to molecules
    if me["target"] != "MOLECULES":
        print("GOTO MOLECULES")
        continue

    # 7. Collect only what target needs
    need = total_needed(target_sample, me["expertise"])

    best_type = None
    best_need = -1

    for i in range(5):
        missing = max(0, need[i] - me["storage"][i])
        if missing > 0 and available[i] > 0:
            if missing > best_need:
                best_need = missing
                best_type = i

    if best_type is not None and sum(me["storage"]) < 10:
        print("CONNECT " + "ABCDE"[best_type])
    else:
        print("WAIT")