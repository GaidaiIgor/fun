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

while True:
    players = []
    for _ in range(2):
        inputs = input().split()
        players.append({
            "target": inputs[0],
            "eta": int(inputs[1]),
            "score": int(inputs[2]),
            "storage": list(map(int, inputs[3:8])),
            "expertise": list(map(int, inputs[8:13]))
        })

    me = players[0]
    total_expertise = sum(me["expertise"])

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

    # ---- PHASE SELECTION ----
    if total_expertise < 6:
        target_rank = 1
    elif total_expertise < 12:
        target_rank = 2
    else:
        target_rank = 2  # keep stable for now

    # 1. Get samples
    if len(my_samples) < 3:
        if me["target"] != "SAMPLES":
            print("GOTO SAMPLES")
        else:
            print(f"CONNECT {target_rank}")
        continue

    # 2. Diagnose
    if undiagnosed:
        if me["target"] != "DIAGNOSIS":
            print("GOTO DIAGNOSIS")
        else:
            print(f"CONNECT {undiagnosed[0]['id']}")
        continue

    # 3. Produce ASAP
    for s in diagnosed:
        if can_complete(s, me["storage"], me["expertise"]):
            if me["target"] != "LABORATORY":
                print("GOTO LABORATORY")
            else:
                print(f"CONNECT {s['id']}")
            break
    else:
        # 4. Filter feasible
        feasible = [
            s for s in diagnosed
            if is_possible(s, me["storage"], me["expertise"], available)
        ]

        if not feasible:
            if me["target"] != "DIAGNOSIS":
                print("GOTO DIAGNOSIS")
            else:
                worst = max(diagnosed, key=lambda s: sum(s["cost"]))
                print(f"CONNECT {worst['id']}")
            continue

        # prioritize lowest remaining cost
        target = min(feasible, key=lambda s: remaining_cost(s, me["storage"], me["expertise"]))

        # 5. Go to molecules
        if me["target"] != "MOLECULES":
            print("GOTO MOLECULES")
            continue

        need = total_needed(target, me["expertise"])

        best_type = None
        best_missing = -1

        for i in range(5):
            missing = max(0, need[i] - me["storage"][i])
            if missing > 0 and available[i] > 0:
                if missing > best_missing:
                    best_missing = missing
                    best_type = i

        if best_type is not None and sum(me["storage"]) < 10:
            print("CONNECT " + "ABCDE"[best_type])
        else:
            print("WAIT")