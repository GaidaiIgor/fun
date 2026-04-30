import sys
import itertools
from typing import Dict, List, Optional, Tuple

MODULES = ["START", "SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY"]

# Travel matrix (distance in turns)
DIST = {
    "START":     {"SAMPLES":2, "DIAGNOSIS":2, "MOLECULES":2, "LABORATORY":2},
    "SAMPLES":   {"DIAGNOSIS":3, "MOLECULES":3, "LABORATORY":3},
    "DIAGNOSIS": {"SAMPLES":3, "MOLECULES":3, "LABORATORY":4},
    "MOLECULES": {"SAMPLES":3, "DIAGNOSIS":3, "LABORATORY":3},
    "LABORATORY":{"SAMPLES":3, "DIAGNOSIS":4, "MOLECULES":3},
}

class Bot:
    def __init__(self):
        self.projects: List[Dict[str,int]] = []
        self.my_mol = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_exp = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_samples: List[Dict] = []          # each: id, diagnosed, rank, gain, health, cost dict
        self.cloud_samples: Dict[int, Dict] = {}  # id -> sample dict
        self.diagnosed_ids = set()
        self.next_rank = 1                        # 1, 2 or 3
        self.pending_sample_connect = False
        self.samples_before = 0
        self.store_mode = False                   # if True, we want to store a sample at DIAGNOSIS

    def parse_initial(self):
        project_count = int(input())
        for _ in range(project_count):
            a,b,c,d,e = map(int, input().split())
            self.projects.append({'A':a,'B':b,'C':c,'D':d,'E':e})

    def parse_turn(self):
        # first line: player data
        parts = input().split()
        target = parts[0]
        eta = int(parts[1])
        score = int(parts[2])
        self.my_mol['A'] = int(parts[3])
        self.my_mol['B'] = int(parts[4])
        self.my_mol['C'] = int(parts[5])
        self.my_mol['D'] = int(parts[6])
        self.my_mol['E'] = int(parts[7])
        self.my_exp['A'] = int(parts[8])
        self.my_exp['B'] = int(parts[9])
        self.my_exp['C'] = int(parts[10])
        self.my_exp['D'] = int(parts[11])
        self.my_exp['E'] = int(parts[12])

        # opponent line (ignore)
        input()

        # available molecules
        avail = list(map(int, input().split()))
        available = {'A':avail[0],'B':avail[1],'C':avail[2],'D':avail[3],'E':avail[4]}

        # samples
        sample_count = int(input())
        new_carried = []
        new_cloud = {}
        for _ in range(sample_count):
            sp = input().split()
            sid = int(sp[0])
            carried_by = int(sp[1])
            rank = int(sp[2])
            gain = sp[3]
            health = int(sp[4])
            costA = int(sp[5]); costB = int(sp[6]); costC = int(sp[7]); costD = int(sp[8]); costE = int(sp[9])
            s = {
                'id': sid,
                'rank': rank,
                'gain': gain,
                'health': health,
                'cost': {'A':costA,'B':costB,'C':costC,'D':costD,'E':costE},
            }
            if carried_by == 0:
                new_carried.append(s)
            elif carried_by == -1:
                new_cloud[sid] = s

        # update my samples, keeping diagnosed flag
        new_my_samples = []
        for s in new_carried:
            s['diagnosed'] = s['id'] in self.diagnosed_ids
            new_my_samples.append(s)
        self.my_samples = new_my_samples
        self.cloud_samples = new_cloud

        return target, eta, available

    def project_bonus(self, old_exp, new_exp):
        bonus = 0
        for proj in self.projects:
            old_ok = all(old_exp[t] >= proj[t] for t in 'ABCDE')
            new_ok = all(new_exp[t] >= proj[t] for t in 'ABCDE')
            if new_ok and not old_ok:
                bonus += 50
        return bonus

    def evaluate_sequence(self, sample_list, available):
        """Return (total_score, gather_plan_dict) or (None, None) if infeasible."""
        exp = dict(self.my_exp)
        inv = dict(self.my_mol)
        initial_sum = sum(inv.values())
        gather = {m:0 for m in 'ABCDE'}
        health = 0

        for s in sample_list:
            needed = {m: max(0, s['cost'][m] - exp[m]) for m in 'ABCDE'}
            # check if we need to gather
            for m in 'ABCDE':
                short = needed[m] - inv[m]
                if short > 0:
                    if gather[m] + short > available[m]:
                        return None, None
                    gather[m] += short
                    inv[m] += short
            # capacity check
            total = initial_sum + sum(gather.values())
            if total > 10:
                return None, None
            # consume
            for m in 'ABCDE':
                inv[m] -= needed[m]
            exp[s['gain']] += 1
            health += s['health']

        bonus = self.project_bonus(self.my_exp, exp)
        return health + bonus, gather

    def select_best_sequence(self, available):
        """Return (best_score, best_sequence_ids) using diagnosed carried samples."""
        diag = [s for s in self.my_samples if s['diagnosed']]
        if not diag:
            return 0, []
        best_score = -1
        best_seq = []
        # try subsets up to size 3
        max_take = min(len(diag), 3)
        for r in range(1, max_take+1):
            for subset in itertools.combinations(diag, r):
                # try all orders
                for perm in itertools.permutations(subset):
                    score, gather = self.evaluate_sequence(list(perm), available)
                    if score is not None and score > best_score:
                        best_score = score
                        best_seq = [s['id'] for s in perm]
                    # tie-break: prefer fewer total gather
                    elif score is not None and score == best_score and gather is not None:
                        old_score, old_gather = self.evaluate_sequence([s for s in diag if s['id'] in best_seq], available)  # not efficient but simple
                        if old_gather and sum(gather.values()) < sum(old_gather.values()):
                            best_seq = [s['id'] for s in perm]
        return best_score, best_seq

    def decide_action(self, target, eta, available):
        if eta > 0:
            print("WAIT")
            return

        current = target

        # Resolve pending sample connect
        if current == "SAMPLES" and self.pending_sample_connect:
            if len(self.my_samples) > self.samples_before:
                # success, keep rank 1 next
                self.next_rank = 1
            else:
                # failure, try next rank
                self.next_rank = self.next_rank + 1
                if self.next_rank > 3:
                    self.next_rank = 1
            self.pending_sample_connect = False

        if current == "SAMPLES":
            if len(self.my_samples) < 3:
                self.samples_before = len(self.my_samples)
                print(f"CONNECT {self.next_rank}")
                self.pending_sample_connect = True
            else:
                # full, go diagnose
                undiag = any(not s['diagnosed'] for s in self.my_samples)
                if undiag:
                    print("GOTO DIAGNOSIS")
                else:
                    # all diagnosed, try to produce
                    print("GOTO MOLECULES")

        elif current == "DIAGNOSIS":
            undiag = [s for s in self.my_samples if not s['diagnosed']]
            if undiag:
                s = undiag[0]
                print(f"CONNECT {s['id']}")
                self.diagnosed_ids.add(s['id'])
            else:
                # all diagnosed
                if self.store_mode:
                    # we came here to store a sample
                    diag = [s for s in self.my_samples if s['diagnosed']]
                    if diag:
                        # store the one with highest total cost
                        s = max(diag, key=lambda x: sum(x['cost'].values()))
                        print(f"CONNECT {s['id']}")
                        self.my_samples = [x for x in self.my_samples if x['id'] != s['id']]
                    self.store_mode = False
                    # after storing, go get new samples
                    print("GOTO SAMPLES")
                else:
                    # check feasibility
                    best_score, best_seq = self.select_best_sequence(available)
                    if best_seq:
                        # have a feasible plan, go gather molecules
                        print("GOTO MOLECULES")
                    else:
                        # no feasible plan, store the most expensive sample to free slot and try again
                        diag = [s for s in self.my_samples if s['diagnosed']]
                        if diag:
                            s = max(diag, key=lambda x: sum(x['cost'].values()))
                            print(f"CONNECT {s['id']}")
                            self.my_samples = [x for x in self.my_samples if x['id'] != s['id']]
                            # stay in DIAGNOSIS? We'll go to SAMPLES next turn
                            # but we already used CONNECT, so next turn we will re-evaluate.
                            # Set a flag to go to SAMPLES after?
                            # We'll simple: after this CONNECT (store), we can output next command same turn? No, we output one command per turn.
                            # So we just stored; next turn we will be still at DIAGNOSIS. We'll then go to SAMPLES.
                            # So set a flag or just rely on next turn logic: all diagnosed, no feasible plan -> store again? That would loop.
                            # Better: after storing, set a flag to go to SAMPLES.
                            self.store_mode = True   # but careful: store_mode triggers another store next turn. We'll redesign.
                            # I'll restructure: after storing, we want to go to SAMPLES. Since we can't GOTO in same turn,
                            # we'll let next turn start at DIAGNOSIS, but we need to GOTO SAMPLES. So we'll change logic:
                            # after all diagnosed and no feasible plan, we don't store immediately; we first GOTO SAMPLES to get more samples,
                            # but then we need to free a slot before going. So better: if no feasible plan and we have diagnosed samples,
                            # go to SAMPLES anyway? But we can't carry more than 3 samples. So we must store one to make room.
                            # Thus the correct sequence: store one (CONNECT id), then next turn we'll still be at DIAGNOSIS,
                            # we need to then GOTO SAMPLES. So we'll introduce a variable: after_store_go = "SAMPLES".
                            # So after storing, we set self.next_action_after_store = "SAMPLES". Then next turn at DIAGNOSIS, if there is no undiag and we have next_action_after_store, we print GOTO that destination and clear it.
                            # Let's implement that.
                            self.next_action_after_store = "SAMPLES"
                        else:
                            # no diagnosed samples, go get more
                            print("GOTO SAMPLES")
            # handle next_action_after_store
            if hasattr(self, 'next_action_after_store') and self.next_action_after_store:
                dest = self.next_action_after_store
                del self.next_action_after_store
                print(f"GOTO {dest}")
                return

        elif current == "MOLECULES":
            best_score, best_seq = self.select_best_sequence(available)
            if not best_seq:
                # no feasible plan, need to change samples
                # go to DIAGNOSIS to store expensive sample, then SAMPLES
                self.store_mode = True
                print("GOTO DIAGNOSIS")
                return

            # first sample in the plan
            first_id = best_seq[0]
            try:
                first_s = next(s for s in self.my_samples if s['id'] == first_id and s['diagnosed'])
            except StopIteration:
                # sample not carried, reset
                print("GOTO SAMPLES")
                return

            needed = {m: max(0, first_s['cost'][m] - self.my_exp[m]) for m in 'ABCDE'}
            if all(self.my_mol[m] >= needed[m] for m in 'ABCDE'):
                # ready to produce
                print("GOTO LABORATORY")
            else:
                # gather a missing type
                for m in 'ABCDE':
                    if self.my_mol[m] < needed[m] and available[m] > 0:
                        print(f"CONNECT {m}")
                        break
                else:
                    # needed molecules not available, change plan -> store first sample
                    self.store_mode = True
                    print("GOTO DIAGNOSIS")

        elif current == "LABORATORY":
            best_score, best_seq = self.select_best_sequence(available)
            if not best_seq:
                print("GOTO SAMPLES")
                return
            first_id = best_seq[0]
            try:
                first_s = next(s for s in self.my_samples if s['id'] == first_id and s['diagnosed'])
            except StopIteration:
                print("GOTO SAMPLES")
                return
            needed = {m: max(0, first_s['cost'][m] - self.my_exp[m]) for m in 'ABCDE'}
            if all(self.my_mol[m] >= needed[m] for m in 'ABCDE'):
                print(f"CONNECT {first_s['id']}")
                self.my_samples = [s for s in self.my_samples if s['id'] != first_s['id']]
                # after production, expertise will increase; target sequence will be recomputed next turn
            else:
                # shouldn't be here, but fallback
                print("GOTO MOLECULES")

        elif current == "START":
            print("GOTO SAMPLES")
        else:
            # unknown, go to samples
            print("GOTO SAMPLES")

        # Clear next_action_after_store if we printed a GOTO that wasn't from the store logic
        # but careful: we might have printed GOTO earlier; ensure we clean up.
        # We'll just delete attribute after use in DIAGNOSIS section; it's fine.

def main():
    bot = Bot()
    bot.parse_initial()
    try:
        while True:
            target, eta, available = bot.parse_turn()
            bot.decide_action(target, eta, available)
    except EOFError:
        pass

if __name__ == "__main__":
    main()