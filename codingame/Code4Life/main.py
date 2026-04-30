import sys
import itertools
from typing import Dict, List, Optional, Tuple

MODULES = ["START", "SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY"]

# Distance matrix
DIST = {
    "START":      {"SAMPLES":2, "DIAGNOSIS":2, "MOLECULES":2, "LABORATORY":2},
    "SAMPLES":    {"DIAGNOSIS":3, "MOLECULES":3, "LABORATORY":3},
    "DIAGNOSIS":  {"SAMPLES":3, "MOLECULES":3, "LABORATORY":4},
    "MOLECULES":  {"SAMPLES":3, "DIAGNOSIS":3, "LABORATORY":3},
    "LABORATORY": {"SAMPLES":3, "DIAGNOSIS":4, "MOLECULES":3},
}

class Bot:
    def __init__(self):
        self.projects: List[Dict[str,int]] = []
        self.my_mol = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_exp = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_samples: List[Dict] = []          # each: id, rank, gain, health, cost dict, diagnosed bool
        self.cloud_samples: Dict[int, Dict] = {}  # id -> sample dict
        self.diagnosed_ids = set()
        self.next_rank = 1                        # 1,2,3 cycle
        self.pending_sample_connect = False
        self.samples_before = 0
        self.mol_gather_wait = 0                  # turns waiting for unavailable molecule
        self.next_action_after_store = None       # used after storing at DIAGNOSIS

    def parse_initial(self):
        project_count = int(input())
        for _ in range(project_count):
            a,b,c,d,e = map(int, input().split())
            self.projects.append({'A':a,'B':b,'C':c,'D':d,'E':e})

    def parse_turn(self):
        # my player line
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
            costA = int(sp[5])
            costB = int(sp[6])
            costC = int(sp[7])
            costD = int(sp[8])
            costE = int(sp[9])
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

        # update local samples preserving diagnosed flag
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
            for m in 'ABCDE':
                short = needed[m] - inv[m]
                if short > 0:
                    if gather[m] + short > available[m]:
                        return None, None
                    gather[m] += short
                    inv[m] += short
            # capacity check
            if initial_sum + sum(gather.values()) > 10:
                return None, None
            # consume required molecules
            for m in 'ABCDE':
                inv[m] -= needed[m]
            exp[s['gain']] += 1
            health += s['health']

        bonus = self.project_bonus(self.my_exp, exp)
        return health + bonus, gather

    def select_best_sequence(self, available):
        """Return best sequence of sample IDs (ordered) from carried diagnosed samples."""
        diag = [s for s in self.my_samples if s['diagnosed']]
        if not diag:
            return []
        best_score = -1
        best_seq = []
        # try all non-empty subsets, any order
        for r in range(1, len(diag)+1):
            for subset in itertools.combinations(diag, r):
                for perm in itertools.permutations(subset):
                    score, gather = self.evaluate_sequence(list(perm), available)
                    if score is not None and score > best_score:
                        best_score = score
                        best_seq = [s['id'] for s in perm]
                    elif score is not None and score == best_score:
                        if gather and best_seq:
                            # prefer fewer total gathered molecules
                            prev_score, prev_gather = self.evaluate_sequence(
                                [s for s in diag if s['id'] in best_seq], available
                            )
                            if prev_gather and sum(gather.values()) < sum(prev_gather.values()):
                                best_seq = [s['id'] for s in perm]
        return best_seq

    def decide_action(self, target, eta, available):
        if eta > 0:
            print("WAIT")
            return

        current = target

        # resolve pending sample connect
        if current == "SAMPLES" and self.pending_sample_connect:
            if len(self.my_samples) > self.samples_before:
                # success: cycle rank to next
                self.next_rank = self.next_rank % 3 + 1
            else:
                # failure: also cycle (maybe that rank exhausted)
                self.next_rank = self.next_rank % 3 + 1
            self.pending_sample_connect = False

        if current == "SAMPLES":
            if len(self.my_samples) < 3:
                self.samples_before = len(self.my_samples)
                print(f"CONNECT {self.next_rank}")
                self.pending_sample_connect = True
            else:
                # inventory full, go diagnose if any undiagnosed
                if any(not s['diagnosed'] for s in self.my_samples):
                    print("GOTO DIAGNOSIS")
                else:
                    # all diagnosed, see if we should store a weak one to get better
                    total_health = sum(s['health'] for s in self.my_samples)
                    if total_health < 20:   # threshold to discard low-value samples
                        # prepare to store the worst one
                        worst = min(self.my_samples, key=lambda s: (s['health'], -sum(s['cost'].values())))
                        # go to DIAGNOSIS to store it
                        self.next_action_after_store = "SAMPLES"
                        print("GOTO DIAGNOSIS")
                    else:
                        # proceed to produce
                        print("GOTO MOLECULES")

        elif current == "DIAGNOSIS":
            # handle any pending action after a store
            if self.next_action_after_store is not None:
                dest = self.next_action_after_store
                self.next_action_after_store = None
                print(f"GOTO {dest}")
                return

            undiag = [s for s in self.my_samples if not s['diagnosed']]
            if undiag:
                s = undiag[0]
                print(f"CONNECT {s['id']}")
                self.diagnosed_ids.add(s['id'])
            else:
                # all carried diagnosed, maybe fetch from cloud if space
                if len(self.my_samples) < 3:
                    owned_cloud = [sid for sid, s in self.cloud_samples.items() if sid in self.diagnosed_ids]
                    if owned_cloud:
                        # pick the best by health
                        best_cloud_id = max(owned_cloud, key=lambda sid: self.cloud_samples[sid]['health'])
                        print(f"CONNECT {best_cloud_id}")
                        # will be added to carried next turn
                        return
                # decide production or storage
                best_seq = self.select_best_sequence(available)
                if best_seq:
                    print("GOTO MOLECULES")
                else:
                    # no feasible plan, if we have 3 diagnosed samples, store the most expensive
                    if len(self.my_samples) == 3:
                        worst = max(self.my_samples, key=lambda s: sum(s['cost'].values()))
                        print(f"CONNECT {worst['id']}")
                        self.my_samples = [s for s in self.my_samples if s['id'] != worst['id']]
                        self.diagnosed_ids.discard(worst['id'])  # not necessary, but keep
                        self.next_action_after_store = "SAMPLES"
                    else:
                        # less than 3, go get more samples
                        print("GOTO SAMPLES")

        elif current == "MOLECULES":
            # reset wait counter if we just arrived? don't
            best_seq = self.select_best_sequence(available)
            if not best_seq:
                # can't produce anything, go store or get new samples
                print("GOTO DIAGNOSIS")
                self.next_action_after_store = "SAMPLES"
                return

            first_id = best_seq[0]
            try:
                first_s = next(s for s in self.my_samples if s['id'] == first_id and s['diagnosed'])
            except StopIteration:
                print("GOTO SAMPLES")
                return

            needed = {m: max(0, first_s['cost'][m] - self.my_exp[m]) for m in 'ABCDE'}
            if all(self.my_mol[m] >= needed[m] for m in 'ABCDE'):
                # ready to produce
                print("GOTO LABORATORY")
                self.mol_gather_wait = 0
            else:
                # gather a missing type that is available
                for m in 'ABCDE':
                    if self.my_mol[m] < needed[m] and available[m] > 0:
                        print(f"CONNECT {m}")
                        self.mol_gather_wait = 0
                        break
                else:
                    # none available, wait (maybe later restocked)
                    self.mol_gather_wait += 1
                    if self.mol_gather_wait > 10:
                        # abandon this plan, store the first sample and go to DIAGNOSIS
                        worst = first_s
                        print("GOTO DIAGNOSIS")
                        self.next_action_after_store = "SAMPLES"
                        self.mol_gather_wait = 0
                    else:
                        print("WAIT")

        elif current == "LABORATORY":
            best_seq = self.select_best_sequence(available)
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
                self.diagnosed_ids.discard(first_s['id'])
            else:
                # shouldn't happen, fallback
                print("GOTO MOLECULES")

        elif current == "START":
            print("GOTO SAMPLES")
        else:
            print("GOTO SAMPLES")

def main():
    bot = Bot()
    bot.parse_initial()
    while True:
        try:
            target, eta, available = bot.parse_turn()
            bot.decide_action(target, eta, available)
        except EOFError:
            break

if __name__ == "__main__":
    main()