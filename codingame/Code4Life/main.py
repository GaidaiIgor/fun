import sys
from typing import List, Dict, Optional, Tuple
from itertools import combinations

MODULES = ["SAMPLES", "DIAGNOSIS", "MOLECULES", "LABORATORY", "START"]
MOVES = {
    "START": {"SAMPLES":2, "DIAGNOSIS":2, "MOLECULES":2, "LABORATORY":2},
    "SAMPLES": {"DIAGNOSIS":3, "MOLECULES":3, "LABORATORY":3},
    "DIAGNOSIS": {"SAMPLES":3, "MOLECULES":3, "LABORATORY":4},
    "MOLECULES": {"SAMPLES":3, "DIAGNOSIS":3, "LABORATORY":3},
    "LABORATORY": {"SAMPLES":3, "DIAGNOSIS":4, "MOLECULES":3},
}

class Bot:
    def __init__(self):
        self.projects = []
        self.my_mol = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_exp = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_samples = []            # list of dicts: id, carried_by, rank, gain, health, cost dict, diagnosed bool
        self.diagnosed_ids = set()
        self.cloud_samples = {}
        self.phase = 'START'
        self.target_set = []            # list of sample IDs we currently aim to produce (ordered)
        self.current_target_idx = 0     # index in target_set of the sample we are gathering for
        self.next_rank = 1              # rank to request at SAMPLES
        self.pending_sample_connect = False
        self.sample_before_connect = 0

    def parse_initial(self):
        project_count = int(input())
        for _ in range(project_count):
            a,b,c,d,e = map(int, input().split())
            self.projects.append({'A':a,'B':b,'C':c,'D':d,'E':e})

    def parse_turn(self):
        # my data
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

        # opponent (ignore)
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
                'carried_by': carried_by
            }
            if carried_by == 0:
                new_carried.append(s)
            elif carried_by == -1:
                new_cloud[sid] = s

        # update my state
        self.my_samples = []
        for s in new_carried:
            s['diagnosed'] = s['id'] in self.diagnosed_ids
            self.my_samples.append(s)
        self.cloud_samples = new_cloud

        return target, eta, available

    def net_cost(self, sample, final_exp=None):
        """Return dict of remaining cost after expertise (and after production order)."""
        if final_exp is None:
            final_exp = self.my_exp
        req = {}
        for m in 'ABCDE':
            req[m] = max(0, sample['cost'][m] - final_exp[m])
        return req

    def feasible_production_sequence(self, samples, available, inventory):
        """Return (best_score, sequence_of_sample_indices) for given list of samples.
        Samples are dicts with 'cost','gain','health'. We simulate production order."""
        best_score = -1
        best_seq = []
        # brute force all orders
        import itertools
        for perm in itertools.permutations(range(len(samples))):
            exp = dict(self.my_exp)
            mol = dict(inventory)
            total_health = 0
            feasible = True
            for idx in perm:
                s = samples[idx]
                # required after current expertise
                needed = {m: max(0, s['cost'][m] - exp[m]) for m in 'ABCDE'}
                # check if we have enough molecules
                for m in 'ABCDE':
                    if mol[m] < needed[m]:
                        feasible = False
                        break
                if not feasible:
                    break
                # produce: consume molecules and gain expertise
                for m in 'ABCDE':
                    mol[m] -= needed[m]
                exp[s['gain']] += 1
                total_health += s['health']
                # check project completion bonus
                # bonus = self.project_bonus_before_after(old_exp, exp) but too complex for small sets, ignore for target selection (we'll rely on actual production)
                # but we can add bonus if all projects completed, but not necessary now
            if feasible and total_health > best_score:
                # tie-break: prefer fewer molecules consumed total
                best_score = total_health
                best_seq = list(perm)
        return best_score, best_seq

    def project_bonus(self, old_exp, new_exp):
        bonus = 0
        for proj in self.projects:
            prev = all(old_exp[t] >= proj[t] for t in 'ABCDE')
            now = all(new_exp[t] >= proj[t] for t in 'ABCDE')
            if now and not prev:
                bonus += 50
        return bonus

    def select_target_set(self, available):
        """Choose a set of diagnosed samples to produce, trying up to 3,
        that yields max health using currently available molecules + our inventory.
        Also ensure total gathered molecules (net needed minus inventory) <= available.
        Returns list of sample IDs in production order, or empty list if none."""
        diag = [s for s in self.my_samples if s['diagnosed']]
        if not diag:
            return []
        best_score = -1
        best_seq_ids = []
        for r in range(1, min(len(diag), 3)+1):
            for subset_indices in combinations(range(len(diag)), r):
                subset = [diag[i] for i in subset_indices]
                # Compute total molecules needed from market: sum of net_cost minus current inventory.
                # We'll simulate full sequence, but also check market feasibility.
                # We need to consider that we will gather missing molecules before producing.
                # Simulate production order to find net molecules needed.
                best_order_score, order = self.feasible_production_sequence(subset, available, self.my_mol)
                if best_order_score > best_score:
                    # also verify that we can gather the required molecules within available and inventory capacity
                    # feasible_production_sequence already checks inventory sufficiency at each step,
                    # but assumes we already have those molecules. We'll check if we can gather them.
                    # The function used inventory as current my_mol, so it means we need to have those molecules now.
                    # That's too strict. We need a plan where we first gather needed molecules from market.
                    # We'll redo: compute total required after final expertise, subtract current inventory, that's what we need to gather.
                    pass
        # Instead, implement a proper plan: for a subset, compute final expertise after producing all,
        # then aggregate cost after final_exp, then needed_gather = max(0, agg - current_inv).
        # Then check if needed_gather <= available and current_inv + needed_gather <= 10 and slot constraint (molecules per type <=5 available).
        # But expertise gain during production can reduce later sample costs. We can approximate by assuming we produce all in order that minimizes needed molecules? Not trivial.
        # I'll use a simpler greedy: produce easiest (lowest cost) first to build expertise, then hardest.
        # We'll select subsets that are feasible under current market and inventory after gathering optimally.
        # Let's brute force all orders and simulate gathering + production in one go from market.
        diag_with_idx = list(enumerate(diag))
        best = (0, [])
        for r in range(1, min(len(diag),3)+1):
            for idx_list in combinations(range(len(diag)), r):
                subset = [diag[i] for i in idx_list]
                # Try all permutations
                for perm in itertools.permutations(subset):
                    exp = dict(self.my_exp)
                    inv = dict(self.my_mol)
                    gather_needed = {m:0 for m in 'ABCDE'}
                    feasible = True
                    health = 0
                    for s in perm:
                        # compute needed after current exp
                        needed = {m: max(0, s['cost'][m] - exp[m]) for m in 'ABCDE'}
                        # what we have + gather so far
                        for m in 'ABCDE':
                            if inv[m] < needed[m]:
                                # we will need to gather more, but only if not already planned
                                extra = needed[m] - inv[m]
                                if gather_needed[m] + extra > available[m]:  # market limit
                                    feasible = False
                                    break
                                gather_needed[m] += extra
                                inv[m] += extra   # immediately simulate gathering
                        if not feasible:
                            break
                        # check capacity (total gathered + initial inventory)
                        if sum(inv.values()) > 10:
                            feasible = False
                            break
                        # produce
                        for m in 'ABCDE':
                            inv[m] -= needed[m]
                        exp[s['gain']] += 1
                        health += s['health']
                    if feasible:
                        # calculate project bonus (approximate)
                        bonus = self.project_bonus(self.my_exp, exp)
                        total = health + bonus
                        if total > best[0]:
                            best = (total, [s['id'] for s in perm])
        return best[1]

    def decide_action(self, target, eta, available):
        if eta > 0:
            print("WAIT")
            return

        current = target
        # Resolve pending sample connection
        if current == "SAMPLES" and self.pending_sample_connect:
            if len(self.my_samples) == self.sample_before_connect:
                # failed to get sample, this rank probably exhausted
                self.next_rank = self.next_rank + 1 if self.next_rank < 3 else 1
            else:
                # succeeded, reset rank cycle
                self.next_rank = 1 if self.next_rank == 3 else self.next_rank + 1
            self.pending_sample_connect = False

        # Phase transitions based on situation
        # Diagnostic logic: if we are at SAMPLES or DIAGNOSIS and have undiagnosed samples, go diagnose.
        # If we are carrying diagnosed samples and no target set, compute target set.
        if current == "SAMPLES":
            # If we have less than 3 samples, try to get more
            if len(self.my_samples) < 3:
                self.sample_before_connect = len(self.my_samples)
                print(f"CONNECT {self.next_rank}")
                self.pending_sample_connect = True
            else:
                # inventory full, go diagnose any undiagnosed
                undiag = [s for s in self.my_samples if not s['diagnosed']]
                if undiag:
                    print("GOTO DIAGNOSIS")
                else:
                    # all diagnosed, go produce
                    print("GOTO MOLECULES")
        elif current == "DIAGNOSIS":
            undiag = [s for s in self.my_samples if not s['diagnosed']]
            if undiag:
                s = undiag[0]
                print(f"CONNECT {s['id']}")
                self.diagnosed_ids.add(s['id'])
            else:
                # all diagnosed; we could store heavy samples if no plan
                # Compute if we have any feasible production
                feas = self.select_target_set(available)
                if feas:
                    self.target_set = feas
                    self.current_target_idx = 0
                    print("GOTO MOLECULES")
                else:
                    # No feasible plan with current samples; need to drop heavy ones
                    # Store the most expensive sample (by total cost) to cloud to free space for cheaper
                    diag = [s for s in self.my_samples if s['diagnosed']]
                    if diag:
                        # pick sample with highest total cost
                        s = max(diag, key=lambda x: sum(x['cost'].values()))
                        print(f"CONNECT {s['id']}")  # this stores it to cloud
                        self.my_samples.remove(s)   # local removal
                    else:
                        print("GOTO SAMPLES")
        elif current == "MOLECULES":
            if not self.target_set:
                # try to select a target set from diagnosed samples
                self.target_set = self.select_target_set(available)
                if not self.target_set:
                    # no feasible plan, go back to get more samples
                    print("GOTO SAMPLES")
                    return
                self.current_target_idx = 0
            # We have a target set, gather molecules for the current target index
            target_samples = [s for s in self.my_samples if s['id'] in self.target_set and s['diagnosed']]
            if not target_samples:
                # our targets no longer exist (should not happen), reset
                self.target_set = []
                print("GOTO SAMPLES")
                return
            # For the first target in the sequence, compute needed molecules
            s = target_samples[0]  # first in plan
            # Simulate production order to know final expertise later? We'll assume we produce in order of target_set.
            # But our plan already considered optimal order; we'll just follow the order of target_set.
            # Compute needed molecules for this sample given current expertise and inventory
            needed = {m: max(0, s['cost'][m] - self.my_exp[m]) for m in 'ABCDE'}
            have = {m: self.my_mol[m] for m in 'ABCDE'}
            gather_mol = None
            for m in 'ABCDE':
                if have[m] < needed[m] and available[m] > 0:
                    gather_mol = m
                    break
            if gather_mol:
                print(f"CONNECT {gather_mol}")
            else:
                # We have all molecules needed for this sample, go produce
                print("GOTO LABORATORY")
        elif current == "LABORATORY":
            if not self.target_set:
                print("GOTO SAMPLES")
                return
            # Produce the first sample in target_set that we can
            target_samples = [s for s in self.my_samples if s['id'] in self.target_set and s['diagnosed']]
            if not target_samples:
                self.target_set = []
                print("GOTO SAMPLES")
                return
            # Find a sample we can produce now
            produced = False
            for s in target_samples:
                needed = {m: max(0, s['cost'][m] - self.my_exp[m]) for m in 'ABCDE'}
                if all(self.my_mol[m] >= needed[m] for m in 'ABCDE'):
                    print(f"CONNECT {s['id']}")
                    self.target_set.remove(s['id'])
                    produced = True
                    break
            if not produced:
                # shouldn't happen, but fallback
                print("GOTO MOLECULES")
        elif current == "START":
            print("GOTO SAMPLES")
        else:
            print("GOTO SAMPLES")  # fallback

def main():
    bot = Bot()
    bot.parse_initial()
    import itertools
    while True:
        try:
            target, eta, available = bot.parse_turn()
            bot.decide_action(target, eta, available)
        except EOFError:
            break

if __name__ == "__main__":
    main()