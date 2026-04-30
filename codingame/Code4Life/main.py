import sys
import math
from typing import Dict, List, Optional

class Bot:
    def __init__(self):
        self.projects = []          # list of dicts: [{'A':..., 'B':...}, ...]
        self.my_molecules = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_expertise = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_samples = []        # list of dicts with keys: id, rank, gain, health, cost dict, diagnosed
        self.diagnosed_ids = set()
        self.cloud_samples = {}
        self.we_own_cloud = set()   # not used heavily
        self.mol_target_set = []    # list of sample ids we are currently gathering for
        self.next_rank = 3
        self.pending_connect = False
        self.pre_connect_count = 0

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
        self.my_molecules['A'] = int(parts[3])
        self.my_molecules['B'] = int(parts[4])
        self.my_molecules['C'] = int(parts[5])
        self.my_molecules['D'] = int(parts[6])
        self.my_molecules['E'] = int(parts[7])
        self.my_expertise['A'] = int(parts[8])
        self.my_expertise['B'] = int(parts[9])
        self.my_expertise['C'] = int(parts[10])
        self.my_expertise['D'] = int(parts[11])
        self.my_expertise['E'] = int(parts[12])
        my_target = target
        my_eta = eta
        my_score = score

        # opponent line (discard)
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

        # update my_samples
        self.my_samples = []
        for s in new_carried:
            s['diagnosed'] = s['id'] in self.diagnosed_ids
            self.my_samples.append(s)
        self.cloud_samples = new_cloud

        return my_target, my_eta, available

    def net_required(self, sample, expertise, molecules):
        req = {}
        for m in 'ABCDE':
            req[m] = max(0, sample['cost'][m] - expertise[m] - molecules[m])
        return req

    def project_bonus(self, current_exp, new_exp):
        # count projects that become completed under new_exp but weren't before
        bonus = 0
        for proj in self.projects:
            already = all(current_exp[t] >= proj[t] for t in 'ABCDE')
            new_ok = all(new_exp[t] >= proj[t] for t in 'ABCDE')
            if new_ok and not already:
                bonus += 50
        return bonus

    def select_best_subset(self, available):
        diag = [s for s in self.my_samples if s['diagnosed']]
        best_score = -1
        best_subset = None
        # brute force all non-empty subsets
        from itertools import chain, combinations
        for r in range(1, len(diag)+1):
            for subset in combinations(diag, r):
                # compute final expertise after all produced
                final_exp = dict(self.my_expertise)
                for s in subset:
                    final_exp[s['gain']] += 1
                # aggregated required after final expertise
                agg = {m:0 for m in 'ABCDE'}
                total_health = 0
                for s in subset:
                    total_health += s['health']
                    for m in 'ABCDE':
                        agg[m] += max(0, s['cost'][m] - final_exp[m])
                # needed to gather now
                needed = {}
                total_gather = 0
                feasible = True
                for m in 'ABCDE':
                    needed[m] = max(0, agg[m] - self.my_molecules[m])
                    total_gather += needed[m]
                    if needed[m] > available[m]:
                        feasible = False
                        break
                if not feasible:
                    continue
                # inventory after gathering
                total_inv = sum(self.my_molecules.values()) + total_gather
                if total_inv > 10:
                    continue
                # score = total health + project bonus
                bonus = self.project_bonus(self.my_expertise, final_exp)
                score = total_health + bonus
                # tie-break: prefer less total gather
                if score > best_score or (score == best_score and total_gather < best_gather):
                    best_score = score
                    best_subset = list(subset)
                    best_gather = total_gather
        return best_subset

    def decide_action(self, target, eta, available):
        if eta > 0:
            # moving: wait
            print("WAIT")
            return

        current = target

        # SAMPLES pending connect resolution
        if current == "SAMPLES" and self.pending_connect:
            if len(self.my_samples) == self.pre_connect_count:
                # didn't get a sample → rank unavailable
                self.next_rank = max(1, self.next_rank - 1)
            else:
                self.next_rank = 3   # got a sample, reset
            self.pending_connect = False

        if current == "SAMPLES":
            # collect samples if we have room
            if len(self.my_samples) < 3:
                self.pre_connect_count = len(self.my_samples)
                print(f"CONNECT {self.next_rank}")
                self.pending_connect = True
            else:
                # go diagnose
                undiag = [s for s in self.my_samples if not s['diagnosed']]
                if undiag:
                    print("GOTO DIAGNOSIS")
                else:
                    print("GOTO MOLECULES")

        elif current == "DIAGNOSIS":
            undiag = [s for s in self.my_samples if not s['diagnosed']]
            if undiag:
                s = undiag[0]
                print(f"CONNECT {s['id']}")
                self.diagnosed_ids.add(s['id'])
            else:
                # all diagnosed, go gather molecules
                print("GOTO MOLECULES")

        elif current == "MOLECULES":
            if not self.mol_target_set:
                best = self.select_best_subset(available)
                if best:
                    self.mol_target_set = [s['id'] for s in best]
                else:
                    # no feasible set, go get new samples
                    print("GOTO SAMPLES")
                    return

            # we have a target set -> gather needed molecules
            # recompute needed dynamically
            target_samples = [s for s in self.my_samples if s['id'] in self.mol_target_set and s['diagnosed']]
            if not target_samples:
                # targets no longer exist (shouldn't happen)
                self.mol_target_set = []
                print("GOTO SAMPLES")
                return

            final_exp = dict(self.my_expertise)
            for s in target_samples:
                final_exp[s['gain']] += 1
            agg = {m:0 for m in 'ABCDE'}
            for s in target_samples:
                for m in 'ABCDE':
                    agg[m] += max(0, s['cost'][m] - final_exp[m])
            needed = {m: max(0, agg[m] - self.my_molecules[m]) for m in 'ABCDE'}
            total_needed = sum(needed.values())

            if total_needed > 0:
                # gather one molecule of a needed type that is available
                for m in 'ABCDE':
                    if needed[m] > 0 and available[m] > 0:
                        print(f"CONNECT {m}")
                        break
                else:
                    # can't gather now, wait (molecule may be restocked later)
                    print("WAIT")
            else:
                # all molecules gathered, go produce
                print("GOTO LABORATORY")

        elif current == "LABORATORY":
            if self.mol_target_set:
                # try to produce a sample from the target set
                target_samples = [s for s in self.my_samples if s['id'] in self.mol_target_set and s['diagnosed']]
                # order by highest health first (optional)
                target_samples.sort(key=lambda x: x['health'], reverse=True)
                produced = False
                for s in target_samples:
                    req = self.net_required(s, self.my_expertise, self.my_molecules)
                    if all(req[m] == 0 for m in 'ABCDE'):
                        print(f"CONNECT {s['id']}")
                        self.mol_target_set.remove(s['id'])
                        produced = True
                        break
                if not produced:
                    # should not happen; fallback
                    print("GOTO MOLECULES")
            else:
                # no target set, go get new samples
                print("GOTO SAMPLES")

        elif current == "START":
            print("GOTO SAMPLES")
        else:
            # fallback
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