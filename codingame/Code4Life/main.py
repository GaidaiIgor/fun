import sys
from typing import Dict, List, Optional

class Bot:
    def __init__(self):
        self.projects = []                         # not used heavily but kept for initialization
        self.my_mol = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_exp = {'A':0,'B':0,'C':0,'D':0,'E':0}
        self.my_samples: List[Dict] = []           # carried samples, each: id, rank, gain, health, cost dict, diagnosed
        self.cloud_samples: Dict[int, Dict] = {}   # id -> sample in cloud
        self.diagnosed_ids = set()                 # ids we have diagnosed (own)
        self.wait_counter = 0                      # turns waiting for molecules
        self.target_sample_id = None               # id of the sample we are currently focusing on

    def parse_initial(self):
        project_count = int(input())
        for _ in range(project_count):
            a, b, c, d, e = map(int, input().split())
            self.projects.append({'A':a,'B':b,'C':c,'D':d,'E':e})

    def parse_turn(self):
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

        input()  # opponent – not used

        avail = list(map(int, input().split()))
        available = {'A':avail[0],'B':avail[1],'C':avail[2],'D':avail[3],'E':avail[4]}

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

        # update carried list, keeping diagnosed flag
        new_my_samples = []
        for s in new_carried:
            s['diagnosed'] = s['id'] in self.diagnosed_ids
            new_my_samples.append(s)
        self.my_samples = new_my_samples
        self.cloud_samples = new_cloud

        return target, eta, available

    def missing_to_produce(self, sample: Dict) -> Dict[str, int]:
        """Molecules we still need to gather for this sample (given current inventory & expertise)."""
        need = {}
        for m in 'ABCDE':
            need[m] = max(0, sample['cost'][m] - self.my_exp[m] - self.my_mol[m])
        return need

    def can_produce_now(self, sample: Dict) -> bool:
        """True if we have enough molecules in inventory to produce the sample right now."""
        for m in 'ABCDE':
            if self.my_mol[m] < max(0, sample['cost'][m] - self.my_exp[m]):
                return False
        return True

    def sample_is_feasible(self, sample: Dict, available: Dict[str, int]) -> bool:
        """Can we eventually produce this sample given the current market and our inventory capacity?"""
        missing = self.missing_to_produce(sample)
        for m in 'ABCDE':
            if missing[m] > available[m]:
                return False
        if sum(self.my_mol.values()) + sum(missing.values()) > 10:
            return False
        return True

    def select_best_feasible_sample(self, available: Dict[str, int]) -> Optional[Dict]:
        """Among carried diagnosed samples, pick one that can be produced after gathering,
           preferring highest health (ties: fewer total missing molecules)."""
        diag = [s for s in self.my_samples if s['diagnosed']]
        best = None
        best_health = -1
        best_gather = 999
        for s in diag:
            if not self.sample_is_feasible(s, available):
                continue
            missing = self.missing_to_produce(s)
            gather = sum(missing.values())
            if s['health'] > best_health or (s['health'] == best_health and gather < best_gather):
                best_health = s['health']
                best_gather = gather
                best = s
        return best

    def select_sample_to_produce_now(self) -> Optional[Dict]:
        """Carried diagnosed sample that can be produced immediately (no gathering needed), highest health."""
        diag = [s for s in self.my_samples if s['diagnosed'] and self.can_produce_now(s)]
        if not diag:
            return None
        return max(diag, key=lambda s: s['health'])

    def decide_action(self, target: str, eta: int, available: Dict[str, int]):
        if eta > 0:
            print("WAIT")
            return

        current = target

        if current == "START":
            print("GOTO SAMPLES")
            return

        if current == "SAMPLES":
            if len(self.my_samples) < 3:
                # always ask for rank 1 – low cost, quick expertise
                print("CONNECT 1")
            else:
                if any(not s['diagnosed'] for s in self.my_samples):
                    print("GOTO DIAGNOSIS")
                else:
                    # all diagnosed, head for production
                    if self.select_sample_to_produce_now():
                        print("GOTO LABORATORY")
                    else:
                        print("GOTO MOLECULES")
            return

        if current == "DIAGNOSIS":
            undiag = [s for s in self.my_samples if not s['diagnosed']]
            if undiag:
                s = undiag[0]
                print(f"CONNECT {s['id']}")
                self.diagnosed_ids.add(s['id'])
                return

            # all carried are diagnosed
            # try to retrieve a feasible cloud sample if we have room
            if len(self.my_samples) < 3:
                own_cloud_ids = [sid for sid in self.cloud_samples if sid in self.diagnosed_ids]
                feasible_cloud = []
                for sid in own_cloud_ids:
                    s = self.cloud_samples[sid]
                    if self.sample_is_feasible(s, available):
                        feasible_cloud.append(s)
                if feasible_cloud:
                    best_cloud = max(feasible_cloud, key=lambda s: s['health'])
                    print(f"CONNECT {best_cloud['id']}")
                    return

            # check if we can produce something
            best_feasible = self.select_best_feasible_sample(available)
            if best_feasible:
                self.target_sample_id = best_feasible['id']
                if self.can_produce_now(best_feasible):
                    print("GOTO LABORATORY")
                else:
                    print("GOTO MOLECULES")
            else:
                # no feasible production – store the hardest sample to free a slot
                if self.my_samples:
                    worst = max(self.my_samples, key=lambda s: sum(s['cost'].values()))
                    print(f"CONNECT {worst['id']}")
                    self.my_samples = [s for s in self.my_samples if s['id'] != worst['id']]
                    # stay at DIAGNOSIS, next turn we might retrieve a better cloud sample or go to SAMPLES
                else:
                    print("GOTO SAMPLES")
            return

        if current == "MOLECULES":
            # can we produce anything right now?
            prod_now = self.select_sample_to_produce_now()
            if prod_now:
                self.target_sample_id = prod_now['id']
                print("GOTO LABORATORY")
                return

            best_feasible = self.select_best_feasible_sample(available)
            if best_feasible is None:
                # nothing producible with current market; go store a sample
                self.wait_counter = 0
                print("GOTO DIAGNOSIS")
                return

            self.target_sample_id = best_feasible['id']
            missing = self.missing_to_produce(best_feasible)
            for m in 'ABCDE':
                if missing[m] > 0 and available[m] > 0:
                    print(f"CONNECT {m}")
                    self.wait_counter = 0
                    return

            # required molecules not available – wait a few turns
            self.wait_counter += 1
            if self.wait_counter > 3:
                self.wait_counter = 0
                self.target_sample_id = None
                print("GOTO DIAGNOSIS")   # go store this problematic sample
            else:
                print("WAIT")
            return

        if current == "LABORATORY":
            prod = self.select_sample_to_produce_now()
            if prod:
                print(f"CONNECT {prod['id']}")
                self.my_samples = [s for s in self.my_samples if s['id'] != prod['id']]
                self.diagnosed_ids.discard(prod['id'])
                self.target_sample_id = None
                self.wait_counter = 0
            else:
                print("GOTO MOLECULES")
            return

        # fallback (should never reach)
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