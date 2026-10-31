"""Finds the maximum money Blunder can collect walking from room 0 to the exit through an acyclic building."""
from graphlib import TopologicalSorter

money, doors = {}, {}
for _ in range(int(input())):
    room, cash, *exits = input().split()
    money[room], doors[room] = int(cash), [e for e in exits if e != "E"]
# Successors are listed as dependencies, so each room comes after all rooms reachable from it
best = {}
for room in TopologicalSorter(doors).static_order():
    best[room] = money[room] + max((best[d] for d in doors[room]), default=0)
print(best["0"])
