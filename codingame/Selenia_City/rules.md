# Selenia City Rules

## Pod Bundle Generation

- Pod bundles are considered in rounds. Each round consists of one or more bundles related to pods, upgrades and reroutes
- Round 0 is just tube construction for the considered path (if necessary) + pod configuration copied from the currently selected bundle
  - For iteration 1, since there is nothing to inherit, round 0 considers reroute of 1 pod, or construction if no pods exist
  - Round 0 should consider all reroute choices
- Each subsequent round considers the following bundles
  - Previous round's winner +1 pod (if layout capacity allows more pods)
  - Previous round's winner +1 upgrade (if congestion exists in previous round's winner)
  - Previous round's winner +1 pod +1 upgrade (if congestion exists in the +pod bundle)
- The winner in each round is chosen based on the largest marginal efficiency
- New rounds are generated if last round's winner's efficiency was greater than the one in the round before
- Largest efficiency bundle in all rounds is selected as final
- Teleports should be round-0 only

## Pod Dispatcher Rules

- Pods should not lock their chosen paths between days. Each day they pick available paths again. Partially transferred groups have their own paths.
- Dispatcher does (after all) maintain daily assignment map between pods and loads. Then, when dynamic pod assignments are considered, the dispatcher will know in advance which loads need to be reserved for existing fixed pods and only distribute the remaining ones among the dynamic pods. Of course, if there are no more loads left, then it's ok to start distributing the reserved ones in order to finish them faster.
- Loads should be assigned uniformly in a round robin fashion. No load should receive X+1 assigned pod while there are still loads with <X pods assigned. This applies to priorities of the same tier.

There should be only 1 (pre-conflict) assignment layer.

Namely, pods should be allocated to loads in the following priority:

0. Capacity-respecting assignments
1. Higher priority assignments
2. Uniformity-respecting assignments
3. 10+ would-be-departing passengers
4. Shorter distance to load origin
5. Shorter path length
6. Smaller number of passengers delivered to the target module
7. Larger total number of load's passenger

For each load, check how many non-reserved passengers would be boarding if a pod was departing in the direction of the load. Prefer loads with larger, but capped at 10, number of passengers.

That's it. 8 rules total, 1 of which is new. 9 if you count the last load id deterministic fallback. 1 layer before conflict resolution.

Then conflict resolution layer kicks in. If more pods want to use a given edge than edge capacity allows, then it tries to re-assign some of the loads to avoid conflict.

- If some pods want to traverse the edge in the opposite directions, then the easiest fix is to just swap their assignments. That should resolve the conflict.
- If they are going in the same direction then a new load should be chosen for one of them such that it sends a pod to some other conflict-free edge. Currently we do not have any rules for how to choose which pod will be reassigned so it falls straight to deterministic ID resolution (lower id pods rerouted first).
- For the pod chosen to be rerouted, the remaining loads continue to be traversed in the same preference order according to the list above until the first load that sends it to a non-conflicting edge is found.
- Then the process is repeated for other conflicting pods, if any.
- If such conflict-free loads are not found for the chosen pod, then conflict is unavoidable and the dispatcher just keeps the original assignment.

That's it for the conflict resolution. No other special cases or rules.
