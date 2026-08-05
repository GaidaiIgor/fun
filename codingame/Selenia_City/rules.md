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

- Assignments are recalculated every day. Partially transferred groups become separate loads with their remaining paths.
- Fixed pod routes reserve passengers by day. Reserved passengers are excluded from would-be-boarding counts for dynamic pods.
- Before conflict resolution, assign dynamic pods to loads sequentially using this priority order:

1. Respect assignment capacity
2. Prefer higher-priority loads
3. Prefer uniform allocation: no load receives X+1 pods while an equal-priority load has fewer than X
4. Prefer more non-reserved passengers that would board immediately, capped at 10
5. Prefer shorter distance to the load origin
6. Prefer shorter remaining path
7. Prefer the target module with fewer delivered passengers
8. Prefer larger total remaining load
9. Prefer the smaller load ID

- If assignments exceed an edge capacity, resolve conflicts as follows:
  - Swap assignments of pods moving in opposite directions when the swap resolves the conflict
  - Otherwise, try pods in ascending ID order and give each the first later load from the same priority order that produces a conflict-free move
  - Repeat until no conflict remains; if no reassignment works, keep the original conflicting assignment

## Ambiguous Loads And High-Priority Assignments

- A passenger group is ambiguous when it has equally short remaining paths to multiple matching modules.
- The pod dispatcher divides ambiguous groups among their tied destinations in a way that equilibrates the total number of inbound passengers at each module from all groups.
- A passenger group is considered inbound to a target module if the distance to that module is the shortest and no other modules have the same distance.
- Directions toward modules that received non-zero planned passenger counts from ambiguous groups are set to high priority on that day.
  - Other ambiguous directions (that received 0 planned passenger counts) from the same node are set to low priority on that day.
- For each edge corresponding to a high priority direction, the dispatcher inspects the actual would-be-boarding passengers in that direction (batch). If the number of passengers in that batch for whom that direction is low priority is greater than the number of passengers for whom that direction is high priority, then priority of that direction is set to low on that day.
  - Batches reserved by lower id pods should not be inspected again by higher id pods considering the same node.
