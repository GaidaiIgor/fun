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

- A passenger group is ambiguous at a node when the same passengers have equally short remaining paths to multiple matching modules.
- The diversity planner divides an ambiguous group among one or more destinations to maximize diversity score. Each enabled destination becomes a load capped at its allocated passenger count; other destinations are not active loads.
- An enabled ambiguous load is high priority while its destination remains unresolved and its cap has not been reached.
- Moving along a path shared by multiple enabled destinations does not resolve ambiguity. After paths diverge, the resulting partial load has a unique destination and normal priority.
- High priority is considered after assignment capacity and before uniformity, as specified by the dispatcher priority list.
- Loads sharing passengers also share their available supply. Assigning a pod reserves its would-be-boarding passengers, so another pod or direction can use only the remaining passengers.
