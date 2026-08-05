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
