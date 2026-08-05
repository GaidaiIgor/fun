# Selenia City Rules

## Bundle Generation

### Layouts

- Bundle generation follows the hierarchy pool, paired pool, layout, and pod round.
- A tube layout represents a particular source-to-destination path. More expensive tube layouts are considered only when they shorten the path.
- A teleport is a separate layout and has only round 0. It has no pod or upgrade rounds.
- A new layout inherits the pod configuration from the currently selected bundle. The pod configuration does not start from scratch.
- Candidate actions are relative to the currently selected bundle. The final selected line shows the complete resulting bundle.

### Round 0

- Round 0 builds the tubes required by the considered path, when necessary, and otherwise keeps the currently selected pod configuration.
- On the first iteration, where no pod configuration can be inherited, round 0 reroutes one eligible existing pod. If no pod exists, it builds one.
- Rerouting prefers the closest eligible pod and prefers a pod that is already dynamic. Every pod tied after those criteria produces a separate round-0 candidate; pod ID is not a selection tiebreaker.
- The most efficient affordable round-0 candidate becomes the parent of round 1.

### Pod Rounds

- Each round after round 0 starts from the winner of the previous round and may generate these candidates:
  - `+pod`, if the resulting number of pods does not exceed the layout's total tube capacity.
  - `+upgrade`, if the winner of the previous round has tube congestion.
  - `+pod+upgrade`, if temporarily adding the pod creates tube congestion. Upgrade eligibility for this candidate is based on the `+pod` probe, not on the previous winner.
- The `+pod` probe is still performed when a standalone `+pod` bundle would exceed current tube capacity, because the simultaneous upgrade may make the combined bundle valid.
- The winner of a round is its affordable candidate with the greatest marginal efficiency.
- Another round is generated only when the current winner has greater marginal efficiency than the winner of the previous round.
- The final candidate for the layout is the bundle with the greatest marginal efficiency across all generated rounds, not necessarily the winner of the last round.

### Cost And Labels

- Marginal efficiency is marginal global score gain divided by marginal cost, both measured against the currently selected bundle.
- Bundle labels use `{X}p{Y}u{Z}r`: `X` is the number of newly built pods, `Y` is the number of upgrades, and `Z` is the number of reroutes.
