# Joint representative choice: registered follow-up, not a novelty claim

2026-10-09, while full-population v2 seed0 queue is still running. No v3 model
has run. Same authorized7200s cumulative ledger and all1152TRAIN population.

An explicit synthetic counterexample exposes a limitation of the first-slot
protection heuristic: with two valid left routes at y=-.1 and y=-1 and an uncovered
right witness at y=1, locking the first route forces the second to move2m; retaining
the second lets the first move1.1m while preserving left coverage. This changes
when the input candidate order changes. The issue is target construction, not
evidence that it explains the observed model-level failure.

The registered next comparison, if v2 does not support hard locking, changes only
assignment: append currently verified/classified predictions as singleton positive
witnesses, and require their MODE support in the target set, while jointly choosing
which candidate represents each mode. Optional known groups compete for unused
slots by correction cost. Standard assignment uses dummy columns to leave optional
groups uncovered when group count exceeds8; mandatory current modes cannot go to
dummies. No full coverage is requested above the8-slot budget. Extra slots when
groups<8 may choose any admissible positive. Existing reference-group RNG calls
are still consumed so actual input and group RNG comparability can be audited.

Two arms isolate geometry regions from assignment:
- set_point: only original finite witnesses and verified current singleton routes.
- set_project: same assignment and positives plus the same2cm-capped certified
  local geometry regions as v1/v2. Mode-invalid selected proposals fall back to
  their witness. Zero-distance edges with known wrong modes are explicitly removed.

These are standard constrained-assignment constructions. Neither a new solver nor
global nearest feasible projection is claimed. Accepted target mode support is
guaranteed by rechecking the target; learned output mode retention is empirical.
Region fallback means even minimum candidate-cost matching is not a global optimum
over the entire verified feasible set. No no-forgetting guarantee is implied.

Before model training require synthetic counterexample, unique-optimum permutation
equivariance, valid distinct set zero-correction, overbudget novel-mode retention
and cross-mode zero-distance checks. No claim of equivariance among tied optima.
Compare fixed1200 seed0 to same-data ordinary/gate/project and historical parent;
no change in learning rate, update count, score function or main acceptance gate.
Do not run more variants merely because budget remains. A promising arm requires
paired continuation seeds and the simpler set_point/control comparisons before
method or paper claims. Stored replay targets and actual verification counts stay
available for an equal-information ordinary control.
