# Failure analysis and stopping decision

**Recovering a word differs from improving the complete set.** Frozen C has 370
omitted DEV words recoverable by sampled replacement, but only 77 have a positive
net gain without quality-count loss. TRAIN has 1590 recoverable / 181 net-positive
(180 nondegrading). TRAIN 5114 and DEV 1101 replacements add a word without
positive net coverage. Even with seven companion identities fixed, companion
coordinates can change. Local sampled oracle gain 0.181 TRAIN / 0.306 DEV is a
diagnostic, not an achievable score or bound on all possible query sets.

**The first net estimator fails on TRAIN too.** Predicted-safe-positive precision
is 16.0% TRAIN / 9.86% DEV. Its selected sampled-neighborhood mean gain is −0.0052
TRAIN and zero DEV. Four aggregate targets poorly identify rare beneficial
interventions in this fit. Dense actual-word classification helps over scalar
regression but loses to ordinary success in five head seeds (one numerical tie).
No-peer also exceeds peer attention in seed0. Interaction has not earned its
complexity.

**The simple proposal absorbs much of the local opportunity.** Around its TRAIN
sets, sampled oracle gain drops from 0.181 to 0.072 words/request and net-positive
recoverable omissions from 181 to 33. This is not a global impossibility result.
It makes false-positive replacement decisions particularly costly.

**More geometry updates damage usable behavior.** Ordinary3600 gives U8=6.6215,
V4=92.708%; KL-only gives 6.6111/92.882%. Gap plus refreshed dense reaches the
frozen-simple U8 but loses 0.694 percentage point returned validity and 37 retained
witnesses in seed0. Hard sampling does not rescue it. These budget-matched screens
have branch-dependent RNG and do not refute every possible joint-training design.

**Reference averaging is not the measured cause.** All 10168 TRAIN per-mode mean
reference paths pass the checker and preserve their word. No valid baseline case
is pulled toward an invalid mean in this diagnostic. Nonzero target variance is
insufficient evidence for conflicting targets. Optional nearest-positive code
was removed without being run; no canonical-draws experiment was performed.

**Adaptation is incomplete.** Success seed0 gives 235/270 repairs, 30 same-mode
invalid and 5 other-valid-only cases. Dense gives 236, 29, 4 and one all-invalid.
Zero copied-old category cases does not imply every output avoided old coordinates:
successful same-mode repair takes precedence in classification. Success loses a mean
161/1872 originally covered chances and recovers 68.6/297 missing; dense loses
162.2 and recovers 67.0. The +0.4 mean repair count has an interval crossing zero.
Neither reaches the old parent's survival retention.

**Cost exceeds demonstrated benefit.** Up to two token-neighborhood sweeps precede
one decode. The limited full-API benchmark is 11.925 ms dense vs 6.782 ms success.
All TRAIN counterfactuals and initial discarded decodes are charged separately.

**Engineering failure retained.** The initial collector hit NumPy-int JSON
serialization after 64 saved requests. Failure and time are preserved; those
files were resumed with identical generator/design and a recorded source fix.
Head and geometry actual100 vs50+50 checks exactly match model, optimizer, RNG,
stream, step, history and settings.

The stop follows context repair, dense credit, no-peer/added-only controls,
displacement-free geometry alternatives, snapshot refresh, five head seeds and
target-conflict diagnosis. Repeating configurations or selecting early peaks
would not answer a new question. This mechanism is unsupported in this setting;
all possible coverage-driven generators are not disproven. Removed historical
time caps did not determine stopping. A future joint design first needs a new
explanation for coordinate-regression damage and exact-stream comparison against
frozen C+success, before more new-scene or robot experiments.
