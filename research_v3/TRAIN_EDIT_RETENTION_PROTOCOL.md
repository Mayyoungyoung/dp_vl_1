# Next registered diagnostic: TRAIN edit residual and witness support

The current strong safety_mean DEV audit finds23/40/23/23 slot-feasible lost
modes using only already-correct-endpoint or duplicate slots across the four
directions. Many are already in the destination teacher-mode set; missing
reference support and shared endpoint failure are not complete explanations.
This does not establish that a new consistency objective has usable TRAIN
signal or will generalize. Historical R2/R3 negatives remain binding evidence.

Next use the already sealed full1152 TRAIN safety_mean predictions from
posttrain_collision_v1/predictions.npz, SHA
`9e3a136a578eb49977b0308555c6141a21cc2cfc6c6b39e1a1128d962febe301`.
Parent checkpoint SHA
`ff2dfbc5463a38e9acb2af740e9b605cbfda5d1317cc146f5f2c4e65c2a7a60c`.
Require exact manifest SHA,1152 TRAIN IDs/128 families and all3 edit variants
per target. Reuse the exact current checker, mode definition, start/goal
equality checks and all4 edge directions. No new model forward or optimization.

Report survival losses, total versus correct-endpoint slot-feasible opportunities,
all-endpoint failure contribution, and known versus unreferenced mode overlap.
Also measure each eligible surviving path's nearest same-mode destination
teacher path (registered mean Euclidean point distance over24 points), reporting
all finite distances/quantiles and missing-mode counts without filtering by a
favorable distance threshold. This tests whether the observed gap is already
present on TRAIN and whether potential positives add geometry beyond nearby
teacher perturbations. Keep every direction and case.

Do not automatically train a new loss or relabel unreferenced modes invalid.
If TRAIN opportunities are scarce, treat generalization as a separate issue;
if plentiful, a later proposal must compare verified positive augmentation
against any proposed correspondence mechanism under the same data/updates.
Cap120s CPU geometry work inside1342.190788 remaining command seconds; existing
GPU1/35%/four-thread wrapper, immutable source and all receipts. No DEV-driven
threshold, score-role fit, TEST_LOCKED access or raw collector traversal.
