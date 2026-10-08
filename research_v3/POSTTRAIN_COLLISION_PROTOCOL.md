# Post-training collision persistence diagnostic

The fixed safety_mean model has207/2304 DEV post-collision candidates, including
196 with correct semantic endpoints. All207 satisfy the workspace floor. The
previous four-batch gradient audit used the earlier full-set checkpoint, so it
does not establish whether collision loss remains poorly fitted after the
ordinary mean continuation. Do not infer this from DEV alone.

Run one observation-only inference pass on all1152 original paired TRAIN
requests, from the unchanged safety_mean/last.pt. Seal all M8/H24 predictions
before opening geometry for this diagnostic. Reuse the existing exact checker
and signed segment clearance; report each candidate, goal/clearance/floor
errors, clearance deficit distribution, and violating segment counts. Keep
all128 families, current Qwen/cache/stride2/model settings, no updates, no
training subset search, no scoring/calibration data, no TEST_LOCKED.

Compare descriptive TRAIN frequencies with the already frozen DEV outputs.
Near-zero TRAIN collision with persistent DEV collision favors studying
generalization, whereas appreciable TRAIN collision keeps optimization or
representation mismatch plausible. Neither observation alone is causal proof,
and there is no automatic new-loss adoption gate. The previously rejected
gradient-matched maximum penalty and local/global refiner remain rejected.
This diagnostic cannot establish novel methodology. One job, timeout120s,
GPU1/35%/4threads under the remaining cumulative V3 budget.
