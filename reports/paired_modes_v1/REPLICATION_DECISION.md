# Three-seed diagnosis decision

All nine original generator trainings and evaluations completed. R2 minus R1 averaged across seeds: validity -3.805pp, modes +0.222, reference coverage -2.198pp, shared relation recall +0.083pp. Seed2 reverses the shared-relation advantage and loses11.111pp candidate validity. The original seed0 result is not a stable mechanism win. See three_seed_extended/RESULTS.json, including parent-family intervals and all failures.

The prearranged q and full-set control queue remains unchanged. Prepare only a read-only diagnosis after it closes: same four TRAIN batches, each final R1/R2 seed; compare partial-pair gradient norm/direction against the exact R1 objective, clearance and relation terms. Measure how many witnessed shared classes have an actually relation-satisfying predicted candidate on both sides. The existing nearest-band rule always pairs a class even when neither prediction has reached that band; this is a concrete possible failure mechanism, not yet a measured conclusion.

No optimizer updates, DEV gradients, new hyperparameter sweep, checkpoint selection or relabeling are authorized by this diagnostic. Its outcome will decide whether one of the two bounded mechanism revisions is warranted. TEST_LOCKED remains unopened.

Post-seed0 relation-band sensitivity is retained separately. It exactly reproduces primary labels before expanding classification of the top clearance band; primary validity and outputs remain unchanged.

## R3 seed0 negative; one more diagnosis, not an automatic revision

R3 seed0 valid62.6302%, modes3.72222, shared24.8016%; the individual-seed validity condition already fails. All three planned seeds still complete. Goal errors428/2304 versus R1 seed0 172/2304 accompany the drop; collisions/workspace failures566/2304 also remain. This motivates a distinct read-only diagnosis after all scheduled jobs close: on the same four TRAIN batches and each final R1/R2/R3 checkpoint, measure pair-gradient direction against endpoint error and grounding, partitioned into geometry/anchor module and route head. No optimizer update or DEV gradient. A possible shared semantic-geometry gradient interaction is a hypothesis, not an established cause.

Only consistent, substantial TRAIN evidence of pair pressure conflicting with semantic grounding would justify considering the second permitted mechanism revision. Mixed or supportive gradients should close this proposed explanation; do not invent a gradient-conflict fix from the DEV failure alone. No second revision is implemented or scheduled by this note.
