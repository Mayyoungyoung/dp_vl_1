# Ordinary optimizer-state continuation control

Prospective registration after the four TRAIN scratch pairs completed, before
long training or DEV evaluation. Initial gradients are exactly equal and fresh
predictions replay the previous probe exactly. Restored step1200 moments reduce
initial update norms and path-coordinate jumps across all four batches. This
supports testing a continuation confound, not a new algorithm or generalization.

Train one restored-optimizer seed0 arm for exactly1200 additional updates from
the same original safety_mean checkpoint as completed margin_mean. Keep all
model tensors, straight-through anchor, full set objective, lr3e-4, decay1e-4,
clip1, batch32, support and input/assignment RNG streams unchanged. Only restore
the checkpoint's AdamW moments and step1200 counter, verifying their tensor
digest and hyperparameters. Reset sampler/RNG just as margin_mean did; this
isolates optimizer state, and is not exact chronological resumption. Fixed last
checkpoint only; no sweep, early DEV selection or repeated seed0 restarts.

Reuse the completed margin_mean1200 as the fresh control. Its source is84d7e4d;
the new source has an opt-in restoration branch, default behavior unchanged.
The scratch fresh branch already exactly replays. Verify complete common parent,
initial model digest, actual sampler audit, mode exposure and unique reference
counts, final optimizer steps1200 vs2400 and unchanged hyperparameters.
No claim of simultaneous runs or common artifact hashes across source versions.

Evaluate once on all288 DEV_MODEL requests using the same fixed complete paired
q seed0 bundle. Preserve candidate arrays/events/labels and all metrics. Compare
restored-minus-fresh and restored-minus-parent with the existing10000-resample
family bootstrap. For an ordinary-baseline advancement require validity gain
>=.02 with lower95%CI>0 versus fresh, distinct-mode gain>=.15, rare recall
>=-.02, any-valid>=-.01 and Brier increase<=.01 (same gate as linear control).
Also report full1152 TRAIN geometric/semantic residual counts for both controls;
reuse previously closed fresh audit. Even a positive gate is not novelty.

Budget: training cap360s, DEV60s, TRAIN60s, tests/analysis60s each, serial under
the remaining2008.522719 command seconds of the existing7200s cap. GPU1,
35%memory/four threads, immutable source/launcher, all receipts and failures
retained. No score/calibration fitting, TEST_LOCKED or raw collector traversal.
