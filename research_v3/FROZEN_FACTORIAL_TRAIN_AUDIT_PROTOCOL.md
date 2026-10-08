# Final-state TRAIN error decomposition of the frozen factorial

Both pre-registered gates failed. Frozen plain validity81.64% and frozen
augmented83.46% remain below parent86.81%. DEV collisions295/285 versus
parent207; wrong endpoints151/115 versus parent108, although all frozen input
tensors are byte-identical. Frozen augmented versus plain fixed-witness
retention improves2.7663pp[.8182,4.9935], below5pp; the interaction interval
crosses zero. Do not tune thresholds, learning rates or freeze subsets.

Run the existing full1152TRAIN final-state audit for both frozen models. No
model/scorer update; seal all predictions before controlled oracle geometry.
Report the same collision/semantic/floor/shared-failure counts as all three
prior arms. This distinguishes TRAIN fitting loss from DEV generalization and
tests whether endpoint drift remains despite unchanged input encoders. It
does not establish causality of any specific decoder parameter group.

Two jobs capped60s each within428.750523 remaining command seconds. Same
immutable export, source hashes, GPU1/35%/4threads and serial cumulative7200s
wrapper. Only explicit TRAIN records; no TEST_LOCKED, SCORE or CALIBRATION
payload. Retain all requests, checkpoints, RNG, failures and actual receipts.
This audit does not authorize a new training sweep or default replacement.
