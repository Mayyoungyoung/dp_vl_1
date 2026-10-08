# Optimizer-state diagnostic, registered before running

The eight completed anchor scratch steps have exact common forwards. Full
gradient differences are 0.70–2.07%, update differences 3.21–8.16%; update
cosines >=0.99667. Hard derivatives do not consistently improve the post-step
TRAIN objective. Both modes change 30–32/32 peak identities, with maximum
path-coordinate changes 0.165–0.326m and increased grounding/regression/
clearance losses. These discrete peak changes include harmless nearby pixels;
counts alone do not imply semantic failures. Do not launch the long hard-anchor
pair on this evidence.

Next isolate the ordinary optimizer-reset effect. Use the exact same original
safety_mean tensors, four seed0 TRAIN batches and assignment RNG states.
Keep straight_through_peak in both branches. Independently reset each batch
and compare fresh AdamW versus its saved step1200 moments and time index.
Keep lr3e-4, weight decay1e-4, clip1 and the full objective unchanged. Load no
DEV_MODEL or scorer payload. Record the same eight scratch checkpoints and
all predictions, with explicit optimizer branch and actual anchor mode.

Assert restored states have step1200 and identical hyperparameters; common
forwards and full gradients must agree (optimizer acts only after backward).
Check fresh-branch outputs against the completed prior scratch run. Report all
four batches, objective components, actual update norm/direction and peak/path
changes. Do not tune learning rate or select a favorable batch. Cap120s within
the existing7200s cumulative command budget, GPU1/35%/four CPU threads.

This tests a first-step continuation confound, not long-run benefit, novelty,
or failure of Adam. No automatic long-training gate. A long continuation, if
justified by the whole result, needs a separate fixed-budget protocol and
comparison with the existing fresh-optimizer control.

Post-run access-scope clarification (2026-10-09): the shared generator loader
materializes permitted historical/paired DEV caches as well as TRAIN. All
diagnostic index selection, loss, gradients and updates use the explicit1152
TRAIN index subset only; no DEV-driven choice or optimizer update. Earlier
no-DEV-access wording is corrected, not retrospectively redefined. No locked
TEST or scorer-role payload is loaded. Frozen executed source is preserved.
