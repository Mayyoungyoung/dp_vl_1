# Complete-objective scratch update probe

The registered derivative probe completed: identical fp64 forwards, stable
finite differences, and hard-anchor derivatives agree in all four batches.
The straight-through collision gradients differ by1.05–6.83% of full norm,
cosines>=0.99767. This confirms a surrogate, not causal harm. Adam normalizes
coordinates, and regression also uses the anchor; collision gradient norm
alone is insufficient to decide on a long ablation.

Use the same four seed0 TRAIN batches and original safety_mean checkpoint.
For each batch independently reset two scratch models and fresh AdamW states,
matching lr3e-4, weight decay1e-4, clip norm1 and actual full ordinary objective
(set regression +.02 grounding +160 squared clearance/floor). Use identical
reference-assignment RNG state in both branches. One branch retains the
straight-through anchor, one uses forward-identical hard_peak. Exactly one
optimizer update per scratch model; never carry a branch into another batch.

Record component losses and complete gradients at the common start, gradient
and actual parameter-update norm/cosine, changed anchor identities, before/after
paths/events, and TRAIN loss/components after the update using the same
assignment RNG state. Save all eight step1 model/optimizer/RNG checkpoints and
unfiltered predictions. Verify initial forwards exactly equal and the original
checkpoint/model untouched. No DEV or scorer role, no selection of a favorable
batch, no automatic new-method claim or adoption gate. This is a short training
diagnostic, distinct from the previous read-only derivative probe. Cap120s under
existing GPU1/35%/4thread budget; decide from the whole result whether a full
matched estimator ablation is justified.

Post-run access-scope clarification (2026-10-09): the shared generator loader
materializes permitted historical/paired DEV caches as well as TRAIN. All
diagnostic index selection, loss, gradients and updates use the explicit1152
TRAIN index subset only; no DEV-driven choice or optimizer update. Earlier
no-DEV-access wording is corrected, not retrospectively redefined. No locked
TEST or scorer-role payload is loaded. Frozen executed source is preserved.
