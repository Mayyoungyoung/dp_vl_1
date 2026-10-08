# Read-only shared-anchor derivative diagnostic

Motivation: the linear penalty reduces TRAIN collisions but increases semantic
endpoint errors, while DEV collision improvement is small. This co-occurrence
is not proof of gradient conflict. A specific implementation fact is testable:
ObservedGeometryEncoder uses a hard observed attention peak in its forward,
but a soft expectation derivative for anchor_xyz in its backward. The anchor
also enters context fusion and every route's reference line/endpoint.

Next probe, no optimizer updates: use the unchanged safety_mean checkpoint and
the same four fixed32-request TRAIN batches (seed0) as the gradient diagnostic.
Compare current straight-through mode with a diagnostic hard-peak mode whose
forward anchor is exactly identical but whose anchor derivative is zero. Keep
the attention-weighted context derivative and grounding loss intact. First
verify identical forward paths, events and anchors within the same precision.

A positive scaling of attention logits preserves their argmax. Away from the
existing scale clamp, perturb only log_attention_scale by +/-1e-3 and +/-1e-4,
restoring the original parameter every time, and compare central differences
of160*mean(clearance_deficit_squared) against each mode's analytic derivative.
Use float64 for this numerical diagnostic and document any forward deviation
from the original float32 outputs; do not present these as new route results.
Save per-batch loss, derivatives, anchor identities, actual parameter value,
step sizes, numerical agreement and full parameter-gradient norms/directions.
If the scale is clamped, report the clamp and do not silently choose another
parameter or perturbation. If finite differences are numerically unstable,
report inconclusive rather than selecting a favorable epsilon.

The derivative mismatch, if verified, is not by itself a causal failure result:
straight-through estimators intentionally use surrogate gradients. No automatic
training/adoption gate follows from this probe, and no new-method claim follows
from stop-gradient. It may motivate a separately registered same-budget ordinary
ablation only if the mismatch is material and relevant to task/route coupling.
No inference input may include boxes/goals; geometry is used only to measure
the TRAIN loss. No DEV, score, calibration or locked-test role is used by the
probe. Proposed cap120s, GPU1/35%/4threads, inside remaining V3 command budget.
This protocol is registered; implementation and execution are still pending.

Post-run access-scope clarification (2026-10-09): the shared generator loader
materializes permitted historical/paired DEV caches as well as TRAIN. All
diagnostic index selection, loss, gradients and updates use the explicit1152
TRAIN index subset only; no DEV-driven choice or optimizer update. Earlier
no-DEV-access wording is corrected, not retrospectively redefined. No locked
TEST or scorer-role payload is loaded. Frozen executed source is preserved.
