# Recheck edited-scene retention at the current strong baselines

The earlier diagnostic used initial/frequency stage1 models. It found surviving
concrete paths whose modes the edited-scene generator did not retain. It also
noted that substitutions and M8 capacity can explain some losses. The later
safety_mean and ordinary continuation controls have not received this same
audit, so the old weak-stage result cannot establish their bottleneck.

Use only sealed paths/events/labels for safety_mean, safety_worst, margin_mean,
optimizer_restored and frequency_set_canonical on all32 DEV_MODEL families,
96 target pairs per direction: open->closed, open->shifted and both reverses.
Retain the exact existing checker, mode definition and start/goal equality
assertions. Recheck the same source path in destination geometry, with no new
model forward, training, scoring or threshold tuning. No transferred q values.

For each pair count lost modes with a source-valid concrete path that remains
valid after the edit. Then count destination slots occupied by invalid paths
or duplicates of a classified valid mode. Valid unclassified paths consume
slots and are not treated as duplicates. The minimum of lost-mode count and
these slots is a constructive oracle opportunity: copy one surviving source
path per lost mode into a redundant/invalid slot, preserving all existing
classified destination modes within M8. This measures a feasible opportunity,
not a deployable algorithm, inferred q, or guaranteed learned recovery.

Report all directions/models, losses, slot-feasible counts and concrete
adaptation gain. This is exploratory failure attribution, not a new primary
performance gate. No claim that losing any individual source path proves mode
absence, and no claim that every mode substitution is a defect. No new paired
training follows automatically. Cap120s plus tests60s within remaining1355.266225
command seconds; immutable source, existing GPU1/35%/four-thread wrapper (CPU
geometry work), no TEST_LOCKED or raw collector traversal.

## Follow-up decomposition, before inspecting destination semantic subgroups

The completed audit leaves50/496 surviving modes unretained for safety_mean
open->closed,46 slot-feasible. This does not yet isolate mode keeping from
shared semantic failures. Using only the sealed local rows, report all five
models and four directions split by whether every destination endpoint is
semantically wrong. Additionally count replaceable slots that already have a
correct semantic endpoint but fail another validity check, plus duplicated
classified valid modes. min(lost modes, these slots) is a stricter constructive
opportunity that does not rely on repairing a wrong endpoint. Valid unknown
mode slots still consume capacity. Report both decompositions; no subgroup
selection, new training, detector or q claim. Family-bootstrap intervals for
mean available opportunities are descriptive, not a method comparison gate.

Second local decomposition, before reference-overlap inspection: compare each
lost mode with the immutable destination witness-mode set. Count losses inside
and outside that incomplete reference set, and each one's separate feasible
opportunity using correct-endpoint slots. These two opportunity counts can
compete for the same slots and must not be summed. A mode outside references
is not invalid: it already has a checked positive path witness. This separates
missing-support explanations from failure to cover already-supervised modes.
Keep the previous semantic-only decomposition unchanged as v1.
