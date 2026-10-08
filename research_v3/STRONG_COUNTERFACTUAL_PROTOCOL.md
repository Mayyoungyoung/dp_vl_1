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
