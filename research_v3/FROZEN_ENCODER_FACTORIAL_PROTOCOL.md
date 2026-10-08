# Ordinary frozen-input-encoder factorial control

Prospective registration after the verified-positive gate failed, before new
training or DEV outputs. TRAIN semantic errors rise75->187 with augmentation
despite preserved grounding labels; DEV rises106->175. This is consistent with
shared-input-representation drift but does not identify its sole cause.

Complete a 2x2 ordinary control: existing unfrozen margin_mean and
verified_edit_augmented, plus NEW frozen_encoder_plain and frozen_encoder_augmented.
Both new arms start from exact safety_mean checkpoint, fresh AdamW, seed0,
1200steps, lr3e-4, decay1e-4, clip1, quadratic clearance160, grounding.02,
same1152TRAIN requests and original ground labels. Plain uses original support;
augmented uses the existing SHA-verified same-mode support without recollection.
No weight/lr/step sweep, new parameters, paired loss or q fit.

Freeze geometry, head.feature_encoder and head.state_encoder parameters and
buffers, force these modules to eval after model.train, optimize only remaining
parameters. Preserve exact initial/final frozen tensor digests and parameter
counts; test actual forward anchor/context invariance through optimizer steps.
These encoders contain no BatchNorm or Dropout in current source. Frozen
attention auxiliary loss is constant with respect to remaining trainable
parameters; retain its computation/value and explicitly record this consequence.
Freezing does NOT guarantee fixed final endpoints: bounded route residuals train.
Same-update is not equal backward compute; report measured time and target slots.

Evaluate fixed final1200 on all288 DEV_MODEL requests using the same COMPLETE
paired-domain q seed0. Reuse the fixed safety_mean witnesses (2169 opportunities)
for every destination model. All4 directions,32-family bootstrap10000seed610093.
Primary augmentation contrast is frozen_augmented minus frozen_plain, with the
unchanged augmentation gate: retention >=+.05 and lowerCI>0, validity>=-.01,
distinct>=-.05, rare>=-.01, any-valid>=-.01, Brier<=+.01. Report existing unfrozen
contrast and interaction (frozen augmentation delta minus unfrozen delta), not
just frozen_augmented versus parent. Interaction interval uses same family draws.
Always contextualize all4arms against untouched parent. No causal sole-mechanism
claim or generator replication from this seed0 factorial.

Separately register plain-freezing ordinary baseline gate versus parent:
validity>=+.02 with positive lowerCI, distinct>=+.15, rare>=-.02,
any-valid>=-.01, Brier<=+.01. Augmentation/freezing remains ordinary even if a gate
passes; no automatic claim of novelty or Gate B/C/D. No new defaults automatically.

Caps: tests30s, each train360s, each eval30s, analysis30s =840command seconds,
within976.768865 remaining of7200. Wrapper also enforces cumulative limit.
GPU1 UUID7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,35%memory,fourCPUthreads,serial.
Immutable commit export/launcher; keep actual source hashes, commands, failures,
optimizer/scheduler/RNG/sampler checkpoints. Read only permitted explicit splits;
shared loader materializes permitted DEV caches but only TRAIN enters updates.
No locked TEST, score-role changes, raw collector traversal or environment edits.
