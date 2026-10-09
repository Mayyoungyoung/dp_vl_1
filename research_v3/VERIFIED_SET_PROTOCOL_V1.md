# Verified route-set mechanism: stages A/B, prospectively registered

2026-10-09. Status: implemented, not experimentally validated. The user explicitly
authorized **7200 additional experiment command seconds** in this chat. All server
preparation, tests, model loads, evaluations and failures use the separate
`runs/verified_set_v1/jobs` ledger. The earlier V3 budget remains unchanged.
Only wzy3090 / physical GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,
35% memory and four CPU threads. No locked payloads or metrics.

## Hypothesis and fixed scope

Current nearest-within-mode injective matching is already a strong ordinary
baseline. Test whether independently verified, distinct valid predictions should
stop fitting reference coordinates, and whether assigning remaining slots to
projected feasible regions adds value beyond simple gating and extra positives.
No novelty, improvement, complete support recovery or paper-readiness claim yet.
The task is the unchanged reach/end-effector envelope contract, not robot execution.

Original frequency_support TRAIN requests with at most8 original known modes are
eligible. Eligibility is determined before model inference and saved in prepared
manifest. Original and edited data, including runs/main, stay intact. All288 DEV
requests remain in evaluation. Original teacher modes are incomplete; a newly
verified mode can make the union exceed8, which is counted without discarding the
request. No claim of simultaneously retaining and covering an over-budget union.

## Target algorithm and controls

1. Retain the existing8x24 XYZ/events observation-only network, frozen Qwen cache,
   semantic grounding loss and mean continuous geometry penalty. All arms start
   from the same historical safety_mean checkpoint and use fresh identical AdamW.
2. TRAIN checker includes start5mm, nearest requested goal3cm, reach events,
   closed-segment/inflated-box2cm collision, and registered workspace floor.
   Compare the vectorized training implementation against existing independent
   check_candidates on all eligible references, explicit corruptions and fixed
   TRAIN model outputs. No training if any disagreement occurs.
3. Construct local coordinate boxes with half the exact segment clearance slack,
   capped at2cm, also bounded by floor slack; endpoints fixed. These certify only
   obstacle/floor clearance. Check task/event/mode again after target assignment;
   any failed/mode-changing projection falls back to its verified witness.
4. **ordinary** reproduces current finite-witness group matching and its gradient.
   **gate** uses exactly that assignment, then stops regression of one stable first
   valid candidate per distinct signature. **project** protects these candidates,
   assigns unprotected slots to uncovered groups and regresses to local projected
   targets. Remaining spare slots may duplicate an existing valid group.
5. **replay** receives original positives plus all8 actual project targets for the
   identical training draw, using ordinary group matching. Project stores the
   actual query-derived stream and hashes. This control shares acquired positive
   information, not the exact online target assignment; exact identical assigned
   targets would simply reproduce the proposed regression algebraically.

Projected and preserved targets are checked independently. Unknown classifier
outputs do not become invented distinct modes. No training or inference label
comes from learned q. q is held fixed with its complete observation encoder.

## Phase A

Precompute eligible TRAIN regions and verifier agreement. First64 eligible TRAIN
requests are fixed before inference. Save raw predictions and all three target
sets. Compare direct movement imposed on already-valid routes. Replace canonical
witnesses with their first preexisting checked perturbations at identical mode
count/exposure; duplicate canonical witnesses as an invariance sanity check.
This diagnoses target pressure, not causal optimizer damage or generalization.
If most eligible references have zero region slack or no valid predictions exist,
inspect before committing to the full training queue.

## Phase B and decision

Fixed1200 steps, batch32, lr.0003, AdamW weight_decay.0001, clip1, raw8 and returned4.
Shared seeded request sampler; full optimizer/scheduler/Python/NumPy/Torch/CUDA RNG,
loss RNG, actual sample hashes, counters and replay state checkpointed every100.
Use seed0 initially. Gate and project consume the same group RNG as ordinary.
Replay may have extra mode groups and therefore different loss RNG evolution;
actual request stream and additional reference processing are separately recorded.
No checkpoint selection from DEV: fixed last1200 for every completed arm.

Evaluate raw8 validity and distinct valid modes on all288 DEV requests, plus
predeclared original-known-mode<=8 subset. Keep fixed complete paired-domain q;
no q retraining or calibration changes. Report validity, distinct, known/rare
recall, invalids and duplicates; no cherry-picked unreferenced-only improvement.
Compare to both same-update ordinary and untouched historical safety_mean.

Engineering continuation gate: distinct gain>=.3 per request and validity
difference>=-.01 versus strongest simple control. Improvement needs additional
paired continuation seeds1/2 and same-oracle replay before mechanism claims.
These are continuation seeds from a shared historical seed0 initializer, not
independently pretrained backbones. If only the gate works, retain the simpler
control. If extra-positive replay explains the gain, reject target novelty.
No large data collection, LoRA or RFT until this evidence is read.

Every launch must use a fresh immutable commit export, actual source hashes,
frozen launcher, new job id and explicit per-command timeout within the cumulative
7200s ledger. Partial/failed work remains evidence. No duplicate completed jobs.
