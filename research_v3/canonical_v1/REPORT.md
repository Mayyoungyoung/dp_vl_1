# Within-mode augmentation is not necessary for the raw coverage gain

The prospective augmentation gate is **false**. Original-witness set matching
matches full-reference raw validity and has higher rare-mode recall; full
references give better selected-set quality and fixed-q Brier. Neither dominates
all objectives. Keep the existing safety_mean deployment reference.

This corrects the earlier inference that the full-set versus eight-sampled
comparison demonstrated the value of richer within-mode references. That
comparison changed both the reference set and assignment. The new ablation
keeps complete group matching and removes only the four variants per witness.

| Stage1 ordinary method | Target slots | Valid@8 | Distinct@8 | Rare recall@8 | Distinct@4 | Brier |
|---|---:|---:|---:|---:|---:|---:|
|Eight sampled references|307200|76.8663%|6.14931|57.6852%|3.70833|.167908|
|Original witnesses, full matching|313326|82.2049%|6.55208|66.3657%|3.66319|.168060|
|Witnesses plus four variants, full matching|1566630|82.2049%|6.56597|64.3576%|3.72569|.152315|

Full minus original: rare recall-2.0081pp, family95%CI[-3.6227,-.3037]; raw
validity0pp[-1.2153,1.3889]; distinct+.01389[-.07986,.12153]. Full improves
selected distinct modes+.0625[.01389,.13194], selected all-valid+2.7778pp
[.3472,5.2083] and Brier-.015745[-.030390,-.004292]. Do not discard these
unfavorable original-witness metrics or claim equivalence from a nonsignificant
validity difference. Both have exactly1894 valid candidates among2304, but
their predictions/validity identities are not claimed identical.

Original versus sampled: validity+5.3385pp[2.8646,7.9427], distinct+.40278
[.19792,.61458], rare+8.6806pp[6.2152,11.2182]. This is consistent with an
ordinary complete-group matching benefit without5x within-mode reference
expansion. It does not separately identify every assignment/sampling effect.

All use the same original complete fixed q, unlike the later paired-domain q
optimizer controls. Brier is measured on each generator's own candidate pool,
not evidence of an improved scorer on one fixed pool. All intervals use32
DEV_MODEL families and one generator seed, not independent final-test evidence.

## Actual execution and fairness

First source `14da6aa0702870842664fc78f76bb2a67ba02341`, archive SHA
`c57d3d756de915ff5e1373391c92af2c63a4af3c1c772defa98c357d5ea5e183`.
8 tests pass in1.57s. Initial train hit the registered180s timeout, exit124,
wrapper180.323454s; failed receipt, stdout and step600 recovery are preserved.
This was an underestimated execution cap, not nonfinite loss or invalid labels.

Resume source `681579246946984e52ece367773fc85a22ebd44b`, archive SHA
`defcf14a364c2455a7831f7c182a80243687f5337b562e93f3c2657199711923`.
All scripts/routeset/configs bytes equal the first source. Recovery checkpoint
SHA `d82c0b2ab2a939d1ebdc16224434fa5d2310b57a304613b32f0827bf42356d09`
was copied before resume. Strict settings assertion and full optimizer/scheduler/
RNG/sampler/history restoration ran; first resumed logged step700, final1200.
Resume train exit0 in164.283434s, DEV13.975256s, analysis3.527904s.

Total training-command time **344.606889s**, including failed work and reloading;
the saved summary's117.050803s covers only the resumed loop. Do not compare that
partial figure with the full arm's233.951663s complete loop or claim5x speedup.
Per-request matching overhead remains and the runs were not simultaneous.

Initial model, parent, input stream, support and final matching-group RNG are
exactly equal. Original-witness unique TRAIN references9408 versus47040 full;
per-mode exposure and total slots are exactly1:5. These are correlated path
labels, not independent layouts. Both have128 TRAIN families,38400 input draws
and fixed1200 updates; no hyperparameter or checkpoint selection changed.

268 closure files locally SHA verified, including all124 receipts (119success,
5retained failures). Archive and source identities are in
[CANONICAL_CLOSURE_ARTIFACTS_20261009.json](../CANONICAL_CLOSURE_ARTIFACTS_20261009.json).
Frozen checkpoints, predictions, RNG and the failed attempt remain recoverable.
