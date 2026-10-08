# Saved optimizer state does not establish a long-run advantage

The registered ordinary-baseline gate is **false**. Restoring AdamW moments and
time index eliminates the large first-step objective increase in four fixed
TRAIN batches, but the fixed1200-step comparison provides no clear DEV validity
or distinct-mode advantage. Keep the original safety_mean reference. Do not
sweep learning rates or claim that the first-step effect explains prior failures.

## Matched continuation

Source `4b259e6b63b4707a2938fc6f657437ea4d4219d5`; source archive SHA
`8ac5a208f29370ed2fd398e2c49780cd64bd42f6292806d7783c0a07e63292d0`.
Five coordinator jobs exit0 in289s total:11 tests pass in2.19s; train wrapper
249.134969s; DEV12.214039s; full TRAIN21.817580s; analysis2.883232s.
Same parent, initial parameter digest, actual sampler audit, reference mode
exposure and unique references verified. Final Adam counters are1200(fresh)
and2400(restored). Model updates in this comparison are1200 per arm.

Fresh margin_mean is reused from the sealed prior control(source84d7e4d), not
rerun or claimed simultaneous. Saved-state initialization is exact at tensor
level. The sampler is deliberately reset in both arms; this is not chronological
resumption. Objective, lr3e-4/decay1e-4/clip1,8candidates and frozen complete
paired-domain q seed0 are the same. No score training/calibration is performed.

| DEV metric | Fresh AdamW | Saved AdamW | Saved minus fresh, family95%CI |
|---|---:|---:|---|
|Valid fraction|87.2830%|87.4566%|+0.1736pp [-2.5174,2.9948]|
|Distinct valid modes@8|6.67014|6.67361|+0.00347 [-0.21181,0.22569]|
|Rare recall@8|73.8426%|74.7020%|+0.8594pp [-1.8345,3.7326]|
|Any valid request|96.1806%|95.4861%|-0.6944pp [-2.7778,1.3889]|
|Brier|.077099|.078324|+.001225 [-.011040,.014951]|

Versus original safety_mean, restored validity is +.6510pp[-1.1719,2.2569],
distinct modes -.00694[-.15278,.12153], and Brier+.002934. This also fails to
establish a better overall baseline. These are32-family development intervals
conditioned on a single trained generator seed, not training-seed uncertainty
or independent final-test results. All metrics are in RESULTS.json.

## TRAIN fitting differs from DEV benefit

Full1152 TRAIN predictions were sealed before checker geometry was opened.
Both use9216 candidates; all floor checks pass.

| TRAIN count | Fresh | Saved |
|---|---:|---:|
|Valid candidates|8883|9012|
|Post-collision candidates|261|103|
|Wrong semantic endpoints|75|101|
|Requests with any valid|1146|1146|
|All endpoints wrong|6|6|

Saved moments materially change fitting but do not remove the semantic/generalization
bottleneck. Reduced TRAIN collision is not sufficient evidence of a new mechanism.
On DEV the collision count falls200->179, semantic errors rise106->117,
their intersection falls13->7, and all-endpoint failures rise11->13. This yields
only four additional valid candidates among2304; all failures are retained.
The saved final checkpoint SHA is
`038588f52247a32e20c693fc1fb6c94f8c0f991c22dae6b27e61de15547ab092`.
All checkpoints, optimizer/RNG states, predictions and actual job receipts are
retained. No TEST_LOCKED payload was opened, and no default inference mode changes.

The comparison closes the optimizer-reset explanation for this registered
continuation. It does not prove equivalence of all optimizers or learning rates.
