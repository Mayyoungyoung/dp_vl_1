# Original-witness set-matching control

Register before implementation execution or inspecting its DEV predictions.
The full-set arm improves over an eight-reference distinct-mode sampler, but
uses five checked variants per original witness and more target processing.
This comparison has not separated mode-covering assignment from the richer
within-mode reference set. It is a data/compute attribution gap, not evidence
for a new mechanism. No further anchor/optimizer tuning is planned.

Train one seed0 `set_canonical` arm from the identical historical R1_seed0
checkpoint, fresh AdamW, same1200 draws/batch32/input RNG/loss group RNG, lr3e-4,
decay1e-4/clip1,8candidates, grounding and quadratic geometry objective as the
completed frequency_set_matching arm. Use only the original path in each
prepare()-verified witness block (one original then four variants). Keep all
witness modes, including the same randomized eight-of-nine mode subset when
necessary. Do not replace set matching by independent best-of-many fitting.
Validate block structure; record actual route indices/exposures. Same inputs,
fewer target labels and less reference computation by design; do not claim
equal target-processing cost. This is not a fresh VLM training run.

Compare once to the sealed full-set arm and set_sampled arm on all288
DEV_MODEL requests with the original complete fixed q (evaluation_fixed_q_v2),
since those are the already-closed stage1 baselines. No new q fit or calibration.
Verify same initial parameter/sampler audit, source parent and final1200 steps;
canonical unique routes and per-mode exposure must be exactly one fifth of
full-set. Canonical keeps the same known-mode group permutation stream.

Report all raw validity, distinct modes, rare recall, any-valid, q and selected
metrics, family-bootstrap10000 CIs, actual target slots and wall time. Primary
contrast full-set minus canonical: evidence of a useful within-mode augmentation
effect requires rare recall>=.02 with lower95%CI>0 and validity loss<=.02.
Otherwise report no demonstrated benefit at this resolution, not equivalence.
This is attribution of an ordinary baseline, never Gate B/C novelty evidence.

Keep fixed final1200; no reference-count, perturbation-amplitude or step sweep.
Training cap180s, evaluation60s, tests/analysis60s each under the1719.564064
remaining7200-budget command seconds. GPU1/35%/4threads. Immutable source and
launcher, full receipts/checkpoints/RNG. TRAIN and DEV_MODEL only; no score
role fitting, TEST_LOCKED payload or entire raw collector use.
