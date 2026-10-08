# Next action

All V3 jobs are CLOSED. Latest ordinary linear-margin pair fails its gate;
no new coefficient, margin or learning-rate sweep. Read linear_v1/REPORT.md.
110 jobs106success4retained failures;5091.294504/7200 command seconds used,
2108.705496 remain. Source84d7e4d and629043c queues finished; never duplicate.
All252 closure files locally SHA verified, manifest LINEAR_CLOSURE_ARTIFACTS_20261009.json.

Keep safety_mean as the current reference generator and all three new-domain
q seeds in reporting. Extra quadratic updates provide no clear joint benefit;
linear reduces TRAIN collisions but raises endpoint errors and fails on DEV.
The shared-failure q analysis and figure are descriptive, not a novel detector.

Next safe action is to implement and run the prospectively registered
ANCHOR_GRADIENT_DIAGNOSTIC_PROTOCOL.md. It compares the same hard forward anchor
with straight-through versus zero anchor derivative, keeping genuine context
attention gradients, then checks log_attention_scale central differences on
four fixed TRAIN batches. First require exact same forward outputs within
precision; report float64-vs-original deviations. No updates, no DEV scoring,
no automatic training gate. If the scalar is clamped or numerical finite
checks inconclusive, retain that result without adaptive parameter selection.

A surrogate derivative is intentional and has substantial prior art. Its
mismatch is not proof of causal harm; only a separately registered matched
ablation could support a consequence. Do not rebrand stop-gradient as novelty.
Core Gate B/C/D remain unproven; goal ACTIVE, no verified external blocker.
