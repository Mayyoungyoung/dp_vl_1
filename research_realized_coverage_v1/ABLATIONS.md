# Mechanism ablations and their limits

Actual seed0 final-2400 results unless marked otherwise. Complete metrics and
fixed witnesses are in results/RESULTS_TABLES.md.

|Controlled change|Control U8|Variant U8|Interpretation|
|---|---:|---:|---|
|C adaptive → ordinary realized-success proposal|6.7639|6.9826|Ordinary allocation improvement, unchanged decoder|
|Initial → shared success-context feedback, success head|6.9583|6.9826|Small gain from deployment-context feedback|
|Success → scalar net, same two-context pool|6.9826|6.7743|Aggregate net regression loses|
|Scalar net → dense actual outcomes|6.7743|6.8854|Helps this screen, not the strong control|
|No-peer dense → peer-aware dense|6.9444|6.8854|No evidence for peer benefit|
|Scalar net → added-only target|6.7743|6.9201|Net subtraction is not an empirical win|
|Frozen success → gap geometry + refreshed dense|6.9826|6.9826|V4 falls 94.358% → 93.663%|

Added-only preserves architecture, quality targets, feedback, steps and seed;
only U8 ignores lost words. It beats the weak scalar-net fit, so accounting for
losses is not established as the effective innovation. Both lose to success.

Peer/no-peer receive the same 17-way labels and data. Attention is replaced with
a similarly sized tokenwise MLP; set-level quality pooling remains in both.
Five registered head seeds give dense U8=6.9465 versus success=6.9806, with no
U8 win. These are fixed-C head seeds, not geometry seeds.

Ordinary, KL-only, gap and hard arms remove displacement and share positives,
start, learning rate and budget. Hard is the failure-sampling control. Every
600/1800/3600 curve is saved. Branch-specific RNG differs; no exact-stream causal
or multiseed geometry claim. Snapshot feedback is recollected after changing
each generator and both proposal types are refitted, avoiding stale labels.

Fixed-C ordinary/adaptive/balanced sampling was already tested in the preceding
mode_geometry study and is reused. New success and set proposals also use the
same fixed C, isolating proposal computation from geometry training. Dense
token search takes extra work, with no demonstrated returned-quality gain.

All 10168 TRAIN reference means are valid and preserve the intended word.
This rejects the measured averaging-conflict explanation. Optional nearest-
positive repair was never launched and is not counted. New-scene and full-arm
studies were not expanded after the mechanism failed these controls.
