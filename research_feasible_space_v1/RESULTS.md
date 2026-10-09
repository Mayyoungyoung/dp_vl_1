# Feasible-space v1 results

**The implemented bounded decoder does not pass the preregistered mechanism gate; publication readiness is not established.**

The final three paired generator continuations show a small benefit over same-information XYZ, but a clear deficit to returning predicted centerlines. Two measured-cause repairs improved region feasibility; neither established a necessary neural interior generator. Historical defaults remain unchanged.

## Scope and fixed protocol

TRAIN: 1,152 requests / 128 families / 56,920 verified positive paths. Existing DEV_MODEL: 288 requests / 32 repeatedly reused families. Seeds 0–2 are full generator continuations of shared pretrained C0, with exactly matched initialization and sampled streams within each pair. They are not independent VLM/pretraining runs. Seed 0 was reused from exploratory screening; the replication lock preceded reading its final outcome, not its launch.

Each final generator receives 2,400 updates and a matching ordinary success head trained for 2,400 updates on fresh TRAIN outcomes. Center/projection controls receive their own hashed decoder views, fresh feedback and heads. Exactly eight candidates are decoded, followed by the complete original frozen scorer returning four. Truth geometry is used only for TRAIN supervision and independent checking.

The unchanged practical gate requires ≥0.15 additional actual valid modes@8 with a positive paired interval against all matched controls; guards require ΔV8≥−0.01, ΔV4≥−0.005 and ΔU4≥−0.03. Conditional 10,000-draw bootstrap intervals resample seed and scene family. They do not establish untouched-test generalization.

## Final existing-DEV comparison

|Method, mean of three generator continuations|V8|U8|V4|U4|Known recall|
|---|---:|---:|---:|---:|---:|
|Same-information free XYZ|85.89%|6.7176|92.88%|3.7130|78.85%|
|Proposed bounded mapping|87.01%|6.8032|93.20%|3.7280|79.44%|
|Projection, fresh head|86.37%|6.7523|93.08%|3.7222|79.08%|
|XYZ center, fresh head|90.93%|7.1100|93.69%|3.7477|82.64%|
|Bounded center, fresh head|90.99%|7.1100|93.75%|3.7500|82.48%|

Historical references use different training protocols and are not paired causal controls:

|Historical method|V8|U8|V4|U4|
|---|---:|---:|---:|---:|
|C + ordinary success, five head seeds on C0|89.15%|6.9806|94.34%|3.7736|
|Gate, three historical seeds|92.12%|7.2380|94.33%|3.7720|

Bounded mapping remains below both historical references on raw coverage and returned quality. Even the stronger center-only control does not dominate Gate or recover the historical C return quality.

|Bounded minus control|ΔU8|Crossed 95% CI|Practical gain passes|Quality guards pass|
|---|---:|---|---|---|
|Same-information free XYZ|+0.0856|[+0.0336, +0.1424]|False|True|
|Projection, fresh head|+0.0509|[+0.0023, +0.1042]|False|True|
|XYZ center, fresh head|-0.3067|[-0.4259, -0.2014]|False|False|
|Bounded center, fresh head|-0.3067|[-0.4225, -0.2014]|False|False|

**Overall accepted: `False`.** A statistically positive smaller effect cannot replace the +0.15 practical gate.

## Every generator seed

|Method|Seed 0 U8|Seed 1 U8|Seed 2 U8|
|---|---:|---:|---:|
|Same-information free XYZ|6.7986|6.6701|6.6840|
|Proposed bounded mapping|6.8542|6.7847|6.7708|
|Projection, fresh head|6.8368|6.6979|6.7222|
|XYZ center, fresh head|7.1493|7.0903|7.0903|
|Bounded center, fresh head|7.1493|7.1111|7.0694|

All three XYZ/bounded initial-tensor hashes and actual training-stream digests match within seed. No seed extension or checkpoint selection followed these results.

## Reference representation and oracle diagnosis

All 56,920 TRAIN witnesses admit individual geometric certificates. Two per-word prototype corridors cover 51,211 / 56,920 witnesses (89.9701%); all 20,336 prototype centerlines are task-valid and correct-word. No medoid fallback was needed. This establishes representation capacity under oracle TRAIN geometry, not deployable perception.

In the 1,024-route TRAIN reference-cell diagnostic, center interpolation, free XYZ and bounded mapping all reach 100% validity. Correct requested words are 1,024, 1,022 and 1,023 respectively. The oracle experiment does not demonstrate a learned-interior advantage; its centerline is already a verified feasible witness.

## Fixed cross-edit witnesses

|Model|Retained / 2169|Retention|New losses / 1872|Recovered / 297|Same-mode repair / 270|
|---|---:|---:|---:|---:|---:|
|parent|1872|86.31%|0|0|200|
|refreshed_tapered_xyz_seed0:adaptive|1733|79.90%|207|68|222|
|refreshed_tapered_bounded_seed0:adaptive|1727|79.62%|210|65|224|
|refreshed_tapered_projection_seed0:adaptive|1735|79.99%|205|68|224|
|refreshed_tapered_boundcenter_seed0:adaptive|1760|81.14%|190|78|243|
|refreshed_tapered_bounded_seed1:adaptive|1720|79.30%|215|63|219|
|refreshed_tapered_bounded_seed2:adaptive|1721|79.35%|218|67|221|

The denominators are the unchanged parent-defined 2,169 surviving-mode and 270 invalid-old/same-mode-known-new opportunities. Bounded seed 0 repairs more opportunities than the parent but loses more surviving modes; the center control repairs still more. Requested query identity is not treated as actual realized mode identity.



## Frozen fresh-family diagnosis

The fixed16-family/7-variant/336-request collection is running. No fresh-generalization success is claimed before receipt closure.

## Verification, uncertainty and cost

Server tests cover membership, fixed endpoints, candidate-boundary independence, relative decoder gradients, exact tapered clearance and visibility unknown handling. Actual 100-step versus 50+50 recovery has identical model, optimizer, RNG, history, sample stream and settings. The public API reproduces saved paths/events/q and selected indices exactly on 16 requests, with one decoder call per request. Cached-feature API timing excludes online VLM feature extraction.

- deployment_xyz_v2: 28.85 ms/request, maximum replay error 0; selected indices exact.
- deployment_bounded_v2: 28.79 ms/request, maximum replay error 0; selected indices exact.

Depth-only calibrated corner probes classify 9.23% as unknown. Mean unknown fraction is 9.21% on valid routes and 9.38% on invalid routes. These sparse probes are not whole-cell certificates and do not constitute a learned uncertainty predictor.

No reserved TEST_LOCKED payload or metric was inspected. Historical d0d97eb and runs/main remain preserved. Only wzy3090, physical GPU1 UUID7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,35%memory and CPU0–3 were used; rendering uses one worker/software GL. No shared environment was changed.

## Decision

The conditional containment guarantee is implemented and independently checked; the learned region remains an estimate. H1 shows a small relative effect versus XYZ but fails meaningful-benefit and strongest-control requirements. H2 boundary invariance holds structurally, yet the center control has the same protection and better task quality. H3 must be read from fixed-witness and frozen fresh-family results; it cannot be inferred from a few favorable route plots. There is no evidence here for a publication-ready novel interior decoder or a deployment-default switch.

Code, method/proof, full ablations, causal limitations, paper draft, real saved-prediction figures, reproduction instructions and hash receipts are delivered. The requested publication-potential goal remains unmet; no acceptance or real-robot result is invented.
