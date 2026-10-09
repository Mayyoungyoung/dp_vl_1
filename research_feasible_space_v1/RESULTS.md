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


The final standalone controls are additionally evaluated on all three frozen generator continuations with unchanged witnesses:

|Three-seed mean|Retention|Same-mode repair|New losses|Recovered|
|---|---:|---:|---:|---:|
|Same-information free XYZ|79.05%|81.23%|221.33|64.00|
|Proposed bounded mapping|79.42%|81.98%|214.33|65.00|
|Projection, fresh head|79.13%|82.22%|219.67|64.00|
|XYZ center, fresh head|80.94%|89.01%|195.67|79.33|
|Bounded center, fresh head|80.74%|89.01%|199.33|78.67|

|Bounded minus standalone control|Δ retention|95% CI|Δ same-mode repair|95% CI|
|---|---:|---|---:|---|
|bounded minus xyz|+0.37 pp|[-0.32, +1.12] pp|+0.74 pp|[-1.26, +3.06] pp|
|bounded minus refitted_projection|+0.29 pp|[-0.33, +0.97] pp|-0.25 pp|[-2.11, +2.17] pp|
|bounded minus refitted_center|-1.52 pp|[-2.54, -0.67] pp|-7.04 pp|[-12.77, -2.42] pp|
|bounded minus refitted_boundcenter|-1.32 pp|[-2.25, -0.48] pp|-7.04 pp|[-12.23, -2.80] pp|

## Fixed local-parameter transfer

This additional diagnostic fixes the old bounded relative coordinates and maps them through the destination predicted corridor. All events come from the destination prediction. Eligibility requires a valid requested source word also requested at destination; these denominators differ from the fixed2,169/270witness sets. It uses saved predictions only, adds no model decode/update and is not an8-candidate deployment policy.

|Seed|Eligible slots|Old XYZ failures|Copied XYZ success|Transferred relative success|Destination center success|Destination native success|
|---|---:|---:|---:|---:|---:|---:|
|0|2309|275|88.09%|91.25%|93.24%|90.86%|
|1|2287|277|87.89%|90.77%|93.05%|90.51%|
|2|2301|267|88.40%|90.27%|92.26%|90.13%|

|Seed|Repair with transferred parameters|Repair with new center|Repair with native decoder|
|---|---:|---:|---:|
|0|77.82%|82.55%|76.36%|
|1|77.26%|82.67%|75.45%|
|2|75.28%|80.52%|73.78%|

Parameter portability can establish a coordinate representation effect, but its centerline control must still establish whether nonzero learned interior parameters are needed. The diagnostic does not override the failed candidate-set comparison.

## Frozen fresh-family diagnosis

16fresh rendered families,7variants,336requests; frozen3generator continuations; no selection use; sparse teacher recall.

All model/head checkpoints were frozen before collecting these observations. No model selection, update or weight search uses them. All 16 families are new; their physical geometry hashes were checked against the old paired TRAIN/DEV registration. Eighty scenes were actually rendered; 32 additional observation variants are explicitly synthetic RGB/depth corruptions of the open scenes. This is task-level simulation evidence, not whole-arm or real-robot execution.

|Frozen method, three-seed mean|V8|U8|V4|U4|Sparse teacher recall|
|---|---:|---:|---:|---:|---:|
|Same-information free XYZ|79.22%|6.2649|89.78%|3.5883|69.03%|
|Proposed bounded mapping|81.08%|6.4028|90.33%|3.6111|70.02%|
|Projection, fresh head|80.18%|6.3333|90.15%|3.6052|69.56%|
|XYZ center, fresh head|86.69%|6.8323|92.06%|3.6796|74.45%|
|Bounded center, fresh head|86.58%|6.8214|91.96%|3.6756|74.36%|

|Variation|XYZ U8|Bounded U8|Refitted XYZ center U8|
|---|---:|---:|---:|
|closed|5.7569|6.0486|6.4444|
|narrow|6.7083|6.9306|7.3333|
|noise|6.6875|6.7361|7.1875|
|occluded|3.8403|4.0139|4.7083|
|open|7.0069|7.2014|7.3542|
|shifted|7.0208|7.0347|7.4583|
|tall|6.8333|6.8542|7.3403|

|Bounded minus control, fresh families|ΔU8|Conditional crossed 95% CI|
|---|---:|---|
|bounded minus xyz|+0.1379|[+0.0784, +0.1974]|
|bounded minus refitted_projection|+0.0694|[+0.0188, +0.1161]|
|bounded minus refitted_center|-0.4296|[-0.5972, -0.2768]|
|bounded minus refitted_boundcenter|-0.4187|[-0.5833, -0.2659]|

The fresh reference set is incomplete geometric teacher evidence; its recall must not be compared directly to the old all-mode-support recall. Initial fresh recall mixed legacy full-portal reference strings with operational prediction words and incorrectly returned zero. The v2 statistics reclassify the same verified reference paths with the prediction definition, without changing or re-decoding any prediction; v1 recall is superseded. Noise and masking diagnose robustness, not calibrated confidence. This extension does not retroactively relax or replace the failed original mechanism gate.

## Verification, uncertainty and cost

Server tests cover membership, fixed endpoints, candidate-boundary independence, relative decoder gradients, exact tapered clearance and visibility unknown handling. Actual 100-step versus 50+50 recovery has identical model, optimizer, RNG, history, sample stream and settings. The public API reproduces saved paths/events/q and selected indices exactly on 16 requests, with one decoder call per request. Cached-feature API timing excludes online VLM feature extraction.

- deployment_xyz_v2: 28.85 ms/request, maximum replay error 0; selected indices exact.
- deployment_bounded_v2: 28.79 ms/request, maximum replay error 0; selected indices exact.

Actual coupled RGB-D/language-to-four-route timing is 120.32 ms mean / 138.41 ms median over eight DEV requests after one excluded warmup. It includes disk reads, tokenization, frozen Qwen feature extraction, point backprojection, both observation encoders, one8-route decode, full frozen q and selection; it excludes the independent checker and 2.55-second model load. Joint peak allocated GPU memory is 4.322 GB. Online features match the cached features within1e-5. This small timing sample is not a throughput or real-robot claim.

Depth-only calibrated corner probes classify 9.23% as unknown. Mean unknown fraction is 9.21% on valid routes and 9.38% on invalid routes. These sparse probes are not whole-cell certificates and do not constitute a learned uncertainty predictor.

Closed ledger: 151 jobs, 148 completed and 3 failed; 5988.7 measured command seconds (1.66 hours). All failures, source hashes, actual commands, checkpoints/optimizer/RNG state and predictions are retained. Artifact-only archive/verification costs are listed separately in CLOSURE.json.

No reserved TEST_LOCKED payload or metric was inspected. Historical d0d97eb and runs/main remain preserved. Only wzy3090, physical GPU1 UUID7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,35%memory and CPU0–3 were used; rendering uses one worker/software GL. No shared environment was changed.

## Decision

The conditional containment guarantee is implemented and independently checked; the learned region remains an estimate. H1 shows a small relative effect versus XYZ but fails meaningful-benefit and strongest-control requirements. H2 boundary invariance holds structurally, yet the center control has the same protection and better task quality. H3 must be read from fixed-witness and frozen fresh-family results; it cannot be inferred from a few favorable route plots. There is no evidence here for a publication-ready novel interior decoder or a deployment-default switch.

Code, method/proof, full ablations, causal limitations, paper draft, real saved-prediction figures, reproduction instructions and hash receipts are delivered. The requested publication-potential goal remains unmet; no acceptance or real-robot result is invented.
