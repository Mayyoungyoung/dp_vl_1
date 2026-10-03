# Variable-layout TRAIN12 pilot: actual quality

The frozen `eba09945ecade4bdb9a5b4ab123a92da898283ed` collection has **324 requested slots: 108 accepted, 162 attempted failures, and 54 unattempted**. Ten parents have usable initial observations; parents 401005 and 401007 failed initialization. Closure alone was not counted as a successful acquisition. All 36 requested target conditions remain in the denominator, including six with no initial input.

The additional eight parents contributed 43 accepted / 119 failed / 54 unattempted. The original four retain 65 accepted / 43 failed, with all original corpus bytes unchanged. Two run registries only appended new entries; the original pilot4 wrapper remains indexed in its existing family. All twelve regenerated old target sheets are byte-identical.

| Index | Actual post count | Accepted | Route failure | Unattempted | Known / unknown positive |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 25 | 2 | 0 | 23 / 2 |
| 1 | 1 | 8 | 19 | 0 | 8 / 0 |
| 2 | 2 | 26 | 1 | 0 | 25 / 1 |
| 3 | 2 | 6 | 21 | 0 | 6 / 0 |
| 4 | 4 | 14 | 13 | 0 | 10 / 4 |
| 5 | 4 | 0 | 0 | 27 | 0 / 0 |
| 6 | 4 | 3 | 24 | 0 | 3 / 0 |
| 7 | 4 | 0 | 0 | 27 | 0 / 0 |
| 8 | 6 | 20 | 7 | 0 | 12 / 8 |
| 9 | 6 | 2 | 25 | 0 | 2 / 0 |
| 10 | 6 | 2 | 25 | 0 | 2 / 0 |
| 11 | 6 | 2 | 25 | 0 | 2 / 0 |

## Checks and remaining limits

All **270 attempted restores** match the saved world/RGB/observation/camera checks exactly, and all attempted gripper traces remain open. Every accepted raw/H24 check was reproduced with the original frozen helper; all saved accepted H24/H64 arrays are bit-identical to their original resampling. Endpoint failure above 3cm is zero among complete postchecks. The 162 failures are 127 planning-no-path, 21 recorded arm-environment collisions, and 14 complete-route rejections. The latter have overlapping causes: seven raw tip collisions, nine H24 tip collisions, five raw/H24 type instabilities. These are not additive categories.

There are **93 known and 15 unknown positive references**, with 34 repeated known types (all in the original four parents), zero exact pose-hash duplicates, and positive references for **27/36 requested conditions** (27/30 conditions with actual input). No condition has more than four known types. The four six-post parents provide 26 accepted / 82 failed / 108 attempted slots; this establishes actual six-post collection, not a greater-than-K mechanism result or exhaustive solution count.

Accepted raw lengths range from 0.486 to 4.297m; median 0.840m, mean 1.062m, p95 2.450m. Ten exceed 2m and three exceed 3m. Maximum recorded accepted height is 1.808m. All large loops and unknown positives remain present; route type labels are not guide labels or total-solution labels. Full-body checking is the original discrete simulation-step check, not new execution or a continuous certificate. Native scene export exists but roundtrip_verified is false.

## Why 54 slots were unattempted

Both closed members (indices5,7) stopped at `Actual geometry 1mm registration mismatch`, before a usable initial model observation or any route attempt. Saved failed-world reconstruction shows nominal y=±0.0275m versus float readback ±0.027499999850988388m. The difference is 1.49e-10m, but the original `rint(value/.001)` maps these to ±28 versus ±27. Maximum raw geometry discrepancy is 2.623e-8m, within the existing 1e-6m physical readback tolerance. This is a demonstrated quantized-hash boundary issue, not proof of a blocked physical solution. See INITIALIZATION_HASH_DIAGNOSTIC.json; the source and original failures are unchanged.

The two proposed open/closed pairs therefore **do not have an actual closed-member witness/certificate**. No certify stage ran. A future explicit version could repair numeric identity handling and retry only these failed parents under new lineage, but this report does not retroactively accept their initialization or rerun any scene.

## Evidence and visual QA

All 575 original files (82,157,764B) were downloaded and SHA-verified. Binary originals are retained in ignored `runs/synced_observed_layout_variation_pilot12_v1`; exact server recovery paths are in REMOTE_ARTIFACT_INDEX.json. Actual analysis.json SHA: `2e222a917ee15da5d476b039c44516876efe1de1c26f6c18cafefcb61bf7a171`; analysis-source SHA: `6d62edae3ed98b2519f267f0e51d6de23f1ce72c4afe6f303b0c8ff75585d0ef` (different objects). The first local formatting attempt failed on a missing planning_seconds field for zero-attempt parents; its source/log/receipt remain in quality_analysis. The subsequent formatter uses zero for that absent zero-attempt cost and leaves frozen acceptance functions untouched.

The ALL12_RGB sheet plus all 24 new-stage target sheets were actually viewed. Prior full QA of the old twelve sheets is reused only after exact byte comparison. See VISUAL_QA.json and the 36 complete nine-slot sheets under quality_analysis_v2/figures; all six missing-input target sheets explicitly preserve their unattempted slots. Root separately viewed all four old RGBs and each old target0 sheet. Visual projections support QA only and do not replace numerical checks.

## Cost and decision

The added-eight-parent stage took 1220.834587s outer wall time; both stages together took 1893.042100s. Cumulative inner workers took 1864.584737s, with coordinator-measured worker time 1882.710264s. These are alternative nested scopes, not additive cost. Explicit get_path calls: 1391; simulation steps: 38031. The successful local zero-generation quality analysis took 101.910704s. GPU hours for this audit: zero. COST_ACCOUNTING.json explicitly corrects the generic analysis.json scope wording (new8 outer versus cumulative12 inner).

Among attempted scenes there is no new restoration, resampling, or acceptance inconsistency. Nevertheless, known diversity has not exceeded K4, high-post collection is sparse, long arcs remain, and the open/closed premise is unestablished. Preserve these results; repair the proven hash representation issue only in a separately versioned future run if authorized. No model, simulator, planner search, additional acquisition, or certification was run by this quality audit.
