# Variable-layout TRAIN4 actual results

The original fixed pilot4 completed all 108 requested attempts. There is no observed restoration, source, geometry or stored-trajectory integrity error preventing the original bounded pilot12 continuation. The next eight parents still require separate authorization/execution and their own full-denominator quality review; this archive contains only indices 0–3.

Frozen collection source: `eba09945ecade4bdb9a5b4ab123a92da898283ed`. Outer job PID784210 / child784215, 2026-10-03 05:05:03.305940–05:16:15.513453 UTC, exit 0. Only completed TRAIN4 data were read. Independent local verification made **0 simulator/model/search calls and 0 new route proposals**.

## All requested outcomes

| Index / parent | Posts | Height (m) | Accepted | Known positives | Unknown positives | Known-type duplicate positives |
|---|---:|---:|---:|---:|---:|---:|
| 0 / layout_variation_401000 | 1 | 0.10 | 25/27 | 23 | 2 | 14 |
| 1 / layout_variation_401001 | 1 | 0.18 | 8/27 | 8 | 0 | 5 |
| 2 / layout_variation_401002 | 2 | 0.12 | 26/27 | 25 | 1 | 13 |
| 3 / layout_variation_401003 | 2 | 0.18 | 6/27 | 6 | 0 | 2 |

Accepted **65/108 (60.19%)**, failed **43/108**: 37 planning-no-path, 2 recorded arm-environment execution collisions, 4 complete-route acceptance rejections. Complete-path failure components overlap: raw clearance 2, H24 clearance 3, type instability 1, endpoint beyond 3 cm 0. No failure was removed or retried by this audit.

All 12 target conditions have at least one accepted route. Known-type lower bounds, in parent/target order, are `[3, 3, 3, 1, 1, 1, 4, 4, 4, 1, 1, 2]`. None exceeds four yet. The 62 known-positive entries contain 34 repeats of a known type within a condition; exact pose-hash duplicates are zero. The three accepted unknowns remain positive references and are not counted as distinct classified modes or negative examples. Nine attempted proposals do not establish the number of all feasible solutions.

## Integrity and quality

All 108 initial world/RGB restorations are exact; recorded direct pose/open starts, finite sample traces, observation checks, camera restoration and constant-open events pass. All four initial full-body/visibility checks pass; registered and actual 1 mm geometry hashes agree, with no initial/duplicate block. All **65/65 accepted raw and H24 decisions** reproduce using frozen eba functions, 2 cm clearance and 3 cm target tolerance; all accepted stored H24/H64 arrays reproduce bit-for-bit. Supervision references and artifact hashes agree. This is the collector's arc-length H24 representation, not a subsequent event-aware model capacity audit.

Accepted length median **0.741351 m**, mean **0.870276 m**, p95 **1.575157 m**, range **0.486226–2.558789 m**; 2 exceed 2 m and none exceeds 3 m. Maximum recorded accepted height is **1.607033 m**. Full visual QA covered all four RGBs and all twelve nine-slot target sheets; large arcs and failures are retained. See [visual QA](VISUAL_QA.md).

Full-body safety is the original discrete per-simulation-step check, not a newly executed replay or continuous certificate. Native scene files exist, but `roundtrip_verified=false`. Only 1/2-post settings were collected here; 4/6-post settings and open/closed witness certification remain untested. The higher-post layouts have substantially fewer accepted routes, with planning failures dominating; this does not demonstrate that uncollected routes are impossible.

## Cost and reproducibility

Collection outer **672.207513 s**; nested worker bodies **660.980506 s**, attempts **583.580309 s**, restoration **288.249256 s**, planning **172.091349 s**, simulation **100.811540 s**. These are nested costs and must not be added. There were 353 explicit `get_path` calls and 13,624 simulation steps; nested IK API counters are retained per parent. Local read-only rechecking/plotting took **46.834595 s**, with zero GPU hours. Collector rendering is not model inference.

The archive contains 241 original files, 33,893,695 bytes, all rehashed locally. NPZ/scene binaries are preserved under ignored `runs/synced_observed_layout_variation_pilot4_v1`; text, PNG, source and job evidence are here. [REMOTE_ARTIFACT_INDEX.json](REMOTE_ARTIFACT_INDEX.json) contains exact server recovery paths and byte hashes. The original tar SHA256 is `787657cdf899bbdcdb0dd3ea3aa99f47217f4e544fc7806e78e8fe7bdfd73009`.

Actual machine-readable quality: [analysis.json](quality_analysis/analysis.json), SHA256 `f64222816cbc5d480ef8a9d8a813a4a4fabd19ca99c87648dfa5874dd4bb727b`. Frozen source manifest SHA256 `6b779adf227e2822ecf224debb65161c035a2d010ef4c98051d70fc94ee86baf`; local analysis source SHA256 `0ce3a7924e04d930dfbd79db9dcddc3a02b1d4e4e599158268ce2db31e9d394b`. Reproduction is `python .bootstrap/analyze_layout_variation_pilot4.py` after recovering the indexed archive and immutable eba code bundle; it imports frozen checks and never starts a simulator.

Decision: retain all successes, unknowns and failures; continue the originally registered remaining eight technical parents without changing layout, guide, acceptance or type rules. This supports the collection pipeline, not a learned method advantage, R>K claim, complete solution enumeration or conference-readiness claim.
