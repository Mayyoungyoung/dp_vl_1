## V3 collection verified live; calibration follow-up frozen — 2026-10-09

Verified actual processes: collectorPID4127618 and its new worker PIDs live;
source9bbc021 main queueSSH65969 continues unchanged. Latest inspected worker
index78 completed exit0 with initialization true (no complete-data claim).
Calibrator follow-up ACTUALLY waiting inSSH21432, PID4133095, source
0954ff5df97726b0945aaab3866bd06e1b8bd6b5, archiveSHA
9a7e04d87237a3f69975d3ffa79db84cec042026f636ac658c52ae625ef237cb.
It waits for paired_score_coordinator_v1 exit0, then runs2 tests and registered
temperature/monotone-affine newCAL-only controls. Do not duplicate either queue.
New diagnostic figure in research_v3/figures_scoring_v1 is rendered and visually
checked, derived only from closed original snapshots, with source hashes.
Goal remains ACTIVE. No novel mechanism, formal new-method seeds or final-test
result established. Historical running entries below are superseded only where
explicitly closed; the two queues named here really are live.

## V3 new paired score-data queue ACTIVE — 2026-10-09

Current immutable source9bbc0213a542815cd767716ddc5e5869f5d19d4c, archive
402174d005255ab69fb234383f2a62b48e7d96ea95772343944de0c596c33ff2.
SSH65969 runs scripts/run_research_v3_paired_score.sh; do not duplicate.
Registration test and preparation passed; first6 physical smoke parents exit0
in74.554165s, all initialization true, observed first image inspected.
Remaining186 collector is RUNNING (last inspected indices40/41 complete,
42total including smoke, no failures at that read). It then exports3 separate
roles, genuine Qwen per role, four exact pools, three ordinary q fits/calibrations,
seed0 public CLI and analysis. Read paired_score_coordinator_v1/closure.json,
jobs/paired_score_collect_remaining and collection receipts before action.
All older queues closed. New source hashes MUST NOT replace running9bbc export.
Server cumulative completed budget2377.000461 seconds before remaining collector;
running duration not yet included. Collection timeout2220s, total V3cap7200s.
Current matched_old q gate false; source85f8213 scores .1641/.1744/.1456 Brier.
New calibration-only controls prospectively registered while collection running
in CALIBRATION_CONTROL_PROTOCOL.md; local2 tests passed. They are NOT launched
or exported yet. Run after paired queue closes from a fresh immutable export.
Core novelty remains unestablished. Goal ACTIVE, no final-paper completion claim.

## V3 matched-q closed; new disjoint score-data smoke prepared — 2026-10-09


## 2026-10-09 Matched old-domain q completed and rejected
All12 queue jobs exit0 from85f8213, coordinator95s. Three calibrated Brier values.164112/.174366/.145578, mean.161352 vs fixed.148448; gatefalse. Public seed0 subprocess paths/events exact,q maxdifference within1e-6,selection exact. All pools and original pairedDEV arrays exact; roles disjoint. Same generator validity43.42% SCORE_TRAIN,42.71% DEV_SCORE,39.45% CALIBRATION vs86.81% pairedDEV indicates substantial scoring-domain shift; not proof of pure label shift. New95 files downloaded/hashverified, archive2d0488ec4a8ce99c5eca8afccaced9356b3ba8cc51faf6d1648426a59e979f99. Cumulative2296.288556 command seconds. Register64 fresh paired-layout score families before collection, no existing role reuse, same model and training settings. This isolates ordinary scoring data coverage, not architecture novelty. First6-parent physical smoke gate before remainder. No existing queue running.

## V3 matched single-q baseline running — 2026-10-09

Current finite queue source85f821367df92ba932365595933e21d61fbd3b97, archive
057294dcf07e24f18a6569801a75926c2f20ccb3a3e0a9801d5e7f694d05373f,
SSH65284 actually runs scripts/run_research_v3_matched_q.sh. This freezes
safety_mean proposal model and original score feature encoder, constructs
four role-specific exact pools, trains/calibrates all3 ordinary q seeds.
Check matched_q_coordinator_v1/closure.json and receipts before action;
no duplicate launch. Anchor TRAIN gate passed but full DEV route gate FAILED:
valid+1.345pp CI crosses0, any-valid unchanged, Brier worsened.05185.
No anchor adoption or bandwidth tuning. Core novelty remains unsupported.

# Research V3 state — 2026-10-09

Status: audit and six frequency controls completed; gradient-matched safety control completed and rejected, no novel-method claim. Baseline commit 127f547;
branch codex/multiroute-v2. Existing untracked work preserved separately.

Historical evidence: R2/R3 cross-scene correspondence failed against R1 across
three seeds; dual-factor scoring failed against single q across crossed seeds.
These findings constrain the next research choices and will not be rerun as
if new hypotheses. TEST_LOCKED stays closed. Score/calibration roles stay separate.

Audit_v4 source b6727ff, exit0:1440 complete requests,160 families,11760 verified
references, no paired role leakage,6912 prediction labels rechecked exactly.
Operational mode definition fixes above/central merging and narrow unknown
height strips without changing old metrics. Mean M8 OracleValid95.370%,
ValidDistinct5.5914, known witness recall62.281%. K4 recall40.209%, distinct3.5856.
47 probes miss182 continuous post collisions. Generator-empty40/864 requests;
scorer-miss12/864; extra K4 capacity loss149/864 mode slots. Generation coverage
and collisions dominate, while selection is close to its capacity bound.

Controlled frequency hypothesis is supported; original teacher counts are uniform. All47040
unique smooth perturbations passed checks (support hash77f11adf55fe68ae93b7efc42ce4162213f41d12355d907b5ca606d97749d150).
Five1200-step arms from5ae75bf completed, strict100 versus50+50 equality passed.
Both bias gates true: rare recall44.641% uniform,6.904%90:10,0%98:2.
Full-set ordinary matching64.358%, valid82.205%, distinct6.566, but processes
1,566,630 reference slots vs307,200. Initial R1 rare recall65.796%; matching
does not improve that initial minority recall despite more distinct valid modes.
No novel mechanism follows from these conventional controls.

Fixed-q correction2550c9d completed: preserve original coupled-feature evaluation;
corrected evaluation_fixed_q_v2 holds the entire original scoring encoder fixed.
All candidate paths/events/labels unchanged exactly. Six tests and actual exact
deployment q replay passed. All five actual input streams/initializations match;
uniform and balanced final tensors/pools are identical. Replayed exposure agrees
with checkpoints:90:10 leaves272 request-minority modes unexposed,98:2 leaves3915.

Equal-target control and counterfactual diagnosis fromdaf6c01 completed exit0.
set_sampled rare57.685%,valid76.866%,distinct6.149; beats sampled balanced but
loses to full-set matching. Full matching almost eliminates valid duplicates.
Under open-to-closed edit it loses76/498 still-valid source witness modes;
mode substitutions and finite output budget must not be mislabeled failures.

TRAIN diagnosis24c9e42:59/1024 paths collide, all59<=5 bad segments,38 one segment;
prospective safety-control gate true. Coefficient10.158783248832323 fixed by
TRAIN gradient ratios. ACTUAL safety preflight/queue from immutable
609a0b590d0cf07d93ada0ff4c58ccea363a5de2, SSH29657: geometry tests completed,
100-step preflight then mean/worst1200-step paired continuation and evaluation.
ArchiveSHA2b406d405ee10a83b60d060688d3627a18fb330f43c60608d78d7d1d3d2c35c8.
This is a conventional strong-baseline test, not a proposed novel mechanism.

Resources: SSH wzy3090 only; GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,
35% memory, four CPU threads, existing environment. Exploratory launcher budget
7200 cumulative command seconds, includes failures; not GPU utilization hours.
Server experiments use immutable exports; local workspace writes all code.

Superseding checkpoint: safety_mean and safety_worst completed exit0. Worst
validity gain0.217pp CI crosses0; minority recall loses3.108pp. Frozen gate fails;
no sweep. Public CLI110abd9 exact paths/events/q/selection verified,2.801s with
cached Qwen. Closed snapshot244 files allSHA verified,54jobs incl4failures.
Anchor audit de8c791 exit0:432/432 requests have visible goal support. Mean
DEV12 all-endpoint failures all outside anchor residual range; TRAIN2 failures,
1 impossible. Total command budget2141.379957/7200s after this audit.
All prior queues CLOSED. Next registered TRAIN-only mass diagnostic uses all
1152 TRAIN requests and one fixed sigma; no route/model update yet. See
ANCHOR_MASS_PROTOCOL.md. Goal remains active; paper novelty unestablished.
## Current additional diagnostic — 2026-10-09

Prototype localization381de22, archivec2ee654c4acac34778b844b3a40f0ca9773026ec3f7d76a596e5c532b6a7bfae,
ACTUALLY waitsSSH1473/PID4138597 after calibration0954ff5/SSH21432, itself waiting
on paired-score9bbc021/SSH65969. It is a fixed ordinary TRAIN-fitted color
prototype with stride2 observations, not a new route model. Local1 test passes;
actual server test and all288DEV endpoint results pending. No duplicate queues.
Additional primary-source checks in LITERATURE_REVIEW.md constrain unsupported
joint-risk or conformal novelty claims. Research objective remains active.
