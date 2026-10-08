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
