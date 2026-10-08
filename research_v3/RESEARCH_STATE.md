# Research V3 state — 2026-10-09

Status: phase-one audit and five frequency controls completed; equal-target control running, no novel-method claim. Baseline commit 127f547;
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

Frequency hypothesis is unknown: original teacher counts are uniform. All47040
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

ACTUAL current coordinator: sampled_coordinator_v1 from immutable
daf6c013e1152a8ea0e10775f4c294572239f789, SSH53093. Four server tests passed;
train set_sampled1200, fixed-q evaluate, equal-budget analysis and concrete-path
counterfactual diagnosis in sequence. Do not restart completed original arms.
ArchiveSHA d98be20eeb242f7e60298cae8151b372824ac4b8b1cede4310f4f188e59e45cb.

Resources: SSH wzy3090 only; GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,
35% memory, four CPU threads, existing environment. Exploratory launcher budget
7200 cumulative command seconds, includes failures; not GPU utilization hours.
Server experiments use immutable exports; local workspace writes all code.
