# Research V3 state — 2026-10-09

Status: phase-one audit completed; controlled pilot recovery running, no positive method claim. Baseline commit 127f547;
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
Current immutable source5ae75bf, archiveSHA f1d084e81bcb2224159bb69305f409c0750f97fc8b878392debe95ebc3518856.
Coordinatorv2 failed on an extra pause-boundary history row, preserved unchanged.
Coordinatorv3 is ACTUALLY running in SSH43335; five CPU tests passed.
Next: exact100 versus50+50 resume check; five1200-step single-seed retention
controls on fixed support and frozen q. This is a screening experiment, not a
novel mechanism or final independent evaluation. If ordinary controls suffice,
reject frequency balancing as novelty and diagnose the next evidenced bottleneck.

Resources: SSH wzy3090 only; GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,
35% memory, four CPU threads, existing environment. Exploratory launcher budget
7200 cumulative command seconds, includes failures; not GPU utilization hours.
Server experiments use immutable exports; local workspace writes all code.
