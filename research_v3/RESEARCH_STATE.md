# Research V3 state — 2026-10-09

Status: audit in progress, no positive method claim. Baseline commit 127f547;
branch codex/multiroute-v2. Existing untracked work preserved separately.

Historical evidence: R2/R3 cross-scene correspondence failed against R1 across
three seeds; dual-factor scoring failed against single q across crossed seeds.
These findings constrain the next research choices and will not be rerun as
if new hypotheses. TEST_LOCKED stays closed. Score/calibration roles stay separate.

First decision: recheck complete-input multimodality, low/over relation versus
historical portal labels, raw M8 upper bound, scoring and K4 coverage loss, and
continuous collision versus 47 probe samples. Then choose a falsification pilot.

Resources: SSH wzy3090 only; GPU1 UUID GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,
35% memory, four CPU threads, existing environment. Exploratory launcher budget
7200 cumulative command seconds, includes failures; not GPU utilization hours.
Server experiments use immutable exports; local workspace writes all code.
