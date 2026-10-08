# Complete-input multimodality audit

Actual audit source b6727ff; audit_v4 completed exit0 on 2026-10-09 (Asia/Shanghai).
Full identity hashes RGB pixels, depth, camera matrices, current gripper state,
instruction and world-metric convention. No grouping by task category alone.

There are **1440 unique complete requests**, 1152 TRAIN and 288 DEV_MODEL,
from **160 independent layout families** (128/32). Every request has 6, 7 or 9
verified references. Counts: 960 requests have9, 240 have6, 240 have7. All11760
stored H24 references pass the current full-line checker. These are geometric
teachers over actual rendered RGB-D, not execution demonstrations. The historical
22-reference training exclusion came from other merged corpora, not these paired
references. Historical raw data remain untouched.

Each witnessed operational passage has exactly one base teacher reference.
Max/min teacher-count ratio is1. Thus the present corpus does not test frequency
imbalance. Natural RareModeRecall is undefined until an explicit controlled
frequency regime is imposed; a singleton teacher is not a rare-mode estimate.

Modes: first forward crossing per registered obstacle row. Above the 2cm-inflated
top is `over`; otherwise the lateral free gap is `gap0/1/2`. This separates central
low and high passages and removes arbitrary unknown height strips. It is an
operational passage vector, not a proof of different homotopy classes. Historical
portal words are retained separately: their lateral-first precedence merges some
central-low/high references. No old metric or artifact is rewritten.

Zero paired-family role overlap and zero parent-ID overlap across SCORE_TRAIN64,
DEV_SCORE32, CALIBRATION32, FUTURE_GENERATOR_TRAIN96 and the paired corpus. This
rechecks these explicitly registered exports; it does not claim an exhaustive
image-duplicate audit of every historical corpus. Old development parents stay
reused development evidence. TEST_LOCKED payloads and results were not opened.

The witness set is incomplete. A closed expanded low gap has a local geometric
absence certificate for that gap; it does not eliminate above or exterior paths.
For open/shifted inputs, nine known witnesses exceed M8, so full recall is
impossible at that budget. Use capacity bounds in coverage interpretation.

Reproduce with `python -m scripts.research_v3_audit audit --output NEW_DIRECTORY`
from an immutable source export on the authorized server. Actual commands and
source hashes are in runs/research_v3_local/jobs. Request details, reference mode
counts and certificates are in data_requests.json; metrics and input hashes in
metrics.json. Inputs are oracle audit labels; model forward sees observations only.
