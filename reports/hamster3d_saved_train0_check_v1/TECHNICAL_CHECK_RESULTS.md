# HAMSTER saved TRAIN0 four-point check

The sole original K1 candidate **fails the existing reach-route check**. This is a posthoc technical check of already saved output; no additional model, planner, simulator, prompt, candidate, path repair, or checkpoint selection was run. Original generation artifacts remain unchanged and their quality fields remain null; all metrics below live only in this independent family.

The official strict fenced JSON contains four finite world points and actions Close / None / None / Open. The preregistered carry-forward mapping is [0,0,0,1]. The original four-point polyline has exactly three adjacent segments. Nothing was prepended, removed, interpolated to H24, or repaired; the fourth point remains the endpoint.

| Original check | Actual result |
|---|---:|
| Candidates submitted | 1 |
| Finite coordinates / event values | pass / pass |
| Start error (threshold5mm) | 42.427175cm, fail |
| Endpoint to instructed black target (threshold3cm plus identity) | 34.314351cm, fail |
| Nearest target at endpoint | target2, not target0 |
| Reach event sequence (constant initial open) | fail |
| Original 3-segment 2cm tip clearance | segments0/1 fail; segment2 pass |
| TipValid@1 / AnyTipValid@1 | 0 / 0 |
| Classified type | unknown |
| Known-positive type coverage (five known reference types) | 0 |
| Full-arm / IK / execution success | not measured |
| SelectedValid / learning scorer | not available |

The four points are respectively 10.714, 10.803, 36.558, and 34.314cm from the instructed target. The first point's image proximity therefore does not establish reaching the 3cm target, and neither first point nor nearest point was substituted for the actual endpoint. The raw path length is 0.542647m. The three per-segment flags are descriptive decompositions of this same candidate, not three extra proposals.

This exposes a mismatch between the pretrained model's sparse manipulation-waypoint output and this derived reach-route task. The official full-manipulation template generated a Close→Open sequence; the saved world positions additionally fail target and segment geometry checks. These facts cannot establish a complete robotic failure rate from one TRAIN example, nor provide a system-level benchmark comparison. An executor could add connections in another declared protocol, but no such connection earns validity credit here.

## Provenance and cost

Prediction NPZ remains SHA `71a58d2602f8fc5b589f8d389d921c169f68037a3e85a8564c481f3c9bb81213`. The original generation cost remains 151 forward calls /151 generated tokens /one candidate; this check does not add generation exposure. Model input and prediction hashes were already sealed in the original archive. Mapping/no-repair rules were fixed in SAVED_PREDICTION_CHECK_PROPOSAL before root authorization. A first-line metadata preview preceded this folder's machine mapping receipt; this ordering is recorded explicitly, and no prediction/mapping changed after that preview.

Only `two_row_reach_283200_target0` supervision was decoded. Its parent verification geometry and route configuration are used strictly for this evaluation. The supervision file's full bytes were hashed for integrity, but the other two rows were not decoded and no other TRAIN/DEV/reserved sample was opened. The label index binds each original source path and SHA. Four frozen checker dependency byte hashes match the original eba0994 source manifest; original2cm/3cm/5mm/event/type rules remain unchanged.

The local wrapper plus one short CPU3 read-only label transfer took 1.115002s; the primary `scene_metrics` call took 0.000916s, nested within that wall time. The per-segment descriptive checks and final reporting are separate local CPU overhead; no GPU was used. The initial shell-quoting metadata probe failed before labels/model/checker calls and is preserved in metadata_probe_failure.json; the subsequent Python-stdin transfer succeeded. There was no model or candidate retry.

Reproduce the result using analysis_source.py.txt against its fixed existing inputs and a fresh output location. It is intentionally fresh-only; no repeat was run in this investigation. No further generation or adapter is proposed as a way to retroactively rescue this candidate.
