# Diagnose scoring-domain mismatch with new reserved families

Old-role matched-q all3 fits fail its gate. On identical safety_mean generator,
SCORE_TRAIN candidate validity43.42%,DEV_SCORE42.71%,CALIBRATION39.45%, but paired
DEV86.81%. New scorers have DEV_SCORE Brier.0922-.0987 yet pairedDEV.1456-.1744.
This is evidence of a scoring-domain shift, not proof that prevalence alone
causes miscalibration. No DEV temperature or prior-probability correction.

Next distinct information: can ordinary single-q with *prospectively disjoint
paired-layout scoring families* resolve the gap? This isolates data coverage
from scorer architecture. Existing generator/feature encoder remain fixed.
Register64 new families, seed540099, IDs540000-540063: first32 SCORE_TRAIN,
next16 DEV_SCORE,last16 CALIBRATION. All3 edits and3 target instructions inherit
family role. Verify1mm geometry hashes against all existing paired registrations
and within new data. No existing role repurposed, no TEST_LOCKED access.

First collect6 parents (2 families), require successful physical initialization,
matching intended/readback geometry and complete observations; preserve failures.
Only then collect remaining186 parents. Existing software renderer, canonical
robot state, .venv-sim and frozen geometric teachers; no robot execution claim.
Two workers share CPU0-3, no GPU rendering or environment changes. Collection
max2400 command seconds, disk ceiling2GiB, shared7200-second V3 budget includes
collection and failures. Stop before GPU caching if any scene is unresolved.
Export separate role manifests; never train/evaluate raw collector directories.
Cache genuine frozen Qwen per role with existing35%GPU1/four-thread limit.

After complete export, generate same8 frozen proposals and original scorer
features per request. Fit three ordinary q seeds with exactly the prior1200
steps and38400 request draws; report new pool size vs old192 TRAIN requests.
Calibration only newCALIBRATION; checkpoint only newDEV_SCORE. Evaluate original
pairedDEV pool once. Gate unchanged: average Brier improvement>=.01 vs original
q, all3 negative Brier deltas, selected mode loss<=.05 and all-valid loss<=.01.
Any benefit is from new scoring data, not a novel mechanism. No final-test claim.
No further scoring architecture or calibration sweep if this control fails.
