# Verified-set development results

Same initial model and actual per-seed observation streams verified. Fixed final1200; all288 DEV requests retained.
TRAIN population: TRAIN and original known_mode_count<=8, before model inference.
No TEST_LOCKED access, no robot execution claim, no automatic novelty claim.

| Arm | Valid@8 | Distinct@8 | Known recall@8 | Selected distinct@4 |
|---|---:|---:|---:|---:|
|parent|0.8681|6.6806|0.7610|3.7812|
|ordinary_seed0|0.6428|4.6007|0.5597|3.4340|
|gate_seed0|0.8290|6.3368|0.6420|3.7847|
|project_seed0|0.7526|5.5312|0.4283|3.4965|

Engineering gates: `{"project_seed0": {"parent": false, "ordinary_seed0": true, "gate_seed0": false}}`.

Family intervals, actual payload/checkpoint hashes, budget and failure records: RESULTS.json.
