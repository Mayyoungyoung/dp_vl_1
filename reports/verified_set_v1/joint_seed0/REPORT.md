# Verified-set development results

Same initial model and actual per-seed observation streams verified. Fixed final1200; all288 DEV requests retained.
TRAIN population: All1152 original TRAIN requests; preserve open/closed/shifted exposure.
No TEST_LOCKED access, no robot execution claim, no automatic novelty claim.

| Arm | Valid@8 | Distinct@8 | Known recall@8 | Selected distinct@4 |
|---|---:|---:|---:|---:|
|parent|0.8681|6.6806|0.7610|3.7812|
|ordinary_seed0|0.8607|6.7014|0.6706|3.7604|
|gate_seed0|0.9353|7.3611|0.6915|3.7951|
|project_seed0|0.7708|5.1736|0.3344|3.1319|
|budget_match_seed0|0.8880|6.7569|0.7760|3.7292|
|set_point_seed0|0.9180|7.3438|0.7248|3.7431|
|set_project_seed0|0.9280|7.4201|0.6253|3.8264|

Engineering gates: `{"project_seed0": {"parent": false, "ordinary_seed0": false, "gate_seed0": false, "budget_match_seed0": false, "set_point_seed0": false}, "set_point_seed0": {"parent": true, "ordinary_seed0": true, "gate_seed0": false, "budget_match_seed0": true}, "set_project_seed0": {"parent": true, "ordinary_seed0": true, "gate_seed0": false, "budget_match_seed0": true, "set_point_seed0": false}}`.

Family intervals, actual payload/checkpoint hashes, budget and failure records: RESULTS.json.
