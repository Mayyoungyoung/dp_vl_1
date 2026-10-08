# TRAIN contains a smaller edit-retention residual

The registered read-only diagnostic completed. All1152 TRAIN predictions from
safety_mean were hash-verified and rechecked; no model forward or optimizer step.
Source1df437d8fedc067d3d8abb0d595123d82c61dfbd, source archive
d3497e81d49c78772cdb8ff426a16e1bb5b34542f7679a7127dfd320fe168e65.
4 tests pass; audit wrapper27.703155s, exit0.7 closure files locally SHA verified.

| Direction | TRAIN surviving modes | Lost | Slot-feasible | Correct-endpoint slot-feasible | DEV correct-endpoint slot-feasible |
|---|---:|---:|---:|---:|---:|
|open→closed|2281|96|70|45|23|
|open→shifted|2894|207|73|47|40|
|closed→open|1751|348|40|32|23|
|shifted→open|2922|178|23|20|23|

TRAIN has384 pairs per direction, DEV96. The stricter opportunity per pair is
.1172/.1224/.0833/.0521 on TRAIN versus.2396/.4167/.2396/.2396 on DEV.
There is usable TRAIN residual, but a substantial generalization gap remains.
Pairs and directions share families; these totals are not independent samples.

Among eligible lost paths with a same-mode original teacher, median mean
Euclidean distance over24 points is2.189/3.029/2.428/2.976cm. Against all four
perturbations plus original it is2.102/2.944/2.352/2.911cm. Every finite value
and all missing-mode cases are saved; no distance threshold selected examples.
Corresponding finite counts19/43/19/20 and missing-mode counts26/14/23/5 count
paths, not the slot-limited mode opportunities in the table. Several paths may
compete for one slot, so these counts must not be substituted for mode gains.

Known/unreferenced stricter opportunities are19/26,38/12,17/20,18/5 and can
compete for slots. Unreferenced positive paths are not invalid. The diagnosis
does not prove that adding positives or a pair loss will generalize. It motivates
a separately registered ordinary, verified same-mode path-augmentation control
before attributing anything to a new correspondence mechanism.

Original manifest, model and prediction SHA checks pass. TRAIN label/config/
teacher file hashes agree with the immutable support receipt. The support NPZ
also contains permitted DEV arrays; only explicit TRAIN indices are used in
all checks and distance calculations. No scoring/calibration data or locked
TEST payload is involved. Full provenance and rows/distances remain in
TRAIN_EDIT_CLOSURE_ARTIFACTS_20261009.json and its verified archive.
