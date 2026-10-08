# Verified same-mode edit positives: rejected ordinary control

The pre-registered augmentation gate is false. Adding 9027 geometry-checked
TRAIN paths to 47040 original references does not establish improved retention
and worsens route validity and fixed-q Brier. These are ordinary data controls,
not evidence for a new correspondence mechanism.

All results use the same 288 DEV_MODEL requests (32 families), M=8, K=4 and
complete paired-domain q seed0. Parent means unchanged safety_mean; fresh means
margin_mean, its sealed fresh-AdamW 1200-step continuation. Augmented starts
from the identical parent, repeats that input/group RNG and changes only the
within-mode positive regression pool. Original grounding labels stay fixed.

| Metric | Parent | Fresh | Augmented |
|---|---:|---:|---:|
| Fixed-source retained modes /2169 |1872|1775|1788|
| Fixed-source retention |.863071|.818349|.824343|
| Candidate validity |.868056|.872830|.835938|
| Distinct valid modes |6.680556|6.670139|6.395833|
| Rare known-mode recall |.744068|.738426|.720139|
| Any valid candidate |.958333|.961806|.944444|
| Brier |.075390|.077099|.100006|

Retention uses the SAME safety_mean source witnesses for every destination
model. Augmented-minus-fresh is +.005994, family-bootstrap 95% interval
[-.028392,.039172]; the registered minimum is +.05 with positive lower bound.
Validity drops .036892 [-.064670,-.009549], distinct modes drop .274306
[-.496528,-.062500], and Brier increases .022907 [.010315,.037217].
Augmented-minus-parent retention is -.038728 [-.062727,-.018439]. Thus even
repair of additional-training damage is incomplete. All directions and metrics
remain in RESULTS.json and ../verified_edit_parent_v1/RESULTS.json.

The full TRAIN audit shows this is not solely a DEV generalization failure:

| Split/model | Valid candidates | Post collisions | Wrong semantic endpoints | All endpoints wrong requests |
|---|---:|---:|---:|---:|
| TRAIN parent /9216 |8937|189|91|7|
| TRAIN fresh /9216 |8883|261|75|6|
| TRAIN augmented /9216 |8842|192|187|11|
| DEV parent /2304 |2000|207|108|12|
| DEV fresh /2304 |2011|200|106|11|
| DEV augmented /2304 |1926|220|175|16|

Collision and endpoint counts overlap; do not sum them as exclusive failures.
TRAIN augmentation removes 69 collisions versus fresh but adds112 endpoint
errors. Unchanged grounding labels do not imply unchanged learned grounding.
This motivates a separate, prospective frozen-encoder factorial control;
it does not prove the encoder is the only cause.

All 853 valid cross-edit paths outside destination reference modes were retained
as excluded positives, never negative labels. Exact path/event deduplication
leaves56067 total references; mode classes, observations and parameter count
unchanged. Target slots processed increase1566630->1867761; no equal-target-
compute or speedup claim. One generator seed; repeated DEV inspection remains
development evidence, not independent testing. No TEST_LOCKED access.

Training source6dc788e9c6cf7dc2a91ca850425da8afa27371a3, source archive
3f94a8db8c3133a45d32d3b1c7557a2d14bdbb7069f74ecc1b6846a67b65ebd2.
Frozen launcher scripts/run_research_v3_edit_augmentation.sh records actual
commands; all5 jobs exit0, tests12pass, train277.166349 command seconds.
Final checkpoint e0df5b5c02746e093cd1f3b1c44fcd5c8c729ada8ed7bb0401a366257e2c337f.
TRAIN audit source8866936f2796f1b136732090f7b080adbb2a96b4, exit0 in21.536214s.
Local parent comparison command/provenance is preserved alongside its results.
27 primary closure files and6 TRAIN audit files individually SHA verified in
VERIFIED_EDIT_CLOSURE_ARTIFACTS_20261009.json and
VERIFIED_EDIT_FAILURE_CLOSURE_ARTIFACTS_20261009.json (one directory above).
All queues closed:134 jobs129success5retained failures;6223.231135/7200
experiment command seconds spent,976.768865 remain. Retain safety_mean.
