# Verified positive vocabulary completion: rejected ordinary control

The prospectively registered gate is false. All853 previously excluded verified
positive paths were added without discarding any original/known-mode reference.
The resulting56920TRAIN references contain6-13mode classes per request; candidate
budget remains8. This is ordinary data completion, not a new method.

| Metric on288DEV_MODEL requests | Parent safety_mean | Known-mode augmentation | All-mode augmentation |
|---|---:|---:|---:|
| Valid fraction |.868056|.835938|.818576|
| Any valid request |.958333|.944444|.958333|
| Distinct valid modes |6.680556|6.395833|6.291667|
| Rare known-mode recall |.744068|.720139|.685272|
| Fixed complete q Brier |.075390|.100006|.098409|
| Retained fixed-source modes /2169 |1872|1788|1731|

All-mode minus known-mode retention is **-2.6279pp**, family-bootstrap95%CI
[-4.8286,-.3741]. Versus untouched parent it is-6.5007pp[-9.0373,-4.1827].
Validity versus parent falls4.9479pp[-7.1181,-2.9514]; distinct modes-.388889
[-.5625,-.232639]; Brier increases.023020[.012389,.035360]. Retain parent.

The pre-registered descriptive decomposition explains an observed tradeoff:
among2002teacher-known opportunities, retained modes fall1730->1663; among167
unreferenced opportunities,58->68. The latter+5.988pp[1.075,11.656] does not
replace the failed aggregate primary outcome. The original parent retains76
unreferenced opportunities, more than either continuation. More verified mode
labels do not automatically improve fixed-budget set coverage. This observation
does not prove capacity competition is the sole cause of degradation.

Original weights and actual observation-index streams are exact matches to the
known-mode control. Grounding labels, complete scoring function,1200updates and
M8 are unchanged. Final matching RNG differs as expected with enlarged class
sets. Target slots rise1867761->1896439; no equal-target-compute claim. All new
classes are checked positive over-passage variants, not independent demos or
robot-execution certificates. Explicit TRAIN updates; no TEST_LOCKED.

Training source60e562f483fcda7455329b105b06b495c6e23885, archive
47237ae987e88b9915e446c116b8f90d48dc56d4a50b5eff2d2a4849855f452f.
All5coordinator jobs exit0; tests10pass; training273.277296command seconds.
Final checkpoint797347e205474d47baaa414f94f74512b929aaef52896d90435db4f8859a5837.
28closed files SHA verified. Actual commands/source/input hashes and full
checkpoint/optimizer/RNG are preserved by the closure manifest and receipts.

## Additional TRAIN audit: partial completion, overall job failed

Source dc5f5cc6b36c5113d95fc5b5819c771cc6561dc2, archive
5758e93d403a810f444f37aab320411821171407ec1aef0e965f9e5906b4d9b5.
Job all_mode_train_vocabulary_audit reached its50s cap: **exit124**, actual
50.535494command seconds including wrapper/termination overhead. All1152TRAIN
predictions and continuous checker results were saved; subsequent three-model
mode-vocabulary statistics were NOT completed and are not reported as results.
Seven partial-closure files SHA verified. No retry or workload moved elsewhere
to bypass this cap.

| Split/model | Valid | Post collisions | Wrong endpoints | All endpoints wrong requests |
|---|---:|---:|---:|---:|
| TRAIN known-mode /9216 |8842|192|187|11|
| TRAIN all-mode /9216 |8787|294|142|8|
| DEV known-mode /2304 |1926|220|175|16|
| DEV all-mode /2304 |1886|295|140|12|

Errors overlap. TRAIN loses55valid candidates while endpoint errors decrease45
and collisions increase102. The completed substage supports a geometry/endpoint
tradeoff in fitting; missing TRAIN vocabulary statistics prevent finer attribution.
The failed overall job stays in the budget and failure count.

Final recorded budget:148jobs142success6retained experiment failures;
7179.431180/7200command seconds used,20.568820remain. A separate earlier pre-job
relative-path launch failure is retained outside these experiment-job counts.
