# Fixed ordinary anchor intervention

TRAIN mass gate source cb695826f46d9256ea908c9ca02ff62e3de4ade3 passed:
peak6 vs mass1 impossible anchors, within3cm1146 vs1151 /1152 requests.
Register before DEV intervention: same safety_mean weights, replace only peak
anchor rule with exact all-observed-point Gaussian mass peak sigma.025m.
Recompute ordinary context and route head naturally, without endpoint repair.
No training, additional candidate generation, oracle input, temperature fit or
new scorer. Original full q observation encoder remains fixed independently.
M8/H24 and public q-first K4 selector unchanged. Extra inference cost reported.

Use every288 paired DEV_MODEL request, compare sealed original safety_mean
evaluation_fixed_q_v2 against fresh evaluation_anchor_mass_v1. Paired family
bootstrap32 families,10000 draws,seed610091, report all original metrics.
Gate: raw candidate validity gain>=.01 and95%CI lower>0; minority recall delta
>=-.01; any-valid request fraction no worse. This is a pilot gate conditional
on one trained seed, not a novelty or multi-seed conclusion. If false, reject
intervention and do not tune bandwidth. No scale or alternative anchor sweep.
