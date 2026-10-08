# Frozen input encoders x verified-positive augmentation

Ordinary seed0 factorial on288 DEV_MODEL requests/32families; no new algorithm or independent test claim. All models use the same complete paired-domain q seed0.

| Model | Valid@8 | Any valid | Distinct@8 | Rare recall | Brier | Retained/2169 |
|---|---:|---:|---:|---:|---:|---:|
|Parent|0.868056|0.958333|6.680556|0.744068|0.075390|1872|
|Unfrozen plain|0.872830|0.961806|6.670139|0.738426|0.077099|1775|
|Unfrozen augmented|0.835938|0.944444|6.395833|0.720139|0.100006|1788|
|Frozen plain|0.816406|0.947917|6.131944|0.711603|0.095825|1717|
|Frozen augmented|0.834635|0.958333|6.309028|0.736603|0.086163|1777|

Frozen augmentation-minus-plain retention: 0.027663,95%CI[0.008181632400297251, 0.04993537162954837]; registered gate **False**. Interaction (frozen augmentation effect minus unfrozen augmentation effect): 0.021669,95%CI[-0.02237460296047387, 0.06991948930512958]. Ordinary plain-freezing versus parent gate **False**.

Frozen geometry/feature/state tensors have identical initial/final hashes. Grounding loss is computed but constant with respect to remaining trainable parameters. Bounded endpoint residuals still train; freezing alone does not guarantee correct endpoints. Same1200 updates and actual input/group RNG; augmented targets require more processing.

The retention denominator uses fixed safety_mean source witnesses for all destination models. Confidence intervals resample families, not generator seeds. Earlier unfrozen arms are reused sealed runs. Shared loader materializes permitted DEV caches, but only TRAIN enters updates. No TEST_LOCKED, q retraining, robot execution claim or post-result threshold change.

![Family intervals](factorial.png)

Full metrics, direction counts, failure details and provenance remain in the closed artifact archive; compact results accompany this report.

## Full TRAIN and DEV error attribution

| Split/model | Valid | Post collisions | Wrong endpoints | All endpoints wrong requests |
|---|---:|---:|---:|---:|
| TRAIN parent /9216 |8937|189|91|7|
| TRAIN unfrozen plain |8883|261|75|6|
| TRAIN unfrozen augmented |8842|192|187|11|
| TRAIN frozen plain |8884|177|161|13|
| TRAIN frozen augmented |8885|228|109|6|
| DEV parent /2304 |2000|207|108|12|
| DEV unfrozen plain |2011|200|106|11|
| DEV unfrozen augmented |1926|220|175|16|
| DEV frozen plain |1881|295|151|15|
| DEV frozen augmented |1923|285|115|12|

Errors overlap; never sum them as exclusive categories. Frozen augmentation
reduces TRAIN endpoint errors52 but adds51 collisions, leaving valid count+1.
On DEV it reduces endpoint errors36 and collisions10 versus frozen plain, yet
remains below the untouched parent. Frozen inputs alone do not preserve task
endpoints or solve generalization. These are fixed-seed development results.

Training/evaluation source d3e48492b9eda3c5b665ab972285c21eb5ed8675;
source archive f447f6b41d18bcce7bd6c04f596a04289073365d173d8f47c043070c993c26cf.
All6jobs exit0, coordinator549s;10tests pass. TRAIN audit source
f96b94d4c308c91a3933105e1726155ac69162e7, source archive
d5b4881b20b58180540a780070526f91fa8b10707ee2a9f4483ef648df6fad2e;
two audit jobs exit0 in21.598520/21.375803 command seconds. Actual commands,
full source/input hashes and checkpoints/RNG are in closure receipts.41factorial
and12TRAIN audit files individually SHA verified.142jobs137success5retained
experiment failures,6814.223800/7200command seconds spent,385.776200remain.
A pre-job relative-path launcher error was preserved separately; absolute-path
retry used unchanged immutable source. No active queue; retain safety_mean.

The figure was visually checked; trailing whitespace in generated SVG source
was trimmed without changing its graphics. This report's TRAIN attribution was
added after the read-only figure/report generator; source results remain sealed.
