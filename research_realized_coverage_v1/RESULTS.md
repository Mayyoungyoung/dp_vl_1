# Results: a stronger ordinary control, no validated new coordination mechanism

All discriminating comparisons completed. Interaction-aware net coverage did
**not** beat ordinary realization-success allocation. The useful new research
option is frozen C plus the simple success head; the historical deployment
default remains unchanged. No submission-ready claim.

## Registered paired comparison

All 288 reused DEV requests / 32 layout families; 1152 TRAIN requests; identical
two-context feedback and actual minibatch stream within seed; 2400 updates,
final checkpoint, seeds 0–4. These are **allocation-head training seeds on fixed
C0**, not five complete generator training runs.

|Method|Validity@8|Distinct valid modes@8|Known recall@8|Validity@4|Distinct valid modes@4|
|---|---:|---:|---:|---:|---:|
|C0 adaptive, fixed reference|89.106%|6.7639|79.351%|94.271%|3.7431|
|Ordinary success, five-seed mean|89.149%|6.9806|82.319%|94.340%|3.7736|
|Dense interaction net, five-seed mean|89.184%|6.9465|81.712%|94.271%|3.7701|

Dense minus success U8 = **−0.0340**, 95% crossed seed/layout-family interval
**[−0.0743, −0.0014]**; family-only [−0.0514, −0.0174]. This is conditional
reused-development evidence, not an independent final-test inference. It fails
the registered +0.15-word criterion. No favorable seed was selected.

|Seed|Success U8|Dense U8|Success V4|Dense V4|
|---|---:|---:|---:|---:|
|0|6.9826|6.8854|94.358%|94.358%|
|1|6.9861|6.9410|94.271%|94.010%|
|2|6.9757|6.9757|94.358%|94.271%|
|3|6.9792|6.9757|94.358%|94.271%|
|4|6.9792|6.9549|94.358%|94.444%|

Full per-seed metrics, every curve checkpoint, ablation and joint model are in
[complete tables](results/RESULTS_TABLES.md) and [results JSON](results/RESULTS.json).

## Fixed scene-edit witnesses

The unchanged parent defines 2169 surviving-mode witnesses: 1872 originally
covered and 297 missing. The 270 fixed geometry-repair opportunities include
255 that invalidate every original source route of that mode.

|Method|Retained /2169|Lost /1872|Recovered /297|Repair /270|Strict repair /255|
|---|---:|---:|---:|---:|---:|
|Historical parent|1872|0|0|200|187|
|C0 adaptive|1742|204|74|207|195|
|Success, five-seed mean|1779.6|161.0|68.6|235.2|222.2|
|Dense, five-seed mean|1776.8|162.2|67.0|235.6|222.6|

Dense-minus-success retention = −0.129 percentage point, crossed interval
[−0.609,+0.413]; repair = +0.148 point, [−0.727,+1.092]. Neither supports extra
preservation/adaptation. Both remain below parent retention 86.31%. Fractional
counts are seed means.

Seed0 success: 235 valid same-mode repairs, 30 same-mode but invalid, 5 other
valid modes only. Dense: 236, 29, 4 respectively, plus 1 all-invalid case.
Zero cases receive the copied-old-coordinates category (1 mm threshold);
successful same-mode repair takes precedence if both behaviors occur in a set.
Merely producing a different valid mode does not count as repair.

## Geometry and historical comparisons

Four displacement-free arms ran 3600 steps with 600/1800/3600 curves. All adaptive
continuations deteriorated. Matching-generator feedback was recollected and both
ordinary/dense heads refitted for ordinary, gap and hard 600-step generators.
Gap+dense achieves U8=6.9826 but V4=93.663%, 0.694 percentage point below frozen
C0+success seed0, beyond the registered 0.5-point margin. Retention is 1746/2169
and repair 232/270. Geometry screens share budgets but have branch-dependent RNG;
they are exploratory. No five-seed joint-generator success is claimed.

On identical feedback, scalar-net U8=6.7743, no-peer=6.9444, added-only=6.9201,
dense=6.8854, success=6.9826 (seed0 final). See [ablations](ABLATIONS.md).

Historical Gate/Set-Point are reused references, not matched new-data controls:
Gate V8=92.12%, U8=7.238, recall=65.80%, V4=94.33%, U4=3.772, retention=70.20%;
Set-Point 91.58%, 7.322, 68.45%, 93.95%, 3.756, retention=71.62%.
New success trades lower raw validity/coverage for higher known recall and
retention; it does **not** dominate these historical methods.

## Actual output, cost and figures

Each evaluation decodes eight routes once, then full frozen q returns four.
Public-API replay has zero path/event/q error and identical selected indices.
A 20-repeat single-input benchmark measures 6.782 ms success / 11.925 ms dense,
including observed encoder, allocation, decoder and full q, excluding Qwen and
disk input. This is a limited cached-feature microbenchmark.

[Artifact closure](results/CLOSURE.json) records all costs, failed jobs and
hashes. TRAIN counterfactuals and initial discarded decodes are explicitly charged.
There are 25 main training runs, 4 recovery-check runs and 45 full DEV evaluations.
101 serial jobs consumed 2679.381 seconds in total (100 exit0; initial collector
failure retained). This is cached-feature job wall time, including CPU analysis,
not full VLM compute or total elapsed user time. Feedback checked 1,501,152 routes
including the diagnostic DEV pool; an extra 11,520 initial decoded routes were
discarded. Local verification matches 6415 files and 10 immutable source exports.
Real prediction figures: [repair](results/actual_figures/case_0.png),
[lost mode](results/actual_figures/case_1.png),
[same-mode invalid](results/actual_figures/case_2.png),
[rendered RGB-D](results/actual_figures/actual_rgbd.png).
All four were visually inspected. Cases are deterministic diagnostics, not
estimates of average advantage.

No TEST_LOCKED, independent new-distribution claim, whole-arm execution or default
change. The study closes on comparative negative evidence, not an old time cap.
