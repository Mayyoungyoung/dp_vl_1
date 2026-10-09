# Stage A: verified target diagnostics, 2026-10-09

Source a6096e44122c14e56ab2613b14f6c429a539efe3, archive SHA256
0bba312813ca6326d9b74d694a36b9995469c6ae1efb4611bf59d89aa65bb349.
Server tests17pass. Preparation exit0 in18.781s, diagnostic exit0 in77.264s;
tests15.992s. All count toward the explicitly authorized additional7200s.

384 of1152 TRAIN requests have original known mode count<=8;768 were excluded
prospectively by that fixed teacher-only criterion.12480 original/perturbed
witnesses plus1536 corruptions exactly agree between training and independent
evaluation checkers. No model-based request selection and no TEST_LOCKED access.

On the first64 eligible TRAIN requests, the shared initial model has487/512 valid
raw candidates.2048 raw/target slots independently rechecked with zero disagreement.
The target correction diagnostic has these means (XYZ squared meters):

| Arm | All-route correction MSE | Correction MSE on valid routes |
|---|---:|---:|
| ordinary |.0003023433|.0002630562|
| gate |.0000740758|.0000215763|
| project |.0002348886|.0001876222|

Valid-route columns average per-request valid-route means, excluding undefined
requests with zero valid routes; they are not candidate-weighted population means.
Potential distinct valid representatives average6.7344/request. These are only
actually protected in gate/project. Correct duplicates can still be reassigned.
Project fallbacks average.2344/request.14.0625% of requests have an original-known
plus currently-valid mode union exceeding8, despite teacher mode count<=8; those
requests stay in the diagnostic and training accounting.

Replacing original canonical witnesses by first checked variants, with identical
mode support/exposure, changes ordinary targets by mean MSE .00000101069. Exact
duplicate references leave ordinary targets unchanged. The latter is a property
of the existing strong baseline, not a new algorithmic property.

Interpretation: verified correct routes can receive nonzero reference-fitting
pressure. This does NOT establish optimizer-induced degradation, mode forgetting,
causality, generalization or an advantage for the more complex project arm.
Project moves some correct duplicate slots more than gate; its incremental value
must be measured in training and all-candidate evaluation.

Artifacts: stage_a/diagnostic.json and prepared.json, with original source/payload
hashes. A local summary print initially failed on a None mean; rerun explicitly
excluded undefined conditional means without changing source data. Local torch
import failure/skip and successful17 server tests remain separately recorded.

Next: same initial checkpoint, TRAIN population, optimizer and observation stream
ordinary/gate/project seed0, fixed1200 steps. All288 DEV requests and the unchanged
complete scorer remain the evaluation denominator and scoring contract.
