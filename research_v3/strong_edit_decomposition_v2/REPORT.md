# Strong-baseline edit retention: separate capacity and endpoint errors

The current safety_mean still loses some concretely feasible modes after
edits even when redundant/invalid slots could be replaced. This is not solely
a shared endpoint-localization error, but semantic failures explain a substantial
part of the raw opportunity. The audit supplies no trained improvement or
new-method claim, and does not justify reviving the old R2/R3 losses unchanged.

Source `934cb15c1718371319daf7ade38994ae8e78c19e`; archive SHA
`f13a2f8b1c391723152578b4851ff222005c2590a4343028918dbe44dd51cfc8`.
3 tests pass in.18s, wrapper.785085s; all5-model check exit0 in12.290352s.
All inputs are sealed DEV_MODEL predictions. The same concrete source path is
checked in destination geometry; only valid surviving paths certify a mode.
No new model forward, rescoring, optimization or TEST_LOCKED access.

## Current reference generator

Each direction has96 target pairs across32 families. A mode opportunity is
counted per request, so repeated requests/directions are not independent data.

| Direction | Surviving mode opportunities | Lost | Replaceable-slot feasible | Feasible using correct-endpoint slots | Pairs with latter opportunity |
|---|---:|---:|---:|---:|---:|
|open→closed|496|50|46|23|21|
|open→shifted|642|78|51|40|27|
|closed→open|378|84|35|23|15|
|shifted→open|653|85|54|23|18|

The first feasible column allows replacing invalid candidates or classified
valid duplicates. The stricter column allows only invalid candidates whose
endpoint is already correct, plus classified valid duplicates. Each added
mode has an actual checked path, and M8 is unchanged. Valid unclassified paths
consume a slot. These are oracle constructions, not an inference algorithm.

For open→closed,23 of46 opportunities occur in5 requests with every endpoint
wrong;23 do not require replacing a wrong-endpoint candidate. For shifted→open,
31 of54 opportunities lie in such shared semantic failures. Thus raw edit
loss is not a clean mode-preservation metric. The remaining stricter per-pair
means are.2396,.4167,.2396,.2396; family-bootstrap95% intervals respectively
[.1354,.3542],[.2500,.6146],[.1250,.3750],[.1146,.3854]. These are descriptive
uncertainty for a fixed seed0 model, not multi-seed or method-comparison results.

The edits are not assumed to be nested obstacle removals: even closed→open
invalidates281 source-valid routes. Actual checker results, not edit names,
determine survivorship. Opening a named passage does not guarantee every other
path remains feasible under the full edit.

## All models retained

Cells show all replaceable-slot opportunities / stricter correct-endpoint
opportunities. Different models have different source witnesses and destinations;
these counts are diagnostics, not fair standalone model-ranking metrics.

| Model | open→closed | open→shifted | closed→open | shifted→open |
|---|---:|---:|---:|---:|
|safety_mean|46 / 23|51 / 40|35 / 23|54 / 23|
|safety_worst|42 / 21|47 / 28|41 / 17|67 / 17|
|margin_mean|65 / 51|43 / 36|39 / 28|42 / 21|
|optimizer_restored|45 / 17|47 / 31|32 / 22|61 / 29|
|original-witness stage1|48 / 28|60 / 45|60 / 29|63 / 28|

## Incomplete reference support is only part of the explanation

For safety_mean, lost known/unreferenced modes are39/11,63/15,39/45,65/20
in the four directions. Among stricter opportunities, known-mode opportunity
counts are17,35,16,18; unreferenced-mode opportunity counts are7,6,8,6.
These last two columns compete for the same slots and **must not be summed**.
For example17+7 exceeds the23 total in open→closed. Unreferenced means absent
from the supplied witness set, not invalid: each has a checked positive path.

Most stricter open→closed/shifted opportunities concern modes already represented
in destination supervision. Simply adding more nearby variants is not yet a
supported explanation or remedy. First measure the analogous TRAIN residual
and witness geometry before registering another pair-consistency or augmentation
experiment. Historical R2/R3 negatives and ordinary controls remain required
comparisons; no new training is automatically justified by these oracle counts.

## Reproducibility

The6 remote artifact files are SHA verified in
[STRONG_COUNTERFACTUAL_CLOSURE_ARTIFACTS_20261009.json](../STRONG_COUNTERFACTUAL_CLOSURE_ARTIFACTS_20261009.json).
Local source data hashes and every model/direction/subgroup are in RESULTS.json
and rows.json. LOCAL_ANALYSIS_PROVENANCE.json records the actual script hash and
commands. The earlier semantic-only v1 outputs replay exactly with that source;
v1 is preserved. Generator q functions differ among cached pools but are never
compared or reused for transferred paths. No probability conclusion follows.
