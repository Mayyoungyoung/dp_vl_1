# Feasible-space representation v1 — prospective protocol

Start99fd106, local writer, codex/multiroute-v2. Only wzy3090 GPU1 UUID
GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab, .35 memory and four CPU threads.
User's attached2026-10-09 brief removes historical time/update caps. This does
not authorize other servers or altered environments. New ledger only.

Reuse1152TRAIN/288DEV_MODEL and56920 verified positives, frozen observation/VLM
encoder and complete scorer. All oracle geometry stays in TRAIN label preparation
or independent evaluation. No raw collector, reserved TEST_LOCKED, scorer fitting,
robot execution or deployment default changes. DEV is reused development evidence.

## First representation

Ordered swept boxes C_j=conv(c_j+[-r_j,r_j]^3,c_{j+1}+[-r_j,r_j]^3).
Each is convex, including the whole segment. Shared vertex j lies in the cube
around c_j with radius min(r_{j-1},r_j); endpoints are fixed. Output24 nodes is
itself23 degree-one Bézier pieces. No hidden curved resampling that cuts corners.
The .8 exact signed L-infinity clearance slack protects all points of each cell
relative to expanded obstacle boxes and workspace floor, conditional on truthful
reference geometry. Predicted cells carry no real-world safety certificate.

This is a conventional convex construction, closely related to existing certified
coordinate regions; novelty is NOT asserted for its mathematics. Test the learned,
companion-independent corridor boundary coupled to bounded, peer-aware generation
against same-information free XYZ and projection. Corridor(x,m,v) allows two
geometric implementations for a mode; companion queries never change its boundary.

Stage A: measure all-witness certificate coverage and two-prototype corridor
coverage separately, check canonical averages/medoids, full polyline and modes.
Compare oracle-corridor centerline, free and bounded outputs on TRAIN only.
Do not call oracle-corridor values deployment scores.

Stage B: light observation/context+mode+variant corridor predictor, output centers
and radii; joint relative decoder. All models receive identical corridor supervision,
observed features, positives and frozen scorer. Boundaries only see their own query;
peer-aware ablation allows other queries to alter boundaries. Never substitute
true radii/centers at formal inference. Predict conservative radii and report
unknown visibility rather than treating unseen space as observed free.

Stage C seed0 screen: free XYZ+same corridor features, unbounded relative,
bounded map, and peer-dependent boundary ablation; equal2400updates, draws,
initial parameters. Centerline and post-XYZ projection reuse exactly the frozen
predicted corridor and decoder. C+success and Gate are historical references.
Matching-snapshot TRAIN realization feedback and ordinary success heads are required
before comparison with historical C+success. Report fixed-query and refreshed-head
effects separately; no stale C success labels on the changed decoder.

Primary U8 actual distinct valid modes, meaningful gain .15 over strongest matched
simple arm; family-paired95%CI excludes0. Guards V8-.01,V4-.005,U4-.03. Preserve
2169 survival /270 adaptation opportunities. Candidate/query identities differ
from actual words. Route validity is end-effector task-level only.

If a stable signal exists, freeze architecture/schedule then compare exactly three
full generator seeds0–2 against strongest matched control, without seed extension.
Independent new scene families must be registered/generated from observation
rendering after freezing. Never relabel existing DEV as untouched generalization.
Whole-arm execution is conditional on mechanism benefit, never inferred from paths.

Every repair must address an observed cause and be registered before outcomes.
Checkpoints include optimizer/RNG/sample stream and source/data hashes. Job receipts
record commands, failures, budget and actual imported source. No running source edits.

## Envelope repair registered after first XYZ screen, before repaired outcomes

Same-info XYZ2400: cell feasibility7.552% while route validity86.762%. Only298/305
invalid routes have uncertified predicted cells;7fail despite certified cells.
Initial weighted clearance gradient norm .2219 vs route .01997. Thus final-path
clearance alone does not train the whole predicted region. Radius labels measured
around reference centers are optimistic around shifted predicted centers.

Add cell-envelope supervision to BOTH same-info XYZ and bounded arms, from the
same C initialization, seed0, same draws/2400updates. Use existing160clearance
scale, no DEV weight search: `L_cell=mean(relu(r-.8*min(segment_slack,floor_slack))²)`.
Exact segment slack of predicted centers supplies both center/radius gradients,
using TRAIN truth only. This is conventional feasibility supervision, not claimed
novelty. Record radius collapse, endpoint failures and final selected metrics.
Run fresh matching feedback/success fit for both repaired decoders, not obsolete
screen snapshots. Compare centerline and projection of each repaired snapshot.
