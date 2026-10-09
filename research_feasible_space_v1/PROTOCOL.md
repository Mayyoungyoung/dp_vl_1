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

## Reachable-cell repair registered before tapered outcomes

Uniform-cell supervision increases XYZ cell feasibility7.552%->71.181%, but
semantic endpoint failures137->172 while collision/floor failures185->167.
Whole swept cubes expand around fixed endpoints, although the generator cannot
move those endpoint nodes. This introduces unnecessary endpoint/floor pressure.
Replace the supervision region by its exactly reachable convex set:
`conv(c_j+[-rho_j,rho_j]^3,c_{j+1}+[-rho_{j+1},rho_{j+1}]^3)`,rho endpoints0.
Containment mapping stays identical; endpoint boxes now taper to a point.
Exact continuous clearance is the minimum of max6affine faces minus a linearly
varying radius, attained at endpoints or face intersections. This is analytic,
not random point sampling. Keep all old uniform metrics and name the new geometry.
Both XYZ/bounded receive the same tapered supervision, same C initialization,
same2400draws/loss scale; collect new TRAIN feedback/heads. No independent novelty
claim for tapering. Confirm whether endpoint tradeoff and true mode failures improve.

## Final representation replication lock, before inspecting tapered seed0 results

Complete exactly three paired full generator continuations0,1,2 for tapered
XYZ/bounded,2400steps final, matching new feedback and success2400heads. Seed0
screen is reused. This definitive mechanism check is committed irrespective of
seed0 sign, with no significance-seeking seed extension. It supersedes the earlier
conditional seed expansion solely for this final representation. Do not weaken
the .15U8/quality gates or declare success for a significant smaller effect.
Compare bounded against matched XYZ, projection and centerline, reporting every
seed and conditional family/crossed intervals. All start from shared historicalC0;
these are three generator continuations, not independent pretraining runs.
Independent new-family/robot expansion remains conditional on accepted mechanism.
The prospective additional runs are seeds1/2; seed0 is explicitly reused screening
evidence. The lock was made before reading its output, not before its training launch.

Shared-query center/projection outputs isolate the geometric mapping, but their
proposal labels originate from the parent XYZ/bounded output. To give strongest
standalone controls fresh matching feedback, create immutable decoder views with
unchanged weight tensors and declared center/projection mapping, NEW checkpoint
hashes. For each3seeds collect identical TRAIN16word/two-slot outcome design and
fit success2400 on that actual decoder. Retain shared-head ablations separately.
Main acceptance must pass both shared and independently refitted controls; no
claim from a postprocessor inheriting mismatched success labels.

## Frozen fresh-family diagnostic extension, before collection

The user's requested new-scene evaluation is completed even if the representation
fails the old DEV acceptance gate. This supersedes only the conditional collection
clause above. It does not authorize another tuning round, a relaxed gate or a
deployment/robot claim. All generator/head checkpoints already exist and stay fixed.
Config configs/feasible_space_generalization_v1.json fixes seed641009,16new families,
80physical scenes (open,shifted,closed,narrow,tall),32derived observation-only noisy
or centrally occluded scenes,336requests. Every family is assigned DEV_MODEL and
evaluation-only; none enters TRAIN. Check geometry hashes against existing paired
TRAIN/DEV registration, without reading any reserved payload. One rendering worker,
software GL, existing simulator/Qwen environments, GPU1/.35memory and CPU0–3 only.
Retain rendering failures and denominator shortfalls; never substitute scenes.
Compare all3frozen continuations of XYZ,bounded and three independently refitted
controls. No checkpoint selection. Fresh recall uses incomplete geometric teacher
witnesses, so is not directly comparable to old all-mode-support recall. Geometry
variants are actually rendered; noise/occlusion are explicitly synthetic current
observation corruptions, not physical sensor experiments or robot execution.

## Saved-parameter portability diagnosis, before measuring its outcome

For each of the3frozen bounded continuation pools, use all existing directed
parent-witness edit edges. Eligible slots have an actual valid requested source
word and that word requested at destination. Keep source bounded relative node
coordinates `(p_source-c_source)/rho_source` fixed; reconstruct with destination
predicted centers/widths. Compare copied absolute coordinates, transferred relative
coordinates, destination centerline and destination native route, all with the
same destination predicted events. Check task-valid/same-word outcomes and repairs
among copied-coordinate failures. These are matched-slot diagnostic denominators,
not replacements for2169survival/270adaptation opportunities. No new decode, oracle
repair, learned-head fit or deployment selection is introduced. Report everyseed
and the center control even if it makes neural portability unnecessary.
