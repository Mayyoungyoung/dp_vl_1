# Mode-conditioned geometry generation — registered before first run

The user requests a new generation mechanism. Historical R2/R3 aligned relative
crossing descriptors of unconstrained output sets; they did not decode named
modes. This experiment changes route tokens into semantic passage embeddings,
retains the shared observation-conditioned decoder, and directly supervises
nonzero source-to-destination path displacement under a witnessed shared mode.
The mode vocabulary is controlled two-row passage words, not homotopy classes.

Budget: use the verified-set additional-7200 ledger, confirmed 51 terminal jobs,
2935.782764 s spent / 4264.217236 s left. No increase. GPU1 UUID and .35 memory,
CPU affinity0-3, immutable local commits exported to wzy3090 only.

B/C/D share safety_mean initialization, frozen visual-language and observed
geometry encoders, 1152 TRAIN requests and all 56920 previously verified TRAIN
positives. No new positive collection. Six of eight training targets per scene
are positively witnessed same-mode pairs; remaining targets sample other modes
from the full single-scene pool. No unknown mode is a binary negative.
Closed inflated internal passages alone give certified negative word evidence.
Proposal scores use positive normalized allocation likelihood plus masked BCE;
low proposal score is not an absence certificate. Inference uses current inputs
only and emits eight paths. Top-eight score-ranked words are the main deterministic
allocation; categorical sampling with replacement is a fixed-model ablation.
Thus top-eight is a budget heuristic, not a claim of eight feasible classes.

B: original learned query set, ordinary Hungarian matching to the same sampled
mode-balanced targets, same allocation head, same pair displacement objective.
C: semantic mode embedding queries + shared decoder, single-scene objective only.
D: C + supervised generated-path displacement between same-mode scene pairs.
B is deliberately given D's pair information and objective, so D cannot win from
receiving more pair labels. C shares all targets but omits the pair objective.
Use fixed final checkpoints, AdamW .0003, batch32, lambda_mode .001,
lambda_pair1, inherited clearance coefficient160. Encoders/scorer stay frozen.
Mode embedding initialization averages nearest parent query embeddings on TRAIN,
and is shared in all arms (unused for B). Decoder weights are the same parent.
Loss_route includes XYZ/event fit and inherited TRAIN-only geometry penalty;
only three loss groups, no uniform coordinate consistency or new scorer.

Pilot: B/C/D seed0, 400 updates, all288 DEV and fixed2169 witnesses. Examine actual
condition compliance, invalidated-old-coordinate same-mode repair and quality@4.
If necessary permit one diagnosis-driven structural repair, change one factor,
then full1200 steps and replication only after an improvement signal.
Prospective useful-signal criteria: D over same-information B by >=.3 valid modes,
validity no worse than1pp, returned4 validity no worse than1pp, fixed retention
and same-mode adaptation improve; D over C must show a positive pair increment.
These are DEV development gates, not inferential success or test generalization.
No seed selection; group uncertainty by32 layout families. Keep every failed run.

Adaptation denominator: original parent-valid routes that fail the destination
checker AND have a valid destination reference with their original operational
mode. Fixed independently of new models. Deduplicate by (source,destination,mode).
Report valid same-mode repair, copied old coordinates (<1mm mean point distance),
same-mode invalid output, other-mode-only valid output, and no valid output.
These observed categories are proxies for behavior, not inferred model intent.
Retention denominator remains2169, split into1872 previously retained and297
previously absent opportunities. All comparisons use the same frozen source.

No TEST_LOCKED access; no full-arm execution or independent generalization claim.
Keep historical defaults. Publishability is contingent on evidence, not embedding
or pair-loss implementation alone. All commands, checkpoints/RNG and actual source
hashes saved. Never overwrite launched code, outputs or immutable launcher.

## Full-step control addition after first B/C pilot

400-step C reaches87.28% validity/6.653 modes versus pair-weighted ordinary
B74.83%/5.958. This is a structure signal but B's displacement objective could
be hurting its ordinary outputs. Before declaring an advantage run B0 with
identical data/mode/pair target draws and lambda_pair0. It is the requested
ordinary weighted-set baseline; B retains the extra same-pair-objective control.
Run B0/B/C/D for1200 fixed steps, seed0, no hyperparameter changes. The strongest
ordinary arm determines comparison. D must improve over C to support H2/H3.
This clarification is registered before D pilot metrics and full-step outcomes.

## Inference interface correction before reading full-step outcomes

The requested interface must allow fewer than8 proposed words. Add `adaptive`
allocation: use at most8 words with logits>=0 (a .5 sigmoid proposal threshold),
fill spare paths with within-mode variants; if none pass, use the highest-ranked
word. This is not an absence certificate. Keep original `balanced` and `ordinary`
results, evaluate adaptive on the SAME fixed checkpoints, and expose it as the
model API default. This changes no training data, decoder weights or oracle use.
Report strategy-dependent effects instead of attributing them to architecture.

## One structural repair, registered after full-step failure and TRAIN diagnostic

At1200steps D86.85%/6.660 and return4valid93.14% underperform C89.24%/6.778 and
94.79%. Do not choose the better400-step D. TRAIN first128 changed-coordinate
pairs (466 mode opportunities) show D398 correct-valid modes with repeated
companions versus449 after changing companions; mean route shift7.675mm.
C similarly shifts8.317mm. Swapping destination context to source context reduces
correct-valid D398->326, so input geometry matters, but companion interference
is a concrete representation problem. Displacement regression error alone does
not imply route validity or semantic compliance.

Change ONE architectural factor: bypass candidate self-attention in shared
decoder blocks, leaving per-token context modulation/MLP/output shared. Train
C_ind/D_ind1200steps seed0 with unchanged losses, targets and frozen encoders.
Compare adaptive allocation for both old/new checkpoints. No target/weight sweep.
If repaired D has no useful advantage over C, accept lack of pair evidence;
do not keep adding modules. If representation has a useful signal, replicate
the relevant ordinary/C/D controls for seeds1,2, reporting all seeds.

## Resume check catches target-order bug before structural repair launch

The actual100 versus50+50 process-resume check fails: model, optimizer, RNG and
target-stream hashes differ despite matching settings. Inspection finds a Python
set-derived mode dictionary feeding spare-target rng.choice. This is randomized
across processes. Therefore earlier pilot/full runs are exploratory, NOT verified
paired target-stream comparisons. Preserve every output and failed audit.
Canonicalize dictionary keys, then require actual exact-resume success before
further scientific runs. The previously exported independent queue was NOT
launched. New canonical seed0 queue runs B0/C/D and the single communication-off
C/D repair,1200steps each, same adaptive allocation. Check their actual streams
before comparison. These corrected comparisons supersede earlier causal readings;
the within-checkpoint TRAIN companion intervention remains valid.

## Strong ordinary full-positive baseline

Before reading canonical full-step results, add B_set: same parent, modes,
observations,1200updates and8outputs, but minimize matching over ALL previously
available positive coordinates within each sampled mode. B0's sampled single
pair representative could be unnecessarily restrictive for an ordinary set.
B_set's same-mode min followed by Hungarian is a conventional stronger ordinary
control, not another proposed method. It receives the entire same56920-positive
pool; extra per-step reference processing cost is disclosed. No new labels.
Compare against the strongest ordinary baseline rather than only B0/B_pair.

## Three-seed decision

Canonical seed0 C89.11%/6.764 and D88.32%/6.726 exceed B_set77.43%/6.177;
representation has a useful signal, but D does not outperform C in returned
quality/retention. Communication-off C/D have lower aggregate DEV coverage and
repair than communicating C/D, despite zero companion drift on TRAIN. Reject
that repair for expansion. Replicate B0/B_set/C/D for seeds1,2, unchanged1200steps.
This tests stability of H1 AND the negative H2 result; it is not provisional
acceptance of D. Do not select a seed or revert to400step checkpoints.
After this bounded diagnosis/repair/replication cycle, report failure if selective
pair supervision still lacks independent benefit. Remaining budget is not a
reason to conduct unmotivated weight sweeps or add modules.
