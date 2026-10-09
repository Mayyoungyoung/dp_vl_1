# What Does Feasible-Space Parameterization Buy in Observation-Conditioned Multi-Route Generation?

Research draft. This is not a claim of an accepted novel algorithm. The quantitative
main comparison is populated from recorded experiments in RESULTS.md and
results/three_seed_statistics_v1.json. The bounded decoder fails the locked gate.

## Abstract

Generating a finite set of useful end-effector routes from a single RGB-D view
and language requires geometric feasibility and coverage of different passage
patterns. We test a mode-conditioned network that predicts bounded convex cells
independently of companion queries, followed by a peer-aware decoder mapping
parameters into those cells. Connecting segments satisfy an algebraic containment
property; incorrect predicted cells do not certify scene safety. TRAIN-only oracle
labels represent all 56,920 positive routes individually, with 89.97% covered by
two per-mode prototypes. All three oracle controls attain 100% task validity on
1,024 reference-cell paths, offering no evidence that a learned interior decoder
is needed. Three matched generator continuations on 288 reused development
requests produce 6.803 valid modes@8 for bounded mapping, versus 6.718 for free
XYZ and 7.110 for independently refitted centerline controls. The 0.086 gain over
XYZ fails the registered 0.15 meaningful-gain threshold. Frozen diagnosis on
336 requests from 16 newly rendered families yields 6.403, 6.265 and 6.832,
respectively, preserving the deficit to centers. Fixed-parameter transfer improves
on copying old coordinates, but also trails new centers. We therefore separate
a correct containment property and a measurable representation effect from an
unestablished neural interior-generation contribution. No full-arm or real-robot
capability is claimed.

## 1. Introduction

Vision-language-conditioned trajectory models may provide several plausible ways
to reach a target. A finite output budget makes each invalid or duplicated route
costly. Candidate attention can coordinate such a set, but replacing one query can
also change the geometry of other routes. A geometric output representation offers
one way to restrict this coupling: define a region for each mode independently,
then allow interaction only through parameters constrained to that region.

This proposal separates three questions. Can existing valid routes be represented
by the chosen regions? Can those regions be estimated accurately from the legal
observations? Does neural interior generation add value beyond a centerline or
projection using precisely the same regions? These questions must be answered
together. A perfect oracle corridor can already contain a complete feasible route;
feeding such a corridor to a neural decoder does not by itself demonstrate a
useful learned mechanism. Conversely, a bounded output can remain unsafe when the
observation-conditioned region is wrong.

The present study is deliberately narrow: a controlled two-row end-effector reach
task, sixteen operational passage words, eight24-pointXYZ/event candidates and an
unchanged scorer returning four. The aim is to measure the contribution of a
feasible-space parameterization, rather than claim general homotopy reasoning or
robot safety. Claimed contributions must be limited to what survives same-information
controls. At this stage the verified contributions are an implemented comparative
pipeline, exact conditional containment, reference-coverage measurements, and
diagnoses separating region prediction, task semantics and candidate interaction.

## 2. Related work

[3D HAMSTER](https://arxiv.org/html/2606.31329v1) demonstrates RGB-D/language
metric3D trajectory generation and downstream control. The multimodal input and
3D output therefore provide task context, not novelty here.
[CorDriver](https://arxiv.org/html/2504.07507v2) learns corridors and plans
trajectories within them using differentiable optimization. A learned corridor
followed by trajectory construction is direct precedent.
[Graphs of Convex Sets](https://manipulation.mit.edu/trajectories.html) and
[Neural GCS](https://arxiv.org/html/2608.15440v1) provide convex-set planning and
learned acceleration of mixed discrete-continuous planning. Convex regions,
amortized planning and Bézier containment are established tools.
[C-IRIS](https://arxiv.org/abs/2302.12219) certifies regions in a rational
configuration-space representation. Our Cartesian end-effector cells and learned
observation priors do not provide that whole-robot guarantee.

The distinction under investigation is companion-invariant mode/variant boundaries
coupled to peer-aware bounded generation under incomplete observation. Its independent
value must exceed conventional XYZ, relative-coordinate, centerline and projection
methods with the same geometric supervision. No novelty is asserted for any one
of the conventional components.

## 3. Method

The frozen RGB-D/language encoder gives context h and a target surface anchor a.
The observed robot state supplies the start. Each requested operational word m
and geometry variant v produces a token h+E_m+V_v. The boundary network processes
that token with sequence length one, predicting24center nodes and23cell widths.
The independent construction prevents other query identities from changing those
boundaries. A second decoder sees all8tokens plus the predicted cell features and
outputs parameters z. No truth boxes, reference routes, source trajectories or
checker outputs enter this forward pass.

Let rho_0=rho_23=0 and rho_j=min(r_{j-1},r_j) for interior nodes. The final cells
are the reachable convex sets

`C_j=conv(c_j+[-rho_j,rho_j]^3, c_{j+1}+[-rho_{j+1},rho_{j+1}]^3)`.

Construct `p_j=c_j+rho_j*tanh(z_j)`. Each p_j belongs to both neighboring cells;
the segment[p_j,p_{j+1}]belongs to C_j by convexity. These are23degree-one Bézier
pieces, so the public24-node polyline inherits containment without cutting corners
during resampling. This does not provide velocity, curvature, orientation or
full-arm feasibility.

For truth-geometry TRAIN supervision, the signed L-infinity segment clearance to
an inflated obstacle box is the minimum of a maximum of six affine functions.
Subtract the linearly varying node width, and its minimum is attained at the
interval endpoints or one of15pairwise face intersections. The tapered cell
certificate is therefore analytic. A workspace-floor bound is handled by the
two endpoint lower faces. These calculations supervise predicted centers/widths
but are absent from deployed generation. Predicted regions can be wrong or
unobserved; even correct geometric cells can carry a wrong passage word or goal.

All trained arms receive the same route, corridor-center, width, continuous
clearance and reach-event supervision. The final reachable-cell repair adds
cell-envelope supervision equally to XYZ and bounded arms. The simple realization
head is refitted on TRAIN outcomes from each matching decoder. Center/projection
views receive new hashes and their own outcome pools rather than inherit mismatched
success labels. The complete original scorer and return rule stay frozen.

## 4. Experiments

Data comprise1,152TRAIN requests from128layout families and288reusedDEV_MODEL
requests from32families. All56,920TRAIN positives retain their original roles.
The two-channel reference representation uses deterministic within-mode clustering;
the means are independently checked for task/mode validity. Deployment uses
observation-predicted cells. No reserved TEST_LOCKED payload or metric is inspected.

The initial screen compares same-information XYZ, unbounded relative coordinates,
bounded mapping and peer-dependent boundaries. Two measured causes motivate repairs:
reference-centered widths can be optimistic around predicted centers, and uniform
endpoint cubes constrain geometry that the fixed-endpoint generator cannot reach.
The final three paired generator continuations0–2 use the same initialization and
actual observation/target streams within each seed,2,400updates, and matching
ordinary success heads. They share historical pretrained C0 and are not three
independent pretraining runs. All training, feedback, checking and failures are
charged to the new command ledger on the unchanged GPU1/.35memory/four-thread setup.

Primary quality is distinct actual valid words@8. A meaningful gain requires at
least0.15words over the strongest matched controls and a positive paired interval.
Guards permit at most1pointV8,.5pointV4 and.03wordU4degradation. We also report
known-mode recall, fixed2,169survival opportunities,270coordinate-repair opportunities,
new losses/recoveries, region feasibility, width collapse, latency and companion
perturbations. Repeated DEV selection and conditional bootstrap intervals do not
substitute for untouched evaluation.

## 5. Results and interpretation

All results below are actual saved-output checks, with three generator
continuations sharing the pretrained C0. The complete per-seed tables, intervals,
stressors and historical references appear in RESULTS.md. The existing DEV set
was repeatedly reused for research; the fresh set was collected after all model
and head weights were frozen and was used only for diagnosis.

| Population / method | Validity@8 | Valid modes@8 | Validity@4 | Valid modes@4 |
|---|---:|---:|---:|---:|
| Reused DEV, same-information XYZ | 85.89% | 6.7176 | 92.88% | 3.7130 |
| Reused DEV, bounded mapping | 87.01% | 6.8032 | 93.20% | 3.7280 |
| Reused DEV, refitted XYZ center | 90.93% | 7.1100 | 93.69% | 3.7477 |
| Fresh families, same-information XYZ | 79.22% | 6.2649 | 89.78% | 3.5883 |
| Fresh families, bounded mapping | 81.08% | 6.4028 | 90.33% | 3.6111 |
| Fresh families, refitted XYZ center | 86.69% | 6.8323 | 92.06% | 3.6796 |

On reused DEV, bounded minus XYZ is +0.08565 valid modes, with conditional
crossed 95% interval [+0.03356, +0.14236]. This is smaller than the fixed +0.15
practical gate. Bounded minus refitted XYZ centers is -0.30671, interval
[-0.42593, -0.20139]. On fresh families the corresponding differences are
+0.13790 [+0.07837, +0.19742] and -0.42956 [-0.59722, -0.27679]. A positive
small effect against XYZ does not establish superiority to the strongest control.
The fresh diagnosis also exposes a large masking sensitivity. Its sparse teacher
recall uses incomplete references and must not be compared directly with old
all-mode recall. Initial fresh recall used incompatible word definitions; a new
v2 analysis corrects reference words without changing predictions or primary
metrics. The original v1 is retained as superseded evidence.

The first predicted-cell screen has 100% bounded membership but only about
7.6% complete-cell geometric feasibility. Uniform envelope supervision raises
feasibility to about 71%, while endpoint failures increase from 137 to 172 and
returned validity decreases. Tapered reachable endpoint cells address the
identified overconstraint, yet do not eliminate the deficit to centers. On saved
seed-0 bounded predictions, certified-and-contained regions have zero geometric
implication violations, while 78 task endpoint failures remain within certified
cells. Physical containment does not ensure goal or passage semantics.

On the unchanged 2,169 survival and 270 repair opportunities, the three-seed
bounded means are 79.42% retention and 81.98% same-mode repair. Refitted XYZ
centers attain 80.94% and 89.01%. The historical parent retains 86.31% and repairs
74.07%, so increased repair comes with lost surviving alternatives. These
aggregate measurements prevent selecting favorable examples as evidence of
selective preservation. Actual paired XYZ/bounded gain, loss and remaining-failure
figures show the same source/destination observations and predicted tapered cells.
Gray obstacle overlays are evaluation-only truth.

A distinct saved-prediction diagnostic holds old relative parameters fixed and
maps them through destination cells. Its matched-slot denominators differ from
the fixed survival/repair sets. Across seeds, copied-coordinate success is
87.89–88.40%, transferred-parameter success is 90.27–91.25%, and new-center
success is 92.26–93.24%. Transfer repairs 75.28–77.82% of failed old coordinates,
versus 80.52–82.67% for centers. Parameter portability is measurable; learned
nonzero interior coordinates remain unjustified.

The actual online RGB-D/language-to-four-route pipeline averages 120.32 ms over
eight requests after one excluded warmup, with 138.41 ms median and 4.322 GB
joint peak allocated GPU memory. It includes the frozen VLM, point backprojection,
observation encoders, one eight-route decode, full scorer and selection, excluding
the independent checker and 2.55-second model load. This small timing sample
supports neither throughput nor robot execution claims. Cached API replay is
exact on 16 requests; 100-step versus 50+50 recovery is exact for model,
optimizer, RNG, actual sample stream and settings.

There are 151 terminal jobs: 148 completed and three preserved failures. The
measured command ledger totals 5,988.67 seconds including CPU rendering. It
contains 14 research generator trains and 17 matching head fits. Source exports,
actual commands, checkpoint states and predictions are archived; all 24,806
indexed local files pass hash verification. No reserved TEST_LOCKED payload or
metric was inspected. This accounting establishes reproducibility, not a
positive algorithm contribution.

## 6. Limitations and research decision

The task is controlled; its operational words are not generic homotopy classes.
Single-view learned cells do not certify occluded free space. Sparse depth probes
classify unknown regions but do not provide calibrated whole-cell confidence.
The representation already predicts a centerline, explaining why interior
neural generation may be unnecessary. Modes and goals can fail inside a safe
geometric region. The two trained geometry channels are supported, but the final
eight-word allocator chooses only variant zero. The frozen scorer limits returned
gains and is not recalibrated. Centers still compute the unused relative branch,
so timing is conservative and is not an optimized conventional-planner baseline.

Fresh-family simulation evidence supports the negative center-control comparison;
it does not erase repeated selection on the old DEV population. The three seeds
are generator continuations of one pretrained model, not independent pretraining
replications. Synthetic noise/masking are explicit observation stressors rather
than calibrated real sensor perturbations. No whole-arm, dynamics, real-robot or
untouched-test conclusion follows.

The publication-potential goal remains unmet. These results do not justify more
interior-decoder loss searches or a new corridor-planning principle. A subsequent
mechanism must address observed region/goal/mode errors and demonstrate an ability
that same-information centers cannot deliver. The present paper is a documented
research draft, not a submission-ready positive method claim.
