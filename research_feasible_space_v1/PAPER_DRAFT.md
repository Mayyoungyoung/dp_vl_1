# What Does Feasible-Space Parameterization Buy in Observation-Conditioned Multi-Route Generation?

Research draft. This is not a claim of an accepted novel algorithm. The quantitative
main comparison is populated from recorded experiments in RESULTS.md and
results/three_seed_statistics_v1.json. The bounded decoder fails the locked gate.

## Abstract

Generating a finite set of useful end-effector routes from a single RGB-D view and
language requires both geometric feasibility and coverage of different passage
patterns. We study whether an explicit output parameterization can improve these
properties while limiting interference between candidate queries. A mode-conditioned
network predicts an ordered chain of bounded convex cells independently of other
queries. A second decoder coordinates the eight candidates through attention and
maps its parameters into shared endpoint cubes. Their connecting segments remain
inside the corresponding cells. This property is algebraic; an incorrectly
predicted cell does not certify scene safety. We construct TRAIN-only corridor
labels, compare matched free-coordinate and bounded decoders, and include relative,
projection and centerline controls with the same observed information. All methods
retain the complete frozen four-route return interface. Reference labels represent
all56,920valid paths individually and89.97%within two mode-specific prototype
corridors. Centerline, free and bounded oracle-corridor diagnostics all attain
100%task validity on1,024TRAIN paths, showing that containment alone does not
establish a need for learned interior generation. On288reused development requests,
three paired generator continuations yield6.803valid modes@8for bounded mapping,
versus6.718for same-information XYZ and7.110for independently refitted centerline
controls. The0.086gain over XYZ is below the registered0.15meaningful-gain gate,
and the deficit to centerlines excludes zero in conditional bootstrap intervals.
This study therefore separates a correct conditional containment property from
a useful neural interior-generation mechanism. Frozen new-family diagnosis is
reported separately; no full-arm or real-robot capability is claimed.

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

See the recorded tables and every seed in RESULTS.md. StageAalready shows that
reference-cell expressivity is adequate while neural oracle necessity is absent.
The first predicted-cell screen also separates containment from physical safety:
bounded membership is100%but complete uniform-cell feasibility is about7.6%.
Uniform-cell supervision raises region feasibility to about71%, yet introduces an
endpoint tradeoff. The tapered repair explicitly targets reachable endpoint regions.
The final bounded mean is87.01%route validity@8,6.8032actual valid modes@8,
93.20%validity@4and3.7280returned valid modes@4. Same-information XYZ gives6.7176
modes@8; the paired difference is0.08565,95%crossed interval[0.03356,0.14236].
This smaller effect fails the0.15gate. Fresh-head center controls give7.1100,
with bounded-minus-center difference-0.30671,interval[-0.42593,-0.20139].
The bounded decoder also trails historical C+success and Gate references.
On2,169fixed surviving-mode opportunities,boundedseed0retains1,727versus1,872
for the parent; its224/270same-mode repairs are below center-only243/270.
No positive algorithm contribution follows from these results. The paired plots
show both improvement and loss cases and do not replace the aggregate comparisons.

## 6. Limitations

The task is controlled and its words are not generic topological classes. Learned
cells do not certify free space behind a single-view occlusion. A ray utility
distinguishes probes before a visible surface, near that surface and unknown
behind/outside the observation, but neither sparse probes nor a realization head
provide a whole-cell uncertainty certificate. The representation already predicts
a centerline, creating a strong explanation for why interior neural generation may
be unnecessary. Modes can change inside a geometrically safe region. The fixed
scorer may limit returned gains and is not recalibrated. No independent new-family,
full-arm, dynamics or real-robot conclusion follows from these experiments.
The fresh-family extension uses frozen weights and an explicitly registered
evaluation-only protocol; it cannot erase repeated selection on the old DEV set.

The study should be considered a research draft. Submission-readiness depends on
the measured comparison, a defensible distinction from corridor-learning/planning
prior art, and evidence beyond the reused development population.
