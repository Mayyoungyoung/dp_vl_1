# Separating Route Modes from Geometry under Controlled Scene Changes

**Research draft, not a submission-ready paper.** The representation experiment
is positive against matched ordinary controls; the original selective geometry
pair objective does not establish an additional advantage. Conclusions below
include that negative finding rather than presenting a completed new algorithm.

## Abstract

An instruction-conditioned robot route generator must allocate a finite number
of candidates among distinct feasible passages and adapt their coordinates as
obstacles move. We study these requirements separately using a frozen visual-
language and observed-depth encoder, a semantic mode proposal head, and a shared
mode-conditioned3D route decoder. Selective supervision couples the coordinate
displacements of positively witnessed same-mode scene pairs, while leaving
unknown or unavailable relations unconstrained. In a controlled benchmark with
1152training requests,288reused development requests, eight candidates and a
fixed four-route return interface, semantic conditioning improves valid-mode
coverage by0.480 over a matched ordinary sampled-set baseline across three
continuation seeds. However, adding the displacement objective changes coverage
by−0.059 versus conditioning alone, and does not improve fixed edit retention
or returned quality. These results support separating representation from
matching, but do not establish the proposed complete method as a publishable
advance. A proposal-allocation connection repair is reported separately.

## Problem and observation

For current observation x=(RGB-D, language, robot state), produce eight end-effector
routes P={p_k}, p_k∈R^(24×3), and return four using a frozen observation-based q.
The evaluator checks continuous tip-envelope clearance, start state, target
semantics, workspace floor and events. It does not certify robot execution.

A useful passage can survive a scene edit even when one particular old path
collides. We distinguish (i) survival of an operational route word, (ii) unchanged
coordinates, and (iii) generation of a new valid realization of that word.
Historical Gate/Set-point improve raw candidate coverage but retain only70.20%/
71.62% of2169fixed parent witnesses, versus86.31% for the original parent.
This is a controlled problem observation; neither witness sets nor known modes
are an exhaustive enumeration of feasible robot motion.

## Method

Frozen encoders compute context h(x) and observed target anchor a(x). A lightweight
proposal head predicts l_m(x). The shared decoder receives

`z_(m,v)=h(x)+E_m+V_v`,  `p_(m,v)=f_theta(z_(m,v),h(x),a(x),s)`.

E encodes16two-row passage words; V distinguishes repeated realizations of a
word. Mode identities are semantic labels rather than fixed candidate positions.
Intermediate points are learned residuals around the current-to-anchor line;
the original endpoint residual constraint and event head are retained.

`L=L_route+0.001 L_mode+lambda_pair L_pair`.

L_route supervises full coordinates/events with the inherited clearance term.
L_mode learns proposal mass from witnessed positives and masks unknown binary
status. Only certified closed internal passages supply negatives. For witnessed
common words W_ab:

`L_pair = E_(a,b,m∈W_ab) ||(p_b,m-p_a,m)-(y_b,m-y_a,m)||²`.

The pair target generally has nonzero displacement. Deleted/unknown words have
no edge. The objective sends gradients to generated XYZ, rather than merely
matching fixed labels. Algebraically it couples regression errors across scenes,
so its independent value must be established against same-information controls.

Current-scene inference allocates at most eight positively scored words and
uses repeated mode variants to fill eight candidates when necessary. A fixed
scorer returns four. It sees no old path set, parent retention labels, truth
geometry or edit correspondence. The adaptive threshold is a proposal heuristic,
not calibrated evidence that an unselected mode is impossible.

## Experiments and falsifiable conclusions

All paired arms share the parent, frozen encoders, data,1200updates and actual
sampled input/target streams. B0 uses ordinary assignment to sampled mode targets;
B_set minimizes over all available same-mode coordinates. C adds semantic
conditional decoding; D adds witnessed path displacement. An additional ordinary
pair-weighted B control and two targeted repairs are kept as bounded diagnostics.

Three-seed means: B0/C/D valid@8=78.95/88.89/87.85%, distinct valid modes=
6.289/6.770/6.711, fixed retention=79.64/80.56/79.64%, same-mode repair=
66.42/76.79/77.04%, and returned validity=93.34/94.24/93.26%.
C−B0 diversity is0.480, family-bootstrap95% interval[0.281,0.683]. D−C is
−0.059[−0.172,0.042]. D−C repair is+0.25pp[−2.89,3.85]. Report every seed,
not the best; uncertainty is conditional on these continuation runs and reused
DEV families. See RESULTS for all controls and strict repair subsets.

The final D2 connection repair reaches87.76% valid@8,6.725 valid modes,
81.05% retention,78.89% same-mode repair and93.52% returned validity.
Versus C, retention changes+0.49pp[−0.85,1.66] and repair+2.10pp[−1.54,5.72],
while returned validity decreases0.72pp. The repair helps D's retention but
does not establish the full-method claim. Ordinary sampling with the same
weights markedly reduces coverage: gains belong to the conditional-generation
and mode-allocation system, not demonstrably to decoder structure alone.

Thus H1 receives support within this fitting protocol. Original H2 lacks support,
and H3 is not established: conditional structure helps, but the full combination
does not improve over C in retention/returned utility. C also fails to dominate
the historical parent and Gate across the required quality dimensions.

## Why it sometimes works, and why that is insufficient

Changing the mode query changes actual generated paths; permutation of queries
permutes their routes. On TRAIN, swapping edited context for source context
reduces same-mode valid output, demonstrating that the decoder uses scene input.
These are mechanism checks, not generalization proof. Replacing other mode
queries also perturbs a fixed route through self-attention; eliminating that
interference improves forced-mode TRAIN compliance but worsens aggregate DEV
quality. Most fixed retained-mode losses instead arise before decoding, when the
proposal head does not request the mode. Coordinate training alone cannot repair
a missing proposal. D2 tests that specific disconnection without claiming its
ordinary probability-consistency term as an innovation.

## Related work and contribution boundary

Intention/mode-conditioned multimodal trajectory generation is established.
[MTR, NeurIPS2022](https://papers.nips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)
already separates global intention localization and local movement refinement
for motion forecasting. Our explicit passage words, observed robot-task inputs,
fixed edit witnesses and verified displacement targets differ in setting, but
do not by themselves establish a novel general generative principle. No MTR
benchmark comparison or claimed superiority is made.

Historical project R2/R3 align geometry-relative descriptors of ordinary route
sets. This study instead conditions generation on semantic words and supervises
actual nonzero coordinates. Set-point and verified positive augmentation change
targets/assignment; they are not this decoder. However, pair correspondence,
mode embeddings and target-set learning cannot each be claimed as new in isolation.
See the existing VERIFIED_SET_RELATED_WORK_SCOPE and METHOD for exact boundaries.

The defensible present contributions are: a controlled quantification of distinct
failure modes, a reproducible conditional-generator implementation with matched
controls, and a negative/limited-positive mechanism study. The hoped-for stable
geometry-adaptation algorithm contribution remains unestablished.

## Submission positioning and remaining work

This is suitable as an internal research report or a transparent preliminary
workshop study; it is not ready for ICLR/ICML/CVPR/RSS/ICRA main-track claims.
Finishing a top-conference paper requires more than cosmetic experiments:
resolve proposal/geometry coupling without losing return quality, verify against
strong conventional mode-conditioned generation controls, then freeze the method
and evaluate genuinely new scene families/task types and geometry interventions.
Only after that should reserved final evaluation and full-arm execution be used.
Broader visual/linguistic changes, multiple independent parent initializations,
strong external implementations and end-to-end latency are still missing.
No claim is made that merely adding robot execution or a new test set would
turn the current negative pair result into a successful method.
