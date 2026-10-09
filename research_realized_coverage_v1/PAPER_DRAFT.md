# Realized Coverage under Interacting 3D Route Queries

Internal research manuscript, 9 October 2026. Empirical mechanism study with a
negative method result; not a submission-ready ICLR/ICML/CVPR/RSS/ICRA claim.

## Abstract

A finite route generator must turn semantic mode requests into distinct valid
paths. Requesting a missing mode can change companion trajectories and remove
already useful alternatives. We study this distinction using single-query
interventions on an RGB-D/language-conditioned generator that emits eight
24-point 3D end-effector routes and returns four through a frozen scorer.
We implement scalar net-utility learning, dense realized-mode prediction and
deployment-context geometry training, and compare them with an ordinary
realization-success predictor using the same feedback. Although 370 omitted
development modes are recoverable under sampled interventions, only 77 admit
a measured positive-net, quality-preserving replacement. Across five paired
allocation-head seeds, the dense interaction method produces 6.9465 valid modes
versus 6.9806 for the simple control; the difference's crossed seed/family 95%
interval is [−0.0743,−0.0014]. Geometry coordination does not overcome the
control's returned-quality advantage. Results support a useful ordinary
baseline and a reproducible failure diagnostic, not the proposed mechanism's
independent algorithmic contribution.

## 1. Introduction

Mode coverage, route validity and retaining feasible choices after scene edits
are different properties. Historical gate training raises validity but loses
surviving modes. A previously implemented shared mode-conditioned decoder C
improves representation relative to ordinary set controls, yet cross-scene
displacement/KL mechanisms do not supply a stable full-system gain. We therefore
ask whether allocation based on actual generated coverage, coupled to geometry
training on its failures, solves the remaining problem.

The key observation is that nominal query identity does not determine realized
mode identity or validity. Query attention also couples outputs. Recovering one
word is not necessarily a net improvement. Our candidate contribution is a
supervised coordination mechanism grounded in complete realized changes, rather
than new matching weights or more cross-scene positives. This contribution is
explicitly tested and is not established by the experiments.

## 2. Setting and method

Input x contains current observed RGB-D features, frozen visual-language
features and allowed robot state. Sixteen operational two-row passage words
describe controlled layouts; they are not universal homotopy classes. Queries
A={(m_i,v_i)} select semantic words and within-word variants. A shared conditional
decoder returns P_theta(x,A), eight complete 3D paths. Let C(P,x) be the set of
independently verified actual valid words. Define U8=|C|, V8 as valid fraction,
and U4,V4 after the complete frozen scorer's return rule.

A single-query intervention A' preserves all seven companion identities.
We regenerate and check all eight paths and supervise

    u(A) = (U8, 8 V8, U4, 4 V4)
    delta(A,A') = u(A') - u(A).

TRAIN feedback records inputs, queries, full outputs, official valid words,
invalid raw signatures, selected indices and source/checkpoint hashes.
Unknown reference availability is never treated as proven physical infeasibility.

The ordinary baseline learns s_m(x), the empirical success probability that
request m becomes a valid output of m over sampled query contexts, and requests
the eight highest-scoring distinct words. The scalar set model predicts u from
context plus query embeddings/pooling/histograms. It minimizes

    L_set = ||u_hat(A)-u(A)||²
          + ||u_hat(A')-u_hat(A)-delta(A,A')||².

The dense model adds query attention and predicts 17 outcomes per slot: sixteen
valid actual words plus invalid. Its approximate coverage is

    U_hat = sum_m [1 - product_i (1 - p_i,m)].

Cross-entropy on realized outcomes complements set regression. This product
approximation does not establish independent events, monotonicity, submodularity
or any greedy guarantee. A no-peer tokenwise model is a matched control.
At deployment, at most two token-only replacement sweeps require predicted
coverage gain and quality nondegradation before a single final eight-route
decode. The verifier and source-scene routes are absent at inference.

Geometry coordination mixes ordinary witnessed-mode training with contexts
proposed near deployment, injecting a witnessed failure mode with nonzero
sampling floor. Current-scene verified coordinates train the shared decoder;
there is no D2 displacement loss. Ordinary, KL-only and ordinary-companion
hard-example controls share update budgets. Feedback is recollected after
changing decoder weights, and both proposal types are refitted to each snapshot.

## 3. Experimental protocol

Use existing 1152 TRAIN / 288 DEV requests, 128/32 layout families, unchanged
observation encoders, all legal positives, original independent checker and full
frozen scorer. TEST_LOCKED is untouched. Main acceptance was registered before
new results: +0.15 U8 against the strongest same-information control, positive
family interval, and noninferiority margins of 1 point V8, 0.5 point V4 and
0.03 U4. All curves and failures are saved.

Formal seeds 0–4 train success/dense heads for 2400 final updates with identical
actual minibatch streams within seed on fixed C0. Geometry screens use one seed,
3600 steps and matched600-step snapshot refresh. Their branch-specific RNG
differs, so they are exploratory and are not labeled exact-stream causal tests.
Bootstrap intervals condition on the fixed decoder and reused development
families. No new-distribution generalization claim follows from them.

## 4. Results

|Five-head-seed mean|U8|V8|Known recall|U4|V4|
|---|---:|---:|---:|---:|---:|
|Ordinary success|6.9806|89.149%|82.319%|3.7736|94.340%|
|Dense interaction|6.9465|89.184%|81.712%|3.7701|94.271%|

Dense does not win U8 in any seed (one tie). Fixed2169-witness mean retention
is1776.8 versus1779.6, and same-mode repair is235.6/270 versus235.2/270.
Intervals for both differences cross zero. Both retain fewer modes than the
historical parent1872/2169. Gap geometry plus refreshed dense matches simple
seed0 U8=6.9826 but V4 falls from94.358% to93.663%, exceeding the allowed margin.

The simple baseline improves C0 U8 from6.7639 to6.9806 and repair from207/270 to
235.2/270. It does not dominate historical Gate's higher raw coverage/validity.
Added-only and no-peer ablations do not support net subtraction or peer attention
as effective independent mechanisms. All10168TRAIN within-mode reference means
are valid and preserve their word, rejecting the measured target-averaging
conflict explanation. Full per-seed, curve, ablation and failure categories are
in RESULTS.md and results/RESULTS.json.

Actual APIs match stored paths/events/scores exactly, call the decoder once,
and return the same four routes. Limited full-API latency is6.782ms simple versus
11.925ms dense, excluding external Qwen extraction. Qualitative figures include
a repair, a lost mode and a same-mode invalid output; they do not substitute
for the aggregate negative result.

## 5. Related work and contribution boundary

Context-conditioned diversity sampling for pretrained generators, admissibility
supervision and marginal-contribution learning have direct prior art
([Diverse Trajectory Forecasting](https://arxiv.org/abs/1907.04967),
[Admissibility-aware Forecasting](https://arxiv.org/abs/2302.03462),
[SetPO](https://arxiv.org/abs/2602.01062)).
These references establish adjacent mechanisms, not exact equivalence of tasks;
the narrower analysis is in RELATED_WORK.md. Mode embeddings, success
classification, set pooling and hard-example training are not individually new.
R2/R3 correspondence and D2 displacement are prior negative project evidence,
not repackaged contributions.

The defensible outputs are a realized-intervention protocol, a strong ordinary
baseline and reproducible counterexamples to naive coverage credit. The requested
novel coordination contribution is unsupported.

## 6. Limitations, positioning and next evidence

This is controlled end-effector route validity, not robot task success. Semantic
words are layout-specific; no independent new-scene distribution, arm collision,
IK or closed-loop rollout has been validated. Five seeds cover head optimization
only, not full-model/data uncertainty. Reused DEV supports research decisions,
not a final generalization claim.

The work currently fits an internal technical report; a workshop negative-result
study would still need broader reproducibility evidence. It is not ready as a
main-conference method paper. Reaching that standard requires a supported
generator update mechanism that beats frozen C+success in exact-stream, multiseed
comparisons, followed by frozen new-scene evaluation and full-arm execution.
These are unresolved core requirements, not merely optional extra experiments.
No claim that a few supplementary plots would make the present method publishable.
