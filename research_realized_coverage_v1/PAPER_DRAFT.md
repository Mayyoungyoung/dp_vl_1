# Realized Coverage under Interacting Route Queries

Research draft in progress; no accepted novel method or submission-ready claim.

## Abstract

Finite multimodal route generation requires allocating queries that actually yield
distinct valid routes. With query interaction, recovering one requested mode can
destroy others. We measure single-query interventions on a frozen language/RGB-D
conditioned3D route generator and distinguish requested identity from realized
valid-mode coverage. A lightweight success predictor improves coverage; learned
set-net utility, dense realized-outcome factorization and geometry-gap training
are tested against that strong control. Current development results show that
recoverability is substantially larger than net beneficial recoverability, and
the proposed coordination has not yet established additional returned quality.
Formal five-seed results will determine the stability of these findings.

## Introduction and problem

Given current observed scene/language/state, generate exactly eight24-point
end-effector routes and return four with a frozen scorer. Mode labels describe
sixteen controlled two-row passage words. A route can satisfy its requested word,
produce a different word, be invalid, or duplicate an existing valid output.
Independent mode probability does not represent those outcomes. Previous
cross-scene displacement/KL mechanisms did not improve the complete system.

## Method and formulas

Let A={(m_i,v_i)} contain eight semantic/variant queries. The shared current-scene
decoder returns P_theta(x,A). Let C(P,x) be the set of independently valid actual
words. U8=|C|; V8 is valid fraction; U4,V4 use the unchanged frozen return rule.
For A' changing one query, supervise

`delta = (U8',8V8',U4',4V4') - (U8,8V8,U4,4V4)`.

An observation/query-only utility model predicts net changes, chooses at most two
token replacements, then decodes final routes once. Scalar regression is the
initial model. A dense alternative predicts17outcomes per query (sixteen actual
valid words plus invalid). Approximate expected coverage is

`U_hat = sum_m [1 - product_i (1-p_i,m)]`.

This factorization is a predictive approximation; interacting outputs do not
justify independent-event, monotonicity, submodularity or greedy guarantees.
Geometry updates retain witnessed positives and mix conventional coverage with
near-deployment contexts and a controlled failure-mode draw. There is no
displacement loss or gradient through the true discrete verifier.

## Experiments and present findings

Use1152TRAIN,288reusedDEV, unchanged encoders/q, full historical positives and
matched feedback/update budgets. Report actual coverage, validity, returned
quality, fixed2169mode survival and270/255coordinate-repair opportunities.
Local counterfactual oracle improvements are diagnostic, never method results.

On seed0, shared-feedback success sorting gives U8=6.983,V4=94.36%, while dense
set refinement gives6.885/94.36%. Gap geometry plus refreshed dense refinement
gives6.983/93.66%; the stronger simple system remains preferable. These are
development screens, not the pending paired five-seed conclusions.

## Contribution boundary and limitations

See RELATED_WORK.md: context-conditioned diversity sampling, admissibility-aware
generation and marginal-contribution training have direct prior art. Neither
mode embeddings, success classification, set pooling nor hard-example training
is independently new. The candidate contribution requires evidence that actual
query interactions demand coordination which the strongest simple controls cannot
replace. Present results have not supplied that evidence.

Current defensible outputs are a reproducible intervention protocol, a useful
ordinary realization baseline and measured counterexamples to naive coverage
credit. No whole-arm execution or independent new-scene validation has occurred;
neither should be implied by endpoint checks. New distributions should follow a
frozen supported mechanism, not mask a failed reused-DEV comparison. This draft
is presently an internal mechanism study, not an ICLR/ICML/CVPR/RSS/ICRA paper.
