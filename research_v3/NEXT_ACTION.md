# Next action

All V3 coordinators and jobs are CLOSED; no current worker. Do not restart
paired data, q, calibration, prototype, isolation, or post-training diagnostics.
Latest source232ca6d posttrain_collision_v1 exit0;99 total closed jobs,
4436.170683/7200 server experiment command seconds,2763.829317 remain.
Read SCORE_CLOSURE_REPORT.md and posttrain_collision_v1/RESULTS.json first.

Ordinary matched new-domain q passes; meanBrier.076380 versus.148448. Separate
CAL-only affine.107789 and temperature.167084. Same generator; no new mechanism.
Prototype270/288 vs276/288 fails, no localization route intervention. Identity
audit passes, locked data stays closed. Every new result and receipt is locally
SHA verified. Current stronger reproducible pair is safety_mean plus preselected
newq seed0, with all3q seeds retained in reporting.

Residual collision evidence: TRAIN189/9216 (2.0508%) versus DEV207/2304
(8.9844%). TRAIN all1-3 violating segments, median maximum deficit2.720mm,
maximum15.258mm. All TRAIN and DEV satisfy floor. This favors both a small
near-boundary fitting residual and a larger generalization issue, not a proved
global gradient conflict or missing visual evidence. Previous maximum-penalty
and learned local/global refiner failures remain constraints.

Next decision: test whether the quadratic margin loss's vanishing boundary
gradient is worth an ordinary linear-hinge control. Before training, inspect
fixed TRAIN gradient scale/compatibility from the final mean checkpoint and
register one mean-vs-linear pair with identical additional updates, source,
input/target stream and new optimizer. Any coefficient must be fixed from
TRAIN gradient normalization, never DEV search. Do not relabel this standard
loss as a novel method, revive maximum-reduction tuning, or attribute extra
training gains to it. If evidence does not support that controlled comparison,
record why and pursue another mechanism rather than consuming the budget.
Only expand seeds after a substantive matched-control advantage; no final TEST
or paper novelty claim yet. Goal ACTIVE, no verified external blocker.
