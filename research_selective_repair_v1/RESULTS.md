# Development results; research remains active

The learned selective mechanism is not supported by the initial seed0 screen.
The publication gate is +0.15 U8 relative to the strongest matched control,
with family-paired positive uncertainty and the original quality protections.
Simple residual improvement over centers does not establish this mechanism.
Three paired seeds, new physical development layouts, complete matching heads,
independent frozen confirmation and expanded execution remain pending.

Initial reused DEV:288requests/32families, shared frozen C0/center,2400updates.
All methods retain eight final candidates and the complete frozen scorer/return4.
The screen uses the original matching center allocation head for every arm.

| Method/DEV setting | U8 | V8 | U4 | V4 | Repaired | Damaged |
|---|---:|---:|---:|---:|---:|---:|
| Center, zero edits | 7.14931 | .91450 | 3.75347 | .93837 | 0 | 0 |
| Selective, all registered thresholds | 7.14931 | .91450 | 3.75347 | .93837 | 0 | 0 |
| Decoupled selective, selected identity | 7.14931 | .91450 | 3.75347 | .93837 | 0 | 0 |
| Whole residual, scale1 | 7.28819 | .93186 | 3.82292 | .95573 | 40 | 0 |
| Joint center/residual, scale.75 | 7.25347 | .92795 | 3.79861 | .94965 | 33 | 2 |
| Observed surface optimization | 7.22569 | .92448 | 3.76042 | .94010 | 23 | 0 |
| Observed goal-tail rule,2.5cm trigger | 7.12500 | .91059 | 3.73264 | .93316 | 91 | 100 |

Common-center paired repair/damage counts supersede the earlier joint-center
screen counts computed relative to its own retrained drafts. That screen's
overall U8 was correct; its repair110/damage5 is inappropriate for a common-B0
causal comparison. Newly cached moving-draft features also correct a mismatch
in that control's initial training inputs.

Residual minus center:mean+.13889 U8,family bootstrap95%[+.05903,+.23264],
32families/10000resamples/seed206010. This describes one repair seed on reused
DEV, not variability across pretraining or independent confirmation.

Decoupled selective threshold.5 repairs6/damages30; all nonidentity registered
thresholds exceed1%damage and lose net modes. Original selective is identity at
every registered threshold. Positive/negative class balance and direction/gate
decoupling therefore did not resolve the measured bottleneck on these data.

Paired goal-proposal diagnosis attributes all100goal-rule damages to18wrong
proposals. The270accurate proposals repair91routes withoutdamage. TRAIN has
57/1152wrong proposals. Typical observed-surface error is2.55cm; the bad repeated
component is roughly80–89cm from the actual target. A shared TRAIN workspace
envelope and surface residual calibration is now a strong control for all arms.

Actual fixed Panda pilot uses first two DEV families/open+closed/target0,
all four actually returned routes, equal initial joints/quaternion/RNG/budgets,
collision-aware get_path, zero retries. Centers succeed7/16; goal rule8/16.
Family128 open changes0/4->1/4; other three requests remain4/4,1/4,2/4.
This small pilot establishes no execution advantage. Failures include planning
and explicit arm_environment collisions on tip-valid predictions. The actual
arm/gripper collision monitor is enabled; held-object execution is untested.

The first executor import failure and the NumPy locale failure precede any
planning call. Their receipts remain failed. v3 actually executes and closes
all16routes per method after fixing lightweight IO/native checks/child locale.

As of the initial-result snapshot,terminal wrapper command time is2274.1603s,
25completed/3failed. The ongoing144-scene render is excluded until terminal.
No total wall quota is supplied; command time is not a fabricated resource cap.
TEST_LOCKED remains unread. Timing of the initial route heads excludes VLM
inference and observation encoder work; it cannot establish online speedups.

Compact evidence:[INITIAL.json](results/INITIAL.json). Initial figures under
figures/initial are labelled developmental; they will be updated after the
registered new-data experiments. Check RESEARCH_STATE.md for current queues.
# Latest development screen,2026-10-10

No accepted mechanism yet. New physical population has16TRAIN/8DEVfamilies,
6variants,144renderedscenes,432requests (288TRAIN/144DEV),all144initializations
passed. It is development evidence; final confirmation has not been generated.
Legal common goal grounding uses a TRAIN-fitted workspace/surface offset.

|Method|Old DEV U8|New DEV U8|
|---|---:|---:|
|C0 zero correction|7.14931|7.24306|
|Shared observed goal + geometry64steps|7.66319|7.74306|
|Selective seeds0/1/2|7.14931/7.16667/7.14931|7.24306/7.24306/7.24306|
|Whole residual seeds0/1/2|7.35069/7.32986/7.32639|7.43750/7.41667/7.52083|
|Stabilized joint-center seed0|7.34375|7.43750|
|Adapted local diffusion seed0|7.15625|7.25694|
|Optimistic fixed18node prefix ceiling|7.71528|7.72917|

All neural seeds share C0 and have2400continuation updates,batch32,headLR.0003.
Original joint-center runs were unstable, losing4–18%valid routes; they are
preserved but are not competitive controls. Stable control detaches gradients
through recomputed observed features and uses centerLR.00003; two damages
on each population. Strong geometry controls repair150/75routes and damage0/1.
Their V8=.97960/.98524; V4=.99913/.99653. Diffusion gains are negligible.

The fixed-prefix ceiling is only for the actual screening queries. It allows
unrestricted future tails and16possible words for unresolved prefixes; failed
prefix geometry and fixed words cannot be repaired by an immutable tail.
The new ceiling is lower than geometry optimization, which can edit earlier.
This necessitates the prospective learned-constraint full-local operator,
not another gate-weight search. All scores use the frozen C0 allocation head;
per-decoder TRAIN head fitting is still required for formal acceptance.
Compact exact results and current command budget:results/CALIBRATED_SCREEN.json.

## Earlier initial screen (retained historical evidence)
