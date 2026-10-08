# Fixed-q evaluator correction — 2026-10-09

Before reading the completed five-arm comparison, code inspection found that
the first evaluator replaces the entire generator inside ScoredRoutePlanner.
That generator also supplies point features, observation context and anchors
to the scorer. Freezing scorer weights and temperature alone does not freeze
q(x,path). This is an evaluation isolation defect, not a training failure.

Preserve original evaluation and frequency_analysis_v1 as coupled feature
transfer evidence. The corrected evaluation_fixed_q_v2 generates identical
paths/events with a separate model, but uses the entire untouched historical
R1 deployment to score those paths. Assert exact q against original deployment,
and exact unchanged q on identical input/path after separate generator loading.
Run all five arms and retain all results; no checkpoint or hyperparameter changes.
Compare candidate arrays between evaluations, then use corrected scores for
K4, Brier/NLL/ECE and threshold results. Raw M8 validity/coverage remain unchanged.

This correction does not retrain a scorer or fit calibration on DEV_MODEL.
Scientific frequency gate is unchanged. Any future trained scorer will need
the original SCORE_TRAIN/DEV_SCORE/CALIBRATION separation and its own comparison.
