# Match the ordinary single-q baseline to the improved generator

The fixed original score transfers with Brier.14845 on safety_mean and.20030
under the rejected anchor intervention. The latter failed its route gate and
is not used here. Complete the fair baseline for unchanged safety_mean first.
This tests score distribution mismatch, not a new generation mechanism.

Freeze safety_mean seed0 peak-anchor model; use existing reserved SCORE_TRAIN,
DEV_SCORE and CALIBRATION parents only in their original roles. Forward all
M8 actual unfiltered outputs, then apply unchanged continuous validity labels.
Extract scorer features with the COMPLETE original R1seed0 encoder, independent
of proposal generator. Exact paired DEV pools must match stored paths/events/
labels/q before training. Check role parent disjointness before optimization.

Reuse ordinary single-q width64,BCE,1200steps,batch32,lr.0003,weight_decay.0001;
seeds0,1,2 all run. Choose earliest minimum DEV_SCORE BCE every100steps.
Fit one scalar temperature per scorer on CALIBRATION only, float64 objective.
Paired DEV_MODEL is evaluated once after fitting; no temperature or threshold
selection there. Report raw and calibrated Brier,NLL,ECE,top1,K4 reliability,
mode recall and distinct modes using fixed q-first selector on identical pools.
Return all per-seed values and family-paired intervals; no best-seed picking.
Seed0 is the predefined deployment example; test actual new public process.

Ordinary matched-score gate: mean pairedDEV Brier improves>=.01 vs original q,
all three Brier differences negative, mean selected distinct modes decrease
no more than.05 and mean selected all-valid fraction decreases no more than.01.
Regardless of gate this is a baseline/control, never a novelty claim. No further
scorer architecture, factorization, threshold or negative-sampling sweep.
Finite queue shares existing7200 command-second cap; no role expansion.
