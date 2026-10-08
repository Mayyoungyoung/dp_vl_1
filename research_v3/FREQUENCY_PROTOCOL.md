# Frequency retention falsification pilot

Frozen before any new training result. The 160 existing paired RGB-D families
already provide verified complete-input multimodality. Only 128 TRAIN families
feed generator updates; 32 DEV_MODEL families are development evaluation.

Question: with the feasible witnessed set fixed, does frequency-biased generator
fine-tuning erase valid minority passages, and do ordinary methods prevent it?
This is a retention experiment starting from strong trained R1, not a claim
about training a VLM from scratch or estimating natural demonstration density.

For each validated witness, generate four unique smooth path perturbations with
fixed endpoints and events. Check every full H24 segment, workspace floor and
passage identity. Preserve rejection counts and generation seeds; do not replace
failed input requests. The effective independent population is 128 families,
not the number of perturbations. Geometry produces supervised labels only.

Eight reference draws per training input impose uniform, 90:10 aggregate
majority:minority, or 98:2 exposure. All minorities share the residual mass.
This is explicitly weighted resampling from identical unique teacher support.
Record actual per-input mode exposure and unique route usage. Baseline balancing
samples witnessed modes uniformly. Ordinary set matching covers min(M,modes)
distinct classes with Hungarian assignment, then fits spare candidates to any
known positive. All methods use the same architecture, initialization, input
draws, 1200 steps, batch32, grounding and continuous geometry penalties, M8.
The balancing and set controls need only one fit each: their exposure rule is
identical across the synthetic frequency regimes. They have no cross-scene loss.

Operational modes use first forward crossing per obstacle row, above the inflated
top first, otherwise a lateral free gap. Full legacy portal words are reported
separately. The definition separates central low and above routes, avoids an
arbitrary unknown height strip, and is not a strict homotopy claim. Witnesses
are not exhaustive; M8 cannot cover all nine witnessed modes in open scenes.

Evaluate raw M8, K4 from the fixed single-q baseline, rare witnessed recall,
validity, duplicates, closure-specific failures and original-score Brier/NLL.
Every new layout recomputes observed features and checker labels. No q refit or
calibration on paired DEV. All five arms are single-seed hypothesis screening;
replication requires a concrete extra mechanism and a prospectively recorded
comparison. The gate and budget are in configs/research_v3_frequency_v1.json.
