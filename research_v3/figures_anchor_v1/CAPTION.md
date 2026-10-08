# Registered scratch diagnostics

Every branch starts from the same safety_mean tensors independently for each
fixed TRAIN batch. Left: relative change in the complete training objective
after one AdamW update, evaluated on the same batch/reference assignment.
Right: maximum absolute coordinate difference over all32x8x24x3 path elements,
not Euclidean displacement or collision distance. Hard and STE forwards are
exact before the update. Saved-optimizer initial gradients equal fresh exactly.
Four batches are diagnostic draws, not independent training-seed replications;
there are no confidence intervals or DEV generalization claims in this figure.

Reproduce: `python -m scripts.research_v3_plot_anchor_updates --source
research_v3/anchor_diagnostics_v1 --output research_v3/figures_anchor_v1`.
The PNG was rendered and visually checked. Source JSON hashes are in SOURCES.json.
