# Shared-endpoint failure diagnosis

The mean continuation has12/288 empty DEV requests; all12 have incorrect
semantic endpoints for every candidate. The worst-segment arm reduces post
collision failures207 to171 but increases semantic failures108 to149. These
overlapping counts do not support claiming a net safety improvement.

Next read-only diagnostic, fixed before inspection: first16 sorted TRAIN
families (all144 requests), all32 DEV_MODEL families (288 requests), initial
R1 seed0 plus completed mean/worst models. Forward uses only genuine cached
Qwen and observed RGB-D/current state/camera geometry. After predictions,
measure nearest visible point to intended target, positive-region attention
mass/rank, actual learned anchor error, and the necessary representability
bound of its fixed endpoint residual cube. Keep all requests and failures.

This distinguishes lack of visible target support, semantic localization
failure and an endpoint representation limit. It does not justify changing
the target tolerance, inspecting locked data, enlarging the VLM or adding a
new head without further evidence. No optimizer update or new calibration.
