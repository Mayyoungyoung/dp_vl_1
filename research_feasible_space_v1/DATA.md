# Data and information boundary

Reuse legal paired1152TRAIN requests (128layout families) and288DEV_MODEL
(32historically reused families), frozen matching RGB-D/Qwen features and56920
TRAIN positive paths. No new raw collector loading. Explicit role filters are
applied before labels, observations or routes are read. TEST_LOCKED remains unread.
Generator, scorer and calibration roles retain their historical assignments.

Each TRAIN path has23cell radii from exact signed L-infinity segment clearance to
the2cm-expanded post boxes and workspace floor. Radius=min(.06,.8minimum slack).
Zero width is allowed; a zero radius never certifies a colliding center segment.
The floor constraint and task/event checks are independent of region containment.

Within every request/operational word, deterministic two-cluster means define
two geometric corridors. If a mean failed the task/mode checker it would fall
back to a within-cluster medoid; none required this fallback on existingTRAIN.
Each witness keeps its assigned variant. This supports more than one geometry
per mode without introducing slot-specific meanings.

Reference labels use oracle TRAIN boxes; deployment uses only frozen observation
context, observed current pose, learned surface anchor and requested word/variant.
The TRAIN cache contains oracle columns solely for the loss. The forward API
accepts no obstacle boxes, floors, references or truth widths. No inference repair
with the checker is performed. A changed decoder gets fresh TRAIN outcomes.

StageAactual: all56920reference paths admit segment certificates. Two prototype
corridors contain51211paths(89.970%), allowing every24node and every connecting
segment. All20336prototype centerlines are task-valid and correct-word. All-witness
representability is conditioned on individually constructed centers; it must not
be confused with coverage by the two fixed prototype corridors.

Single-view unknown regions are not marked free. Initial model uses learned
geometric priors rather than a ray-certified occupancy map; predicted corridor
containment is an algebraic property, never physical safety or visibility proof.
Observation noise/occlusion evaluation must expose this limitation before any
independent/general robot claim. No new held-out family has been generated yet.

The visibility audit reads only depth/camera_intrinsics/camera_extrinsics from the
permitted current observation file. It classifies shared-node cube-corner probes
as free_at_probe/observed_surface/unknown, preserving camera focal signs. Occluded,
out-of-frame and missing-depth probes are always unknown. Probe classification
is distinct from whole-cell truth certification; neither is a deployable oracle.
The audit changes no trained route or proposal, and makes no learned-confidence
claim. All methods can use the same legal calibration/depth evidence.
