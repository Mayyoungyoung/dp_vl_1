# TRAIN-only anchor mass diagnostic

Registered after anchor_support_audit_v1, before any mass computation.
Use only the sealed safety_mean checkpoint and all1152 TRAIN requests/128 families.
Compare its unchanged peak anchor, the ordinary soft spatial expectation, and
the observed point maximizing exact Gaussian-smoothed attention mass. Sigma
is fixed at .025m, the existing TRAIN grounding target kernel. Every valid
observed point is a candidate and contributes its attention weight; no oracle
target, subset selection, threshold search or DEV result determines the anchor.
The Gaussian is untruncated; chunk256 controls memory only. No model updates.

After all anchors for a request are fixed, read its target for anchor error and
necessary endpoint-cube representability. This is not a full route evaluation.
Register a later route intervention only if original impossible count>=4,
mass reduces it by>=25%, and mass within3cm count is no worse than peak.
Soft expectation is descriptive and cannot replace the registered mass gate.
If this fails, do not sweep scales or claim missing VLM information. Examine
remaining error before choosing another mechanism. At most600 command seconds
for this diagnostic; tests and failures count toward the shared7200 seconds.
