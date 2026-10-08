# Failure decomposition

Counts cover288 requests x3 trained models, not864 independent scenes. All
targets and edits belong to32 parent families. Failure categories overlap.

| Category | Actual count | Meaning |
|---|---:|---|
| Generator empty | 40/864 requests | No valid candidate before scoring |
| Partial known-mode coverage | 806/864 requests | Valid pool misses at least one witness; includes M8 capacity limit |
| Scorer miss with valid pool | 12/864 requests | Highest q invalid despite an available valid candidate |
| K4 loss beyond capacity | 149 mode slots | min(4, raw distinct) minus returned valid distinct |
| Invalid candidates | 1549/6912 | Full task and geometry event false |
| Post collision failures | 1252/6912 | Continuous inflated-box intersection |
| Semantic goal failures | 387/6912 | Wrong intended endpoint |
| Floor/event failures | 0/6912 | Under current reach contract |

47 probe samples (24 vertices +23 midpoints) miss182 continuous post collisions,
2.633% of candidates and 14.537% of all post collision failures. Existing checker
labels are correct; the issue is incomplete observation-feature sampling, not
182 mislabeled successes. Nearest visible surface distance is also not occupancy.

The independent checker intersects each complete closed segment with physical
AABBs inflated by2cm on every axis (a conservative L-infinity tip envelope).
World XYZ is meter-scale, RGB-D is calibrated camera-to-world. Floor is0.775m;
linear segments cannot dip below both endpoint heights. Goal tolerance3cm,
start tolerance5mm, reach events preserved. This covers registered posts and
the workspace lower bound, not full-arm collision, execution error, table mesh,
arbitrary occluded objects or unknown space. Those are outside label Y.

Method implications: ordinary mode-balanced coverage and injective matching
must be tested first. Low-cost exact single-q/dual model replay passes for
all paths, events, scores and chosen indices; historical dual-factor superiority
remains rejected. Failed audit attempts and their receipts remain available.
