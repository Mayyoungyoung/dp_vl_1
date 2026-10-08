# Raw generator upper bound and selection audit

The same three historical R1 models generate M8/H24 without oracle repair or
filtering. All6912 candidate validity labels were independently recomputed with
the continuous checker; disagreements0. Returned K4 uses each generator's
existing independent single-q scorer and calibration, unchanged. This is a
selection analysis, not a new generator training result.

| Seed | OracleValid@8 | ValidDistinct@8 | KnownModeRecall@8 | ValidDistinct@4 | KnownModeRecall@4 | Top1 valid |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 94.792% | 5.2257 | 58.777% | 3.5278 | 39.925% | 92.014% |
| 1 | 94.792% | 5.1319 | 56.787% | 3.4965 | 38.528% | 93.750% |
| 2 | 96.528% | 6.4167 | 71.280% | 3.7326 | 42.174% | 96.181% |

Three-seed means: OracleValid@8 **95.370%**, ValidDistinct@8 **5.5914**,
known witness recall@8 **62.281%**; K4 distinct **3.5856**, witness recall **40.209%**.

Candidate validity 77.590%; selected candidate validity 91.377%. All original
portal counts are present in metrics.json and must not be confused with the
new above-first passage definition. Neither recall denominator is exhaustive.
RareModeRecall is undefined for this uniformly counted base teacher corpus.

Main bottleneck: missed witnessed passages and continuous collisions, rather
than Top1 scoring alone. Most K4 coverage reduction is its four-slot capacity;
extra loss below min(4, raw distinct) totals149 mode slots across864 requests.
Next experiment falsifies frequency-retention motivation against ordinary
balancing and distinct matching before inventing a correspondence mechanism.
