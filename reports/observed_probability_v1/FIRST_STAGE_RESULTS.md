# Fixed-generator route reliability: measured first-stage results

All three scorer runs completed1200 AdamW updates. The generator is the unchanged ordinary last12000 model, M4/H24, trained on95 parents. The q configuration was fixed prospectively; each seed selects its checkpoint by BCE on32 DEV_SCORE parents. SCORE_TRAIN64, DEV_SCORE32 and CALIBRATION32 are disjoint extension TRAIN parents unseen by this generator. All three target instructions stay with each parent.

| Scorer seed | SelectedValid, DEV_SCORE96 | SelectedValid, old DEV_MODEL36 | Old DEV Brier before calibration | After calibration | Temperature |
|---|---:|---:|---:|---:|---:|
| 0 | 62.50% | 69.44% | 0.1266 | 0.1277 | 1.5175 |
| 1 | 63.54% | 69.44% | 0.1315 | 0.1311 | 1.4561 |
| 2 | 63.54% | 63.89% | 0.1514 | 0.1514 | 1.1018 |

On the fixed384-candidate DEV_SCORE pool, first-candidate validity is27.08%, expected uniform-random selection33.85%, shortest40.63%, and AnyValid72.92%. Scorer seed0 improves to62.50%, seeds1/2 to63.54%. These are development/selection data, not an independent final-test estimate. Parent bootstrap intervals are preserved in each JSON and do not correct selection optimism.

On the fixed144-candidate old DEV_MODEL pool, first-candidate validity is25%, expected random38.89%, shortest36.11%, and AnyValid75%. Scorers select25/36,25/36 and23/36 valid routes. These12 old parents were not used to fit/select q, but were historically reused for generator development. Three q training seeds share one frozen generator and candidate pool; they are not three independently trained VLMs or generators.

Calibration uses only32 separately reserved parents after q checkpoints are fixed. A scalar positive temperature leaves all rankings unchanged. Brier on old DEV improves slightly for seed1 and worsens slightly for seeds0/2; selected-candidate Brier is also not consistently improved. Thus reliability ranking has useful evidence, while universally calibrated probabilities are NOT established. All conclusions concern the original tip-only checker (3cm target,5mm start, unchanged reach events,2cm physical-box margin), not whole-arm robot success.

The failed Python3.8 exporter and failed first M8 evaluation attempt remain under server jobs with their costs. The latter completed500 updates but failed before its first saved training checkpoint; it cannot be resumed and is not counted as a completed comparison. The repaired generator driver saves recovery state before evaluation and uses the composite-data verifier. Full M8 and scaling results will be reported separately.
