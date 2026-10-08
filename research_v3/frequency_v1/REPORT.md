# Controlled frequency retention: single-seed pilot

All five arms actually completed; fixed final1200 checkpoints.

| Arm | Valid@8 | Distinct@8 | Known recall@8 | Rare recall@8 | Distinct@4 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| empirical_uniform | 73.611% | 5.7431 | 48.600% | 44.641% | 3.4965 | 0.1972 |
| empirical_90 | 83.594% | 1.5417 | 17.328% | 6.904% | 1.2882 | 0.1549 |
| empirical_98 | 84.592% | 0.8785 | 10.976% | 0.000% | 0.8785 | 0.1646 |
| balanced | 73.611% | 5.7431 | 48.600% | 44.641% | 3.4965 | 0.1972 |
| set_matching | 82.205% | 6.5660 | 66.909% | 64.358% | 3.7257 | 0.1523 |

Frequency gate: {"empirical_90": true, "empirical_98": true}. See RESULTS.json for all contrasts and family intervals.

The majority is a registered left/left passage and minority mass is shared
among all other witnessed passages. Rarity means controlled training frequency,
not path invalidity. These are biased fine-tuning results from a previously
trained R1, not experiments with fresh VLM training. All inputs and geometry
penalties are identical. Set matching processes more reference labels per input
than eight-draw arms; report actual reference counts and wall time in SUMMARY.csv
before interpreting data/compute fairness. The strong ordinary baseline must
remain the comparison target for a new mechanism.

The q head and calibration are frozen from historical R1 seed0. Its probability
metrics measure transfer onto each generator's own pool; they do not establish
an improved scorer on a fixed public pool. No independent final evaluation,
multi-seed new-method result, intermediate task constraint or novelty claim
follows from this pilot. All failures and full candidate sets are retained.
