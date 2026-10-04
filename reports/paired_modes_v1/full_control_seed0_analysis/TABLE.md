# Paired route-set comparison

All numbers use sealed, unrepaired M8 outputs. DEV is development evidence.

| split / arm / seed | valid % | modes | reference coverage % | goal fail % | collision % | frozen q top1 % |
|---|---:|---:|---:|---:|---:|---:|
|paired_dev/R0/seed0|69.05|4.958|60.68|17.97|17.58|84.72|
|paired_dev/R1/seed0|73.48|5.219|67.12|7.47|21.14|92.36|
|paired_dev/R_full/seed0|61.02|3.622|46.89|24.78|18.97|76.39|
|paired_dev/R2/seed0|74.44|5.264|67.23|6.47|20.79|93.40|
|old_dev/R0/seed0|47.57|3.222|23.68|32.64|29.51|66.67|
|old_dev/R1/seed0|65.97|4.667|31.56|10.07|28.47|86.11|
|old_dev/R_full/seed0|60.07|4.278|32.46|15.62|28.82|77.78|
|old_dev/R2/seed0|70.14|4.694|30.55|3.12|27.43|97.22|
|dev32/R0/seed0|49.48|3.396|21.71|26.17|33.46|72.92|
|dev32/R1/seed0|64.45|4.417|26.51|14.45|25.00|84.38|
|dev32/R_full/seed0|56.25|3.927|25.18|25.26|25.39|72.92|
|dev32/R2/seed0|61.85|4.302|31.06|6.25|34.24|93.75|

Seed0 continuation gate: `{'passed': True, 'conditions': {'modes_above_R0': True, 'modes_above_R1': True, 'shared_recall_above_R1': True, 'validity_loss_at_most_02': True}}`.

Family bootstrap results, all failed requests and input-stream identities are saved in RESULTS.json.
