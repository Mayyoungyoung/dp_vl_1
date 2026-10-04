# Paired route-set comparison

All numbers use sealed, unrepaired M8 outputs. DEV is development evidence.

| split / arm / seed | valid % | modes | reference coverage % | goal fail % | collision % | frozen q top1 % |
|---|---:|---:|---:|---:|---:|---:|
|paired_dev/R0/seed0|69.05|4.958|60.68|17.97|17.58|84.72|
|paired_dev/R1/seed0|73.48|5.219|67.12|7.47|21.14|92.36|
|paired_dev/R_full/seed0|61.02|3.622|46.89|24.78|18.97|76.39|
|paired_dev/R2/seed0|74.44|5.264|67.23|6.47|20.79|93.40|
|paired_dev/R0/seed1|79.64|5.733|66.84|6.25|15.36|93.06|
|paired_dev/R1/seed1|75.95|4.448|57.53|5.73|19.27|93.75|
|paired_dev/R_full/seed1|71.96|5.087|66.44|13.19|16.93|86.11|
|paired_dev/R2/seed1|74.70|5.514|64.32|5.69|20.75|93.75|
|paired_dev/R0/seed2|77.34|5.476|64.61|9.81|14.45|90.97|
|paired_dev/R1/seed2|83.33|5.788|75.18|3.60|13.93|96.53|
|paired_dev/R_full/seed2|74.70|5.184|67.37|10.07|16.67|88.54|
|paired_dev/R2/seed2|72.22|5.344|61.68|8.25|21.09|91.67|
|old_dev/R0/seed0|47.57|3.222|23.68|32.64|29.51|66.67|
|old_dev/R1/seed0|65.97|4.667|31.56|10.07|28.47|86.11|
|old_dev/R_full/seed0|60.07|4.278|32.46|15.62|28.82|77.78|
|old_dev/R2/seed0|70.14|4.694|30.55|3.12|27.43|97.22|
|old_dev/R0/seed1|73.26|4.917|28.45|0.69|26.04|97.22|
|old_dev/R1/seed1|77.43|5.222|33.16|0.35|22.22|97.22|
|old_dev/R_full/seed1|73.26|5.417|35.44|7.64|21.18|97.22|
|old_dev/R2/seed1|71.53|4.972|33.59|0.35|28.13|100.00|
|old_dev/R0/seed2|72.22|5.111|27.27|9.38|20.83|86.11|
|old_dev/R1/seed2|82.64|6.056|36.01|3.47|14.58|97.22|
|old_dev/R_full/seed2|64.58|4.500|32.54|11.46|29.17|86.11|
|old_dev/R2/seed2|76.39|5.028|28.74|6.25|18.06|88.89|
|dev32/R0/seed0|49.48|3.396|21.71|26.17|33.46|72.92|
|dev32/R1/seed0|64.45|4.417|26.51|14.45|25.00|84.38|
|dev32/R_full/seed0|56.25|3.927|25.18|25.26|25.39|72.92|
|dev32/R2/seed0|61.85|4.302|31.06|6.25|34.24|93.75|
|dev32/R0/seed1|72.79|4.865|30.07|7.68|21.09|88.54|
|dev32/R1/seed1|74.74|5.031|34.24|3.91|22.27|95.83|
|dev32/R_full/seed1|73.70|5.521|35.05|9.51|18.49|88.54|
|dev32/R2/seed1|72.01|4.781|27.92|4.04|25.00|96.88|
|dev32/R0/seed2|70.96|4.969|31.03|11.33|21.09|88.54|
|dev32/R1/seed2|83.07|6.208|38.98|2.60|14.84|97.92|
|dev32/R_full/seed2|67.58|4.698|30.82|9.90|25.52|87.50|
|dev32/R2/seed2|71.22|4.688|29.58|10.03|20.18|89.58|

Seed0 continuation gate: `{'passed': True, 'conditions': {'modes_above_R0': True, 'modes_above_R1': True, 'shared_recall_above_R1': True, 'validity_loss_at_most_02': True}}`.

Family bootstrap results, all failed requests and input-stream identities are saved in RESULTS.json.
