# Closed-instruction localization information check

The mean generator has12/288 requests with all endpoints incorrect, all with
visible goal support and an anchor outside its correction box. Attention mass
aggregation failed its full-route gate. This does not establish missing visual
information or justify larger VLM training. Reuse the historical ordinary RGB-D
prototype localizer to test a distinct explanation: target color information
may be present but poorly read by the learned shared anchor.

Fit existing observation_prototype_grounding.fit_prototypes on all1152 original
paired TRAIN requests only. Keep every historical CONFIG constant. Use RGB-D
grid stride2 in both fitting and prediction, same sampled points as the route
geometry encoder. No oracle mask, target coordinate or reference at prediction.
Positive demonstration endpoints may select TRAIN pixels, as in the original
baseline. Every parent contributes equally within each instruction; all family
edits/targets remain in their original split. Save model, source hashes, all
predictions and failures. Exact-instruction lookup is a closed-vocabulary control,
not open-language generalization or a new algorithm.

Evaluate all288 original pairedDEV requests once with unchanged3cm goal rule.
Compare endpoint semantic accuracy with the mean model's ANY correct endpoint
(not full route validity). Paired bootstrap32 families/10000 draws/seed610091.
Only consider a later coherent route intervention if gain>=6/288 and CI lower>0.
No threshold/prototype tuning after DEV. Report resource cost and abstentions.
Failing gate closes this alternative; passing gate only establishes an ordinary
localization opportunity. No route-level, robot-execution or novelty claim.
Run serially after current score/calibration queues, under existing V3 budget.
