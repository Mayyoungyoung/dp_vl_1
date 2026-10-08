# Eight-target ordinary set control — registered before training

The five-arm pilot supports biased-retention failure, but full-set matching
processes1,566,630 target slots versus307,200 in sampled arms. Its observed
gain cannot isolate distinct matching from richer reference processing.

New conventional set_sampled arm: randomly choose min(8, witnessed modes)
distinct modes, uniformly sample one actual positive path within each, then
fill any remaining slots with ordinary mode-balanced draws. Match these eight
references to eight candidates with the existing Hungarian regression loss.
No new architecture, paired loss, scorer, geometry feature or oracle inference.
Use the same R1 seed0 initializer, seed0 input stream,1200x32, optimizer and
grounding/clearance losses; same support and all three edit observations.
Exactly307,200 reference target slots, matching empirical/balanced controls.
The loss sampler differs by design; retain its actual state/exposure and replay.

Evaluate fixed last1200, original complete fixed q, all288 DEV_MODEL requests.
Report all generation and protection metrics, father-family bootstrap intervals
versus balanced and full-set matching, actual time and target processing count.
Screening evidence for useful ordinary coverage: rare recall@8 gains at least
5pp over balanced with a positive parent-bootstrap lower bound, candidate
validity no more than2pp lower. This is not a new-method gate or final test.
If it matches full-set quality, use it as an economical strong baseline; if it
fails, the richer within-mode target set or free assignment remains relevant.
No novelty claim can follow from stratified sampling plus Hungarian matching.
