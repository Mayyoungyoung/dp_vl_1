# Whole-route validity versus average segment penalty

After ordinary full-set matching, mean valid candidates6.576 and distinct valid
modes6.566 differ by only0.010 per request. Duplicate candidates are almost
eliminated, so adding diversity repulsion is not motivated. Candidate validity
is82.205%. Inspect a concrete alternative explanation before changing the model.

Using the fixed completed set_matching model, first four TRAIN batches from
the seed0 sampler, measure exact per-segment deficits from the registered2cm
margin, number of violating segments per route, and parameter-gradient norms
and cosines for reference regression, grounding and collision losses. Compare
mean squared deficits over all23 segments with maximum squared deficit within
each path, averaged over paths. No optimizer update, DEV, or locked inputs.

Only if at least5% TRAIN candidate paths collide and at least half of colliding
paths have five or fewer violating segments, consider a two-arm conventional
safety control. Freeze a max-loss coefficient as the median of four ratios of
gradient norms: norm(160*mean penalty)/norm(unscaled path-max penalty). Preserve
the original workspace-floor penalty. This controls initial gradient magnitude
while testing allocation to whole-path bottlenecks, not a coefficient sweep.

If this gate passes, register before training: same set_matching initializer,
same full-set support, same1200x32 stream, plain continuation versus path-max
continuation; same fixed q. Require validity gain>=2pp with positive family CI,
distinct valid modes gain>=0.15 and rare recall loss<=2pp before replication.
Report failures and all metrics regardless of gate. This remains an ordinary
strong-baseline test. It cannot establish algorithmic novelty by itself.
