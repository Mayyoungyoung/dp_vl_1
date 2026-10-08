# Scoring diagnosis

All panels use the same fixed safety_mean seed0 generator. Left: original
reserved scoring domains and paired DEV have different actual validity rates;
this is not proof of pure label shift. Middle: equal-width reliability bins
on exactly the same288 pairedDEV requests, original q and three ordinary
matched scorers fitted only in the old scoring domain. Bins have unequal counts;
lines are descriptive, not uncertainty bands. Right: public fixed q selector
compared with plain top-q and a non-deployable candidate-pool oracle bound.
Oracle uses actual validity/mode labels only for this upper bound. No new
candidates, threshold fitting, training-seed averaging or final-test claims.
