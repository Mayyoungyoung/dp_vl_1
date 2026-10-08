# Ordinary linear margin control, prospective diagnostic and comparison

All189 final mean-checkpoint TRAIN post collisions affect1–3 segments and have
maximum expanded-box clearance deficit <=15.258mm, median2.720mm. The quadratic
penalty's derivative decreases with that deficit. This is a mathematical
property, not evidence that a new algorithm is needed. Test a conventional
linear hinge before claiming a sophisticated boundary/safety mechanism.

Diagnostic: exactly the four fixed32-request TRAIN batches drawn with seed0,
same support and gradient computation as the earlier audit, now starting from
the sealed safety_mean checkpoint. Compare160*mean(deficit²) with unscaled
mean(deficit), across all segments and candidates. Fit one coefficient as the
median ratio of their parameter-gradient norms on batches with nonzero linear
gradient. No coefficient search. Gate requires at least10 colliding paths,
at least3 defined ratios, finite positive coefficient, and median cosine of
the two penalty gradients >=0.5. Report regression/grounding gradient cosines
as context, not as proof of causal conflict. No DEV access in diagnostic.

Only after this gate passes: two ordinary continuations from the exact same
safety_mean checkpoint,1200 steps each, identical complete-set supervision,
same seed0 input/assignment streams and new AdamW optimizer. One repeats the
existing160*mean(deficit²); the other uses fixed coefficient*mean(deficit).
Both keep160*workspace-floor penalty, regression, grounding, architecture and
M8/H24 unchanged. No margin, learning-rate, weight or seed sweep. The pair
explicitly controls extra training; compare both with their common parent.

Evaluate all288 DEV requests with the same complete preselected paired-domain
seed0 scorer and its frozen CAL temperature. Do not refit q or temperature.
Record unfiltered candidates, exact oracle events after inference, selection,
known/rare coverage, Brier and paired32-family intervals. This q comparison is
fixed-function transfer onto each arm's candidates, not a same-pool scoring gain.
For any adoption, linear-minus-mean validity must gain>=2pp with CI lower>0,
distinct valid modes gain>=0.15, rare recall loss<=2pp, any-valid loss<=1pp,
and Brier degradation<=0.01. Passing still establishes an ordinary baseline,
not novelty; failing closes this loss shape without tuning.

Require exact initial tensor and input/target exposure equality, finite losses,
full optimizer/RNG checkpoints. Diagnostic <=60s; each training <=360s,
evaluation <=60s. GPU1 UUID/35%/4threads, all under remaining V3 cumulative cap.
No final TEST, collector directory evaluation, robot execution or altered data.
