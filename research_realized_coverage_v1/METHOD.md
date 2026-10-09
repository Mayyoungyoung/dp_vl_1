# Actual-output coverage research (active)

Input and route decoder are inherited from C: frozen observed RGB-D/Qwen-language
encoders and a shared semantic-word-conditioned24-point3D route head. Candidate
attention stays enabled. Sixteen two-row words are operational labels, not general
homotopy classes. The complete old q encoder/scorer and8-to4rule remain frozen.

For an eight-query multiset A={(m_i,v_i)}, decode P(x,A). Only independently valid
paths have official actual mode labels. Define U8 as the number of distinct valid
actual words, V8 as valid fraction, and U4,V4 after frozen return selection.
For one replaced query A', measure delta=(U8',8V8',U4',4V4')−(U8,8V8,U4,4V4).
Recompute all eight routes. A newly requested word that destroys other outputs
does not receive positive net credit. Invalid raw signatures are diagnostic only.

## Frozen-snapshot feedback

Each existing TRAIN request has the adaptive query set and sixteen replacement
words in two rotating slots, preserving the other seven(mode,variant)pairs.
Store every full path/event, validity, official/raw words, q, returned slots,
added/lost mode bitsets, four changes and exact model/source hashes. Missing
reference is not a physical negative; observed generation failure is an outcome.
The DEV version is diagnostic only and never enters training.

## Matched ordinary and set-aware allocation

The ordinary success head maps observed context to16scores. BCE supervises whether
each queried word actually became a valid route of that same word, averaged over
sampled companion contexts. Eight highest-scoring distinct words are requested.
Scores describe realization, not environmental existence.

The first set model embeds all queries, combines mean/max token features and a
word histogram with current context, and regresses the four actual set outcomes.
Training uses both absolute outcome MSE and measured replacement-difference MSE.
At deployment compute at most two full single-query neighborhoods (128query sets
per sweep), require predicted U8gain>.1 and no quality count decrease below−.02,
then perform exactly one final eight-route decode. These are fixed heuristics,
not oracle selection or monotonicity guarantees. Computation and latency reported.

The simple and set models receive identical feedback pools and update schedules.
The first net model underperformed success sorting; its failure remains recorded.
Success-centered feedback is the next query-distribution repair, shared by both.

## Displacement-free geometry collaboration

All continuations start from identical C and use identical steps/lr. Ordinary
training retains all legal witnessed modes. Gap training keeps50% ordinary draws
and50% near-deployment query contexts with one witnessed failure mode injected.
Failure sampling has a0.1floor. Coordinates come from existing same-mode positives
in the CURRENT scene, with inherited clearance/event losses. Normalize regression
over eligible queried routes, so missing references add no invented target.
Hard control injects the same type of failed mode using conventional companions.
KL-only control isolates old D2 proposal KL with no displacement term whatsoever.

Updated decoders require fresh realization feedback and new bound proposal heads.
No stale failure label is treated as permanent capability. There is no gradient
through true discrete mode counts, no RL, and no new trajectory scoring model.

Novelty remains unestablished. Conditioning, supervised success, set regression,
hard-example training and diversity sampling are conventional components. Only
stable matched controls/ablations can establish an additional coordination mechanism;
see RELATED_WORK.md. Main task outputs certify end-effector route validity only.
