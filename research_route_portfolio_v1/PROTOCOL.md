# Learning diverse three dimensional route portfolios

The current user asks for a useful, differentiated paper prototype within an established research area. A defensible advantage in one relevant direction is sufficient to continue. We therefore study finite-budget route coverage from multimodal demonstrations, using the existing semantic conditional generator as the core and preserving unfavorable comparisons.

## Method and claim

The observation-only generator separates passage proposal from complete route geometry. Training balances positively witnessed passage words rather than treating every reference coordinate sequence as a separate mode. Certified closed internal passages supply negatives; unwitnessed words remain unknown. A shared decoder maps current RGB-D, language, state and semantic mode tokens to eight routes. An optional ordinary realization-success head allocates eight distinct tokens using TRAIN-only feedback. The same complete observation-based scorer returns four.

Mode conditioning, masked supervision, and success prediction each have prior art. The proposed contribution is their concrete use for finite-budget, multimodal 3D robot route portfolios and the evidence separating candidate coverage, geometry adaptation and downstream return quality. Operational two-row passage words are not general homotopy classes. No displacement loss or native-joint feedback is part of the main method.

## Frozen development comparison

Before starting the new experiment, freeze B0, Bset and C generator continuations for all three seeds, the complete shared q, and five generator-bound C0 success heads. No new fitting. Each generator C also receives three separate categorical inference trials. Every trial generates exactly eight routes; repeats are never pooled into a larger set. Store and hash all predictions before opening independent geometry for checking.

Evaluate all 336 requests of the allowlisted DEV_MODEL export from 16 families, including open, shifted, closed, narrow, tall, noise and occluded variants. These families differ from the original training families but were already used in an earlier study; they are reused development evidence. TEST_LOCKED and score/calibration reservations remain inaccessible for method selection.

Primary outcomes are actual distinct valid modes@8 and validity@8. Report duplicate valid-route slots as a mechanism diagnostic. Also report returned distinct modes@4, validity@4 and every variant. Confidence intervals resample full scene families and generator continuations, with inference repetitions averaged within each continuation. Five success-head seeds on C0 are conditional head replications, not five generator seeds. Do not choose a favorable corruption, checkpoint, seed or arm after evaluation.

Historical +0.15 coverage and conjunctive quality gates remain recorded as failed in their original studies. The current scope replaces the demand for universal superiority; it does not turn those failures into passes. Any supported effect is a retrospectively selected research direction, requiring future frozen confirmation.

## Late preference experiment

Freeze a separate appendix experiment before its outcomes are read: all three C continuations, all336requests, and exactly three requester predicates, ground_only (neither row is over), over_any (at least one row is over), and first_gap0 (first row is gap0). These are symbolic desired words supplied by the caller, not geometric truth or learned language parsing. Allocate inside the mask using the existing proposal score; retain the original positive-score repetition rule and highest legal fallback. Decode eight and return four with original full q. Cache observation encodings across independent preference calls; each call still spends eight routes.

The control decodes the ordinary adaptive eight-route portfolio before receiving the preference. Then it prioritizes compatible *nominal query metadata* with the same q selector, filling to four when fewer are compatible. No oracle mode labels enter selection. Evaluate actual valid compatible modes and fraction under the independent checker, including misrealized modes and all failures. Store all preference predictions before checking truth. This measures an additional capability; it neither assumes an advantage nor upgrades ordinary mode masking to an independently novel algorithm.

## Deliverables and remaining evidence

Deliver a method description, accurate related-work distinctions, a complete English paper draft, tables with tradeoffs, reproducible code, source/prediction hashes and actual terminal receipts. A paper prototype is distinct from a ready-to-submit claim. Independent frozen confirmation, modern diffusion or conditional-latent baselines, less restrictive passage topologies, and controlled controller or real-robot validation remain concrete work to justify a robotics submission.

Use only ssh wzy3090, GPU1 UUID7506746b, memory fraction .35, CPU0-3/four threads. Export immutable local commits; freeze launcher/imports. Record all failures and command seconds. Historical source, data and runs/main stay intact. No new total-time cap is introduced by this protocol.
