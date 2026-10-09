# Mode-conditioned geometry-adaptive route generation

This implementation changes the generator's representation. A passage word
selects a learned semantic query; a single shared decoder maps that query and
the **current observation** into a complete24-point3D route. The result is eight
routes, scored by the unchanged complete q model and reduced to four by the
existing selection rule. A mode embedding alone is not claimed as a novel method.

## Observation and mode representation

Let x contain the pinned frozen Qwen RGB/language features, observed RGB-D point
cloud and current gripper state. The historical observed geometry encoder yields
a context h(x) and estimated target surface anchor a(x). These encoders are frozen
in all new arms. During training their outputs are cached, exactly as produced
from each scene's matching images, depth, camera and language features; truth
obstacle boxes never enter this encoding. The complete scorer has its own frozen
historical encoder and is not replaced by the generator's context.

The vocabulary contains16 ordered words in {gap0,gap1,gap2,over}². Each component
refers to the first forward crossing of a registered obstacle row. These labels
are comparable within the controlled two-row scene families, including their
open/closed/shifted variants. They are **not** general homotopy classes and the
model does not solve arbitrary obstacle topologies. Only checker-valid paths
are assigned official valid-mode labels; raw invalid-path signatures appear
solely in failure diagnostics.

A lightweight MLP predicts allocation logits l(h). Evidence is ternary: witnessed
positive, certified impossible internal passage, or unknown. Missing positive
references do not establish impossibility. Negative labels are limited to words
containing a geometrically closed inflated internal passage.

## Shared decoder

For word m and within-word occurrence v, initialize

`z_m,v = h(x) + E[m] + V[v]`.

The inherited conditioned blocks and shared output projection produce residual
points and event logits. Every route begins exactly at the observed current
position. Interior points are offsets from a line to the learned observed anchor;
the endpoint has the existing bounded residual around that anchor. No exact goal
coordinate or test-time collision repair is supplied.

`p_theta(x,m,v) = decode_theta(h(x), a(x), current, E[m], V[v])`.

E is initialized from the mean nearest historical query on TRAIN witnesses, not
from DEV modes. The shared blocks/output start from safety_mean. B retains its
original eight learned queries. B/C/D have identical initial shared tensors and
actual sampled observation/target streams within each continuation seed.

Mode identity is not output position: permuting distinct mode queries permutes
the corresponding coordinates, as tested. Within-word V permits multiple path
variants. The initial decoder retains candidate self-attention, whose possible
effect on semantic identity is explicitly diagnosed.

## Three loss groups

`L = L_route + 0.001 L_mode + lambda_pair L_pair`, with lambda_pair1 for B_pair/D,
and0 for B0/C. L_route is XYZ regression plus constant reach-event regression
and inherited continuous segment-clearance/workspace-floor penalties (weight160).
This is task-route supervision, not a robot control cost. Grounding is frozen.

L_mode combines (i) normalized likelihood of positively witnessed words as
proposal allocations and (ii) BCE only on known positive/negative evidence.
Unknown entries are masked from binary supervision. Allocation competition
necessarily ranks even unknown words; this score is **not calibrated existence**.

For a legal scene pair (a,b), same word m, and verified paths y_a,m and y_b,m:

`L_pair = mean_(a,b,m in W) ||[p_theta(b,m)-p_theta(a,m)]-[y_b,m-y_a,m]||²`.

W contains only positively witnessed common words. Both predicted coordinate
tensors receive gradients. When old coordinates fail, y_b is a new valid route,
so the desired displacement is nonzero. Unknown/deleted words have no edge.
The target pair is chosen from existing verified references, preferring a source
path that fails in the destination if available, then the closest valid same-word
destination path. Six training queries cover shared words, and remaining queries
sample other single-scene positive modes; all56920 prior TRAIN positives remain
eligible. There is no additional positive collection.

This loss couples regression errors: writing e_a=p_a-y_a and e_b=p_b-y_b gives
L_pair=||e_b-e_a||². Thus it is an inductive bias in supervised fitting, **not** a
new source of information beyond paired coordinate targets. B_pair receives
the same displacement objective after ordinary per-scene assignment; B0 is the
ordinary no-pair control. A D advantage must survive both controls and D vs C.

## Inference

Only current observation inputs are accepted. The API's adaptive allocator takes
at most eight words with nonnegative logits and fills spare candidates with
within-word occurrences. If none pass, it uses the top-ranked word. This fixed
threshold is a proposal heuristic, not proof that other modes are impossible.
The API always emits eight routes. Original top-eight distinct-word allocation
(`balanced`) and categorical sampling with replacement (`ordinary`, fixed seed)
are retained as inference ablations on the same weights. No old-scene route,
parent retention label, verifier or human correspondence is used at inference.

## Difference from historical research

R2/R3 kept ordinary slot queries and matched boundary-relative crossing
descriptors across positively witnessed relations; R3 further required predicted
presence. Here semantic words enter the decoder before coordinates are generated,
and paired supervision targets actual nonzero3D path deformation. No old R2/R3
loss is relabeled or rerun. Set-point/current-route positives and retention-priority
matching do not implement semantic conditional decoding.

The distinction is an implementation/mechanism distinction, not proof of novelty
or superior performance. Mode/intention-conditioned multimodal trajectory
generation is established prior art; see the bounded discussion in PAPER_DRAFT.

## Bounded repairs and deployment

Removing candidate self-attention eliminates companion-query interference but
worsens DEV coverage/repair; this variant was rejected after seed0. D2 instead
adds a proposal connection to the same pair-loss group:

`L_pair_D2 = L_displacement + 0.001 * (KL(q_a||q_b)+KL(q_b||q_a))/2`.

Here q is normalized only over positively witnessed common words. Unknown and
unavailable words have no gradient from this term. XYZ displacement supervision
remains active. This conventional consistency term repairs a missing training
connection; it is not claimed as a new algorithm. D2 is reported separately for
all three seeds. B_set strengthens the ordinary control by minimizing over all
existing same-mode positives before assignment, without a conditional decoder.

`load_fixed_scored_mode_planner` preserves the entire frozen scorer and its own
encoder. Actual checkpoint replay exactly matches saved paths, events, scores
and selected indices; see results/DEPLOYMENT_REPLAY.json.
