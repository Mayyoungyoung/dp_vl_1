# Mode-conditioned feasible-space trajectory generator

Frozen observed RGB-D/language context h, surface anchor a and current state s
are inherited from C. Sixteen two-row passage words are operational labels, not
general homotopy classes. Complete historical q and its own observation encoder
return4from8without refitting.

For each(m,v), a copied small conditioned decoder processes one token:
`h+E[m]+V[v mod2]`. Its self-attention sequence has length1. It predicts24center
nodes c and23nonnegative cell widths `r=.06sigmoid(logits)`. Center start=s.xyz;
center endpoint=a+existing bounded residual. This boundary network is independent
of all seven other queries. The peer-boundary ablation uses length8instead.

The cell `C_j=conv(c_j+[-r_j,r_j]^3,c_{j+1}+[-r_j,r_j]^3)` is a bounded convex
swept cube, a polytope with at most16vertices. Shared node j is drawn from
`c_j+[-min(r_{j-1},r_j),min(r_{j-1},r_j)]^3`. Start and endpoint radii are0.
The second copied decoder sees all8query tokens and a learned projection of
current predicted cell geometry. It predicts unconstrained parameters z:

`p_j=c_j+rho_j*tanh(z_j)`.

Phi therefore constructs final XYZ, with no extra XYZ residual outside its bounds.
Adjacent p_j,p_{j+1} are inside the same convex cell C_j, so their entire straight
segment is inside C_j. The24output nodes are the exact shared connectors; there
is no re-sampling that cuts corners. This is23degree-one Bézier pieces, not a
smooth cubic spline. It has positional continuity but does not guarantee C1or
dynamic feasibility. Existing event output is retained and supervised separately.

If a reference cell's radius is strictly below its whole-center-segment clearance
to every expanded obstacle and floor, its entire swept cube is free: signed
L-infinity clearance is1-Lipschitz to L-infinity displacement. This conventional
analytic certificate applies only with truthful geometry. Predicted c,r may be
wrong; membership alone gives no scene, mode, endpoint, IK or full-arm guarantee.
Connectedness is structural because every adjacent cell includes its shared c_j,
even when r=0. A connected but colliding/degenerate chain is a prediction failure.

Matched controls have identical modules, initial tensors and TRAIN sample streams:
freeXYZ `p=c+.06z`, relative `p=c+rho*z`, bounded `p=c+rho*tanh(z)`. PostXYZ
projection clamps that same free offset into shared-node cubes. Centerline sets z=0.
All see predicted corridor features and identical corridor supervision. The
geometric construction is existing mathematics, not claimed original.

Training: final path MSE + corridor center MSE + radius MSE +160continuous
clearance penalty +.01reach-event MSE. TRAIN oracle geometry never enters the
condition encoder. Encoders/C parameters stay frozen; both new decoders update.
The TRAIN oracle diagnostic substitutes reference cells explicitly and excludes
corridor losses. Its results are labeled mechanism diagnostics only.

Potential contribution to establish: bounded peer-aware generation with boundaries
invariant to companion replacement, beyond sameinformation XYZ/relative/projection
and centerline. It requires real coverage, returned-quality and adaptation gains;
mathematical containment is not sufficient experimental evidence.
