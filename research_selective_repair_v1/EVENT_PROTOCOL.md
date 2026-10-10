# First-crossing measure study,2026-10-10

The conditioned recurrent joint-prefix pilot completed2400 updates. On two
held-out TRAIN families its tipRMSE=.331745m,jointRMSE=25.7135rad,crossing
tipRMSE=.190854m,completion-proxy Brier=.247774. It fails to identify meaningful
free state transport. Its unconditioned predecessor already had .073742m
tipRMSE. Preserve all artifacts; no DEV screening of the failed joint model.
The paired conditioned memoryless model is still running and must be reported.

Change the learning object,not the loss weights or projection iteration grid:
learn a censored measure on first-crossing position and controller termination.
Two actual row crossings determine the operational word. Seven periodic joint
coordinates add an unnecessary identification problem for that purpose. Actual
controller prefixes supply first-cross/no-first-cross labels until the first
event or stop; never label the unknown suffix. Use existing exact fulltrace
events,not compressed four-state polylines.

Minimal EventHead:causal64GRU,4shared mixture components,per-segment first-stop
and two first-cross hazards,independent y/z Gaussian position per crossing.
Each mean is relative to the current requested segment endpoint; scale has a
fixed .005+.295*sigmoid parameterization. No state transport,FK,DLS,native IK
or actual future state enters forward. Initial component weights use only the
first requested point/current context; the future requested endpoint cannot
change a previous predicted event. Conditional completed-route successful-clear
factor uses the final causal representation. The joint likelihood sums observed
censored log factors and known crossing Gaussian log density,with no auxiliary
loss-weight search. A shared component couples both row categories; residual
conditional factorization between event,stop and clearance remains approximate.

Integrate Gaussian position mass over current NN-inferred gap/over regions,
then compose16 word probabilities plus failure. No truth geometry in forward.
The same24internal alternatives yield exactly8finals then the unchanged q/4.
Keep the current word-safe eligibility and predicted preservation rule unchanged.
Count24route forwards,4hypotheses and30912 GaussianCDF evaluations per request;
zero native planner/controller/FK calls by this additional event model. All
common upstream geometry/encoders and downstream scoring costs still count.

Compare2400updates,batch32,lr.0003,paired seed0 on completed384TRAIN traces:
causal GRU versus independently per-node MLP with matched parameter scale,
the exact separately fitted binary control and historical17-class classifier.
No second sigma/component/weight/threshold search. First report held-family
crossing position error,occurrence and successful-clear Brier; physical final
advantage cannot be inferred from these. If informative,freeze before a pilot
DEV screen/actualreturned4 comparison to strongest planned-word,binary and
coordinate controls. This is development evidence,not fresh confirmation.
Full2304label training,three paired seeds,matched heads,B1/B3,new isolated
confirmation/native repeats/common all8 damage/fullonline timing still required.
Original U8 and prospective body gates remain unchanged. Existing mathematical
ingredients are not claimed as novel; the selective counterfactual capability
must be demonstrated rather than assumed.
