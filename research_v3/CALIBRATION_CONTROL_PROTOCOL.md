# Separate new scoring supervision from ordinary recalibration

Registered while new paired-score collection is running, before its CALIBRATION
outputs or trained scores exist. The old-to-paired candidate prevalence shift
could partly be addressed by calibration alone. Compare unchanged original
full q with: (1) newCAL temperature scaling, (2) newCAL monotone affine logit
scaling. Retain original q, all three newly fitted scorers and both controls.
No selection between calibrators on DEV. Same actual safety_mean candidate pool.

Both calibration controls use exactly the newCALIBRATION predictions/labels,
never DEV labels for fitting. Temperature logT in[-3,3] minimizes float64 BCE.
Affine map is sigmoid(exp(a)*logit(q)+b), a in[-3,3],b in[-10,10], initialized
a=b=0 and fit with analytic float64 gradient/L-BFGS-B. Positive slope preserves
ranking, though the deployed diversity selector's fixed.5 gate may change.
Report optimizer status, calibration loss, parameters, all DEV probability
and K4 metrics. No new threshold fitting, no claim of distribution-free risk.

New score-data gate stays unchanged. These extra conventional controls delimit
the explanation: benefit from recalibration cannot be claimed as score-head
learning, and benefit from new data cannot be claimed as architectural novelty.
If new fitting has no advantage over these controls, keep the simpler control
as the research baseline and diagnose a different unmet task requirement.
