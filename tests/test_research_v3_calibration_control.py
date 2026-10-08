import numpy as np
from scipy.special import expit
from scripts.research_v3_calibration_control import fit_calibrators


def test_calibration_recovers_known_positive_affine_map():
    # Fractional Bernoulli expectations make optimum known, no random labels.
    x=np.linspace(-3,3,1001);target=expit(.7*x+.8)
    fitted=fit_calibrators(expit(x),target)['affine']
    assert abs(np.exp(fitted['a'])-.7)<1e-3 and abs(fitted['b']-.8)<1e-3


def test_extreme_probabilities_are_finite_and_temperature_cannot_flip_order():
    q=np.array([0.,.1,.3,.7,.9,1.]);y=np.array([0.,0.,0.,1.,1.,1.])
    fit=fit_calibrators(q,y)
    assert all(np.isfinite(r['calibration_nll']) and np.exp(r['a'])>0 for r in fit.values())
