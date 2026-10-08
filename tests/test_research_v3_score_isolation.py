import numpy as np
from scripts.research_v3_score_isolation import geometry_digest


def test_geometry_identity_ignores_object_order_but_detects_size_or_goal_change():
    c=np.array([[.1,.2,.3],[.2,.1,.5]]);h=np.array([[.01,.02,.03],[.03,.04,.01]])
    g=np.array([[.5,0,.8],[.5,.2,.8],[.5,-.2,.8]])
    a=geometry_digest(c,h,g)
    assert a==geometry_digest(c[::-1],h[::-1],g[[2,0,1]])
    hh=h.copy();hh[0,0]+=.003;assert a!=geometry_digest(c,hh,g)
    gg=g.copy();gg[0,0]+=.003;assert a!=geometry_digest(c,h,gg)
