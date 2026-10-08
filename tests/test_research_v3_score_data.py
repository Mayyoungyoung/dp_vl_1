from scripts.research_v3_score_data import registration


def test_new_roles_are_family_disjoint_and_registration_is_deterministic():
    a,b=registration(),registration()
    assert a==b
    plans=a['parent_plan'];assert len(plans)==192
    family_roles={}
    for p in plans:
        family_roles.setdefault(p['family_id'],set()).add(p['role'])
        assert p['role']==p['config']['split']==p['split']
        assert p['seed']==p['config']['seed']
    assert len(family_roles)==64 and all(len(r)==1 for r in family_roles.values())
    assert sum(r=={'SCORE_TRAIN'} for r in family_roles.values())==32
    assert sum(r=={'DEV_SCORE'} for r in family_roles.values())==16
    assert sum(r=={'CALIBRATION'} for r in family_roles.values())==16
    assert len({p['registered_geometry_1mm_sha256'] for p in plans})==192
    assert all(p['geometry_precheck']['passed'] for p in plans)
