import numpy as np
from scripts.paired_modes_data import registration
from scripts import observed_layout_variation as geom
from scripts.collect_observed_layout_variation import route_acceptance


def test_roles_and_pair_identity():
    r=registration();p=r['parent_plan']
    assert len(p)==480
    assert len({x['family_id'] for x in p if x['role']=='TRAIN'})==128
    assert len({x['family_id'] for x in p if x['role']=='DEV_MODEL'})==32
    for i in range(0,480,3):
        a,b,c=p[i:i+3]
        assert a['role']==b['role']==c['role']
        assert a['family_id']==b['family_id']==c['family_id']
        assert a['target_colors']==b['target_colors']==c['target_colors']
        assert a['config']['goal_xyz']==b['config']['goal_xyz']==c['config']['goal_xyz']
        assert a['config']['entry_xyz']==b['config']['entry_xyz']==c['config']['entry_xyz']


def test_closed_low_relation_does_not_claim_over_infeasible():
    for p in registration()['parent_plan']:
        closed=[c for c in p['low_gap_certificates'] if c['closed_for_this_low_relation']]
        assert len(closed)==(1 if p['variant']=='closed' else 0)
        assert all(not c['all_route_classes_exhausted'] for c in closed)
        assert any('over' in s['intent_supervision_only'] for s in p['guide_plans'][0])


def test_geometric_teachers_keep_full_segment_contract():
    for p in registration()['parent_plan'][:12]:
        c=p['config'];accepted=0
        for t,slots in enumerate(p['guide_plans']):
            for s in slots:
                if not s['collection_allowed']:continue
                xyz=np.vstack([c['entry_xyz'],s['waypoints_supervision_only']])
                fields,h24,passed=route_acceptance(xyz,np.asarray(c['goal_xyz']),t,c)
                if passed:
                    accepted+=1
                    assert h24.shape==(24,3)
                    assert fields['tip_polyline_clear'] and fields['tip_polyline_24_clear']
                    assert fields['actual_route_type']==tuple(s['intent_supervision_only'])
        assert accepted>0
