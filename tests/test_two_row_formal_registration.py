import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts import register_two_row_formal as formal

ROOT=Path(__file__).resolve().parents[1]


def inputs():
    load=lambda name:json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))
    return (load('observed_two_row_formal116_v1.json'),load('observed_two_row_pilot_v4.json'),
            load('observed_two_row_formal116_exclusions_v1.json'))


@pytest.fixture(scope='module')
def registration():
    return formal.build_registration(*inputs())


def test_one_shot_registration_is_deterministic_and_does_not_mutate_inputs(registration):
    original=inputs();saved=copy.deepcopy(original)
    assert formal.build_registration(*original)==registration
    assert original==saved
    assert not registration['simulation_started'] and not registration['training_authorized']


def test_all_roles_parents_and_requested_budget_are_preserved(registration):
    plans=registration['parent_plan']
    assert len(plans)==116 and len({p['parent_id'] for p in plans})==116
    assert [p['seed'] for p in plans]==list(range(283200,283316))
    assert Counter(p['role'] for p in plans)==dict(TRAIN=64,DEV_MODEL=12,DEV_SCORE=12,CALIBRATION=12,TEST_LOCKED=16)
    assert registration['requested_routes']==sum(p['config']['requested_route_proposals'] for p in plans)==3132
    for p in plans:
        assert p['role']==p['split']==p['config']['split']==formal.role_for_index(p['index'])
        assert p['config']['selected_target_indices']==[0,1,2]
        assert p['config']['entry_xyz']==[0.,0.,.865]
        assert p['config']['requested_setup_actions']==0 and p['config']['preparation_xyz']==[]
    assert not registration['reference_set_complete'] and registration['all_solution_count'] is None


def test_execution_order_and_two_shards_do_not_reassign_roles(registration):
    order=list(range(16))+list(range(64,76))+list(range(16,64))+list(range(76,116))
    ids=[registration['parent_plan'][i]['parent_id'] for i in order]
    assert registration['execution_indices']==order and registration['execution_order']==ids
    assert registration['shards']==[ids[::2],ids[1::2]]
    assert set(registration['shards'][0]).isdisjoint(registration['shards'][1])
    assert [x for pair in zip(*registration['shards']) for x in pair]==ids


def test_exact_rng_draw_order_geometry_and_official_colors(registration):
    spec,_,_=inputs();rng=np.random.RandomState(283199)
    for p in registration['parent_plan']:
        c=p['config'];sample=lambda bounds:float(rng.uniform(*bounds))
        assert c['row_x']==[sample(b) for b in spec['row_x_ranges']]
        assert c['post_y']==[[sample(b) for b in r] for r in spec['post_y_ranges']]
        assert c['goal_xyz']==[[sample(spec['goal_x_range']),sample(b),.84] for b in spec['goal_y_ranges']]
        idx=rng.choice(20,3,replace=False).tolist()
        assert p['target_colors']==[formal.color_table()[i] for i in idx]
        assert len(set(x['name'] for x in p['target_colors']))==3
        assert c['post_size_xyz']==[.035,.035,.14]
    assert spec['color_table_sha256']==formal.canonical_hash(formal.color_table())


def test_mechanical_hashes_ignore_colors_and_seed_but_detect_geometry(registration):
    p=registration['parent_plan'][0];c=p['config'];centers,halves=formal.batch.collector.geometry(c)
    for suffix,q in [('exact',None),('1mm',.001)]:
        assert p['registered_geometry_'+suffix+'_sha256']==formal.batch.scene_geometry_hash(centers,halves,c['goal_xyz'],q)
    changed=centers.copy();changed[0,0]+=.004
    assert formal.batch.scene_geometry_hash(changed,halves,c['goal_xyz'],.001)!=p['registered_geometry_1mm_sha256']


def test_cross_split_duplicate_closes_both_members_and_retains_budgets(registration):
    plans=copy.deepcopy(registration['parent_plan']);left,right=plans[0],plans[64]
    for suffix in ('exact','1mm'):
        right['registered_geometry_'+suffix+'_sha256']=left['registered_geometry_'+suffix+'_sha256']
    gate=formal.apply_duplicate_gate(plans,[])
    assert all(not p['usable_for_model'] and not p['collection_allowed'] for p in (left,right))
    assert all(p['predeclared_unattempted_routes']==27 for p in (left,right))
    assert {left['parent_id'],right['parent_id']}<=set(gate['blocked_parent_ids'])
    assert any(g['cross_split'] for g in gate['duplicate_groups'])
    assert len(plans)==116 and sum(p['config']['requested_route_proposals'] for p in plans)==3132


def test_prior_geometry_match_is_closed_without_replacement(registration):
    plans=copy.deepcopy(registration['parent_plan']);p=plans[2]
    old=dict(parent_id='two_row_reach_283001',version='v4',source='public metadata',source_sha256='a'*64,
        geometry_exact_sha256=p['registered_geometry_exact_sha256'],geometry_1mm_sha256=p['registered_geometry_1mm_sha256'])
    gate=formal.apply_duplicate_gate(plans,[old])
    assert p['parent_id'] in gate['blocked_parent_ids'] and len(gate['prior_geometry_hits'])==2
    assert not p['usable_for_model'] and p['predeclared_unattempted_routes']==27


def test_precheck_failure_does_not_redraw_geometry_colors_or_budget(registration,monkeypatch):
    calls=[]
    def failed(config):
        calls.append(config['seed'])
        return dict(passed=False,checked=0,error='synthetic fixed precheck failure')
    monkeypatch.setattr(formal.batch,'geometry_precheck',failed)
    result=formal.build_registration(*inputs())
    assert calls==list(range(283200,283316))
    assert result['predeclared_unattempted_routes']==3132
    for old,new in zip(registration['parent_plan'],result['parent_plan']):
        assert old['config']==new['config'] and old['target_colors']==new['target_colors']
        assert old['registered_geometry_1mm_sha256']==new['registered_geometry_1mm_sha256']
        assert new['registration_failure_reasons']==['geometry_precheck_failed']


@pytest.mark.parametrize('key,value',[('layout_rng_seed',283198),('requested_route_proposals',3133),
    ('post_height_m',.13),('setup_path_budget',1),('geometry_resampling_attempts',1)])
def test_changed_registration_rejected_before_any_sampling(key,value,monkeypatch):
    spec,base,excluded=inputs();spec[key]=value
    def forbidden(*a,**kw):raise AssertionError('No drawing before validation')
    monkeypatch.setattr(formal.np.random,'RandomState',forbidden)
    with pytest.raises(ValueError,match='registration changed'):formal.build_registration(spec,base,excluded)


def test_old_pilot_dev_guard_remains_strict(registration):
    config=registration['parent_plan'][0]['config']
    with pytest.raises(ValueError,match='pilot role'):formal.batch.collector.validate_config(config)
    compatibility=copy.deepcopy(config);compatibility.update(split='DEV_COLLECTION',requested_setup_actions=1)
    formal.batch.collector.validate_config(compatibility)
    assert config['split']=='TRAIN' and config['requested_setup_actions']==0


def test_incomplete_tampered_or_locked_exclusions_rejected():
    spec,base,excluded=inputs()
    with pytest.raises(ValueError,match='All nine'):formal.build_registration(spec,base,excluded['entries'][:-1])
    bad=copy.deepcopy(excluded);bad['entries'][0]['role']='TEST_LOCKED'
    with pytest.raises(ValueError,match='public development'):formal.build_registration(spec,base,bad)
    bad=copy.deepcopy(excluded);bad['entries'][0]['geometry_1mm_sha256']='a'*64
    with pytest.raises(ValueError,match='exclusion table changed'):formal.build_registration(spec,base,bad)


def test_canonical_state_is_fixed_and_no_old_beforeworld_reference():
    spec,base,excluded=inputs();spec['canonical_init']['canonical_arm_joints'][0]+=.001
    with pytest.raises(ValueError,match='canonical static'):formal.build_registration(spec,base,excluded)
    spec,base,excluded=inputs();spec['canonical_init']['before_world']='old_failed_scene'
    with pytest.raises(ValueError,match='before-world'):formal.build_registration(spec,base,excluded)


def test_cli_existing_registration_is_never_overwritten(tmp_path,monkeypatch):
    existing=tmp_path/'registration.json';existing.write_text('original evidence')
    monkeypatch.setattr('sys.argv',['register','--spec','never_read','--base-config','never_read',
        '--excluded-hashes','never_read','--output',str(existing)])
    with pytest.raises(FileExistsError,match='never overwrite'):formal.main()
    assert existing.read_text()=='original evidence'
