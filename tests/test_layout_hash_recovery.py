"""Pure saved-readback and coordinator fault tests: no simulator/model."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from scripts import collect_observed_layout_hash_recovery as r

FIXTURE=json.loads((Path(__file__).parent/'fixtures/layout_hash_readback12.json').read_text())

@pytest.mark.parametrize('case',FIXTURE['cases'],ids=lambda x:str(x['plan']['index']))
def test_all_twelve_sealed_readbacks_and_registered_hashes(case):
    p=case['plan'];c,h=r.registration.geometry(p['config']);g=p['config']['goal_xyz']
    assert r.quantized_hash_v2(c,h,g)==p['registered_geometry_1mm_sha256']
    ac,ah,ag=(np.asarray(case[k]) for k in ('actual_centers','actual_halves','actual_goals'))
    assert r.validate_initial_geometry(p,ac,ah,ag)==p['registered_geometry_1mm_sha256']
    assert (r.registration.physical_hash(ac,ah,ag,.001)==p['registered_geometry_1mm_sha256'])==(p['index'] not in (5,7))
    assert r.quantized_hash_v2(ac[::-1],ah[::-1],ag)==r.quantized_hash_v2(ac,ah,ag)

@pytest.mark.parametrize('value',[.0275,-.0275,.0175,-.0175])
def test_half_boundaries_do_not_follow_float_readback_noise(value):
    c,h=r.registration.geometry(FIXTURE['cases'][5]['plan']['config']);g=FIXTURE['cases'][5]['plan']['config']['goal_xyz']
    c[2,1]=value;expected=r.quantized_hash_v2(c,h,g)
    for delta in (-1e-8,1e-8):
        x=c.copy();x[2,1]=value+delta;assert r.quantized_hash_v2(x,h,g)==expected

def test_physical_readback_threshold_unchanged():
    p=FIXTURE['cases'][5]['plan'];c,h=r.registration.geometry(p['config']);c[0,0]+=1.01e-6
    with pytest.raises(ValueError,match='geometry differs'):r.validate_initial_geometry(p,c,h,p['config']['goal_xyz'])

def test_shapes_and_nonfinite_fail_closed():
    p=FIXTURE['cases'][5]['plan'];c,h=r.registration.geometry(p['config']);g=p['config']['goal_xyz']
    with pytest.raises(ValueError):r.quantized_hash_v2(c[:,:2],h,g)
    c[0,0]=np.nan
    with pytest.raises(ValueError):r.quantized_hash_v2(c,h,g)

def original_metadata():
    p=r.policy();v={'parent_plan':[copy.deepcopy(x['plan']) for x in FIXTURE['cases']]}
    rows=[dict(index=i,worker_elapsed_unknown=False,worker_elapsed_seconds=0.) for i in range(12)]
    rows[0]['worker_elapsed_seconds']=p['inherited_worker_seconds'];summaries={};attempts={}
    for i in (5,7):
        name=v['parent_plan'][i]['parent_id']
        summaries[i]=dict(initialization=dict(passed=False,error="ValueError('Actual geometry 1mm registration mismatch')"),route_attempts=0,unattempted_route_proposals=27)
        attempts[i]=[dict(input_id=name+'_target%d'%t,attempt=k,attempted=False,success=False,reason='initialization_gate_closed') for t in range(3) for k in range(9)]
    return v,rows,summaries,attempts,p

def test_only_proven_unattempted_failures_and_exact_inherited_cost():
    v,rows,s,a,p=original_metadata();assert r.validate_original_metadata(v,rows,s,a,p)==1882.7102640580852
    for change in ('attempt','cost','success','partial'):
        rr,ss,aa=copy.deepcopy(rows),copy.deepcopy(s),copy.deepcopy(a)
        if change=='attempt':aa[5][0]['attempted']=True
        elif change=='cost':rr[0]['worker_elapsed_seconds']-=1
        elif change=='success':ss[5]['initialization']['passed']=True
        else:aa[5].pop()
        with pytest.raises(ValueError):r.validate_original_metadata(v,rr,ss,aa,p)

def test_registration_keeps_original_parent_configuration_and_role():
    v,rows,s,a,p=original_metadata();new=r.recovered_registration(v,p,p['inherited_worker_seconds'])
    assert [x['index'] for x in new['parent_plan']]==[5,7]
    assert new['parent_plan']==[v['parent_plan'][5],v['parent_plan'][7]]
    assert new['requested_routes']==54 and new['combined_historical_proposal_slots']==378
    assert new['recovery_lineage']['remaining_worker_soft_seconds_at_start']==2700-p['inherited_worker_seconds']
    lineages=[x for plan in new['parent_plan'] for x in r.lineage_rows(plan)]
    assert len({x['new_attempt_id'] for x in lineages})==54
    assert all(x['new_attempt_id']!=x['original_unattempted_slot'] and x['role']=='TRAIN' for x in lineages)

def test_inherited_budget_never_resets_and_unknown_cost_rejected():
    assert r.cumulative_worker_seconds(1882.7102640580852,[dict(worker_elapsed_unknown=False,worker_elapsed_seconds=500.)])==2382.7102640580852
    with pytest.raises(RuntimeError):r.cumulative_worker_seconds(1882.,[dict(worker_elapsed_unknown=True,worker_elapsed_seconds=None)])
    with pytest.raises(ValueError):r.cumulative_worker_seconds(1882.,[dict(worker_elapsed_unknown=False,worker_elapsed_seconds=float('nan'))])

def test_scoped_validator_restored_even_on_error(tmp_path):
    old=r.legacy.validate_initial_geometry;protocol=r.legacy.PROTOCOL
    case=FIXTURE['cases'][5];p=case['plan'];c,h,g=(np.asarray(case[k]) for k in ('actual_centers','actual_halves','actual_goals'))
    originals=[a.copy() for a in (c,h,g)]
    with pytest.raises(RuntimeError,match='intentional'):
        with r.scoped_validator(tmp_path):
            assert r.legacy.validate_initial_geometry(p,c,h,g)==p['registered_geometry_1mm_sha256']
            raise RuntimeError('intentional')
    assert r.legacy.validate_initial_geometry is old and r.legacy.PROTOCOL==protocol
    assert all(np.array_equal(a,b) for a,b in zip(originals,(c,h,g)))
    receipt=json.loads((tmp_path/'geometry_identity_audit.json').read_text())
    assert receipt['legacy_actual_geometry_1mm_sha256']!=receipt['actual_geometry_1mm_v2_sha256']
    assert receipt['actual_arrays_modified'] is False

def mock_stage(monkeypatch,tmp_path,inherited=1882.7102640580852):
    output=tmp_path/'corpus';run=tmp_path/'run';p=r.policy();p['new_run_root']=str(run)
    v={'parent_plan':[copy.deepcopy(FIXTURE['cases'][i]['plan']) for i in (5,7)],'recovery_lineage':{'inherited_worker_seconds':inherited}}
    monkeypatch.setattr(r,'verify_corpus',lambda _: (v,{}));monkeypatch.setattr(r,'policy',lambda:p)
    monkeypatch.setattr(r.legacy.bounded,'budget_status',lambda *a:None)
    return output,run,v

def test_completed_recovery_never_launches_worker(monkeypatch,tmp_path):
    output,run,v=mock_stage(monkeypatch,tmp_path)
    rows=[dict(index=i,worker_elapsed_unknown=False,worker_elapsed_seconds=10,attempted_lower=27,attempted_upper=27) for i in (5,7)]
    monkeypatch.setattr(r,'checked_closures',lambda *a:rows)
    with mock.patch.object(r.subprocess,'run',side_effect=AssertionError('No child')):
        r.run_recovery(output,run,Path('unused'))
    assert r.read(run/'recovery/complete.json')['cumulative_worker_seconds']==1902.7102640580852

def test_cumulative_cap_stops_before_any_new_worker(monkeypatch,tmp_path):
    output,run,v=mock_stage(monkeypatch,tmp_path,inherited=2700)
    monkeypatch.setattr(r,'checked_closures',lambda *a:[])
    with mock.patch.object(r.subprocess,'run',side_effect=AssertionError('No child')):
        with pytest.raises(r.legacy.bounded.InternalBudgetPause):r.run_recovery(output,run,Path('unused'))

def test_interrupted_parent_is_closed_without_replay(monkeypatch,tmp_path):
    output,run,v=mock_stage(monkeypatch,tmp_path);name=v['parent_plan'][0]['parent_id']
    (output/'parents/TRAIN'/name).mkdir(parents=True)
    closed=[]
    monkeypatch.setattr(r,'checked_closures',lambda *a:copy.deepcopy(closed))
    def closure(*a):
        row=dict(index=5,worker_elapsed_unknown=True,worker_elapsed_seconds=None);closed.append(row);return row
    monkeypatch.setattr(r.legacy.old,'mechanical_closure',closure)
    with mock.patch.object(r.subprocess,'run',side_effect=AssertionError('Interrupted slot replayed')):
        with pytest.raises(RuntimeError,match='Unknown interrupted cost'):r.run_recovery(output,run,Path('unused'))
    assert r.read(output/'closures/005.json')['worker_elapsed_unknown']

def test_worker_rejects_successful_original_parent(monkeypatch,tmp_path):
    v={'parent_plan':[FIXTURE['cases'][i]['plan'] for i in (5,7)]}
    monkeypatch.setattr(r,'verify_corpus',lambda _:(v,{}))
    with pytest.raises(ValueError,match='Only failed'):r.worker(tmp_path,4,tmp_path/'somewhere')

def test_no_certify_stage_or_old_source_mutation():
    with pytest.raises(SystemExit):r.main(['certify'])
    source=Path(r.legacy.__file__).read_text()
    assert 'digest=registration.physical_hash(centers,halves,goals,.001)' in source
