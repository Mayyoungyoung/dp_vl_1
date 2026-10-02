import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import collect_two_row_layout4 as batch

ROOT=Path(__file__).resolve().parents[1]


def fixtures():
    return (json.loads((ROOT/'configs/observed_two_row_layout4_v5.json').read_text()),
            json.loads((ROOT/'configs/observed_two_row_pilot_v4.json').read_text()))


def test_four_registered_physical_layouts_are_distinct_and_budgeted_once():
    reg,base=fixtures();plans=batch.build_plan(reg,base)
    assert plans==batch.build_plan(reg,base)
    assert len(plans)==4 and sum(p['config']['requested_route_proposals'] for p in plans)==108
    assert len({p['registered_geometry_1mm_sha256'] for p in plans})==4
    for p in plans:
        c=p['config']
        assert p['geometry_precheck']['passed'] and p['geometry_precheck']['checked']==27
        assert c['selected_target_indices']==[0,1,2] and c['split']=='DEV_COLLECTION'
        assert c['post_size_xyz']==[.035,.035,.14]
        assert c['corridor_guide_y']==base['corridor_guide_y'] and c['route_height']==base['route_height']
        assert all(lo<=x<=hi for x,(lo,hi) in zip(c['row_x'],reg['row_x_ranges']))
        assert all(lo<=y<=hi for ys,ranges in zip(c['post_y'],reg['post_y_ranges']) for y,(lo,hi) in zip(ys,ranges))
        assert c['preparation_xyz'][1]==c['entry_xyz']


def test_geometry_hash_does_not_count_color_seed_or_setup_fk_drift_as_independence():
    reg,base=fixtures();plan=batch.build_plan(reg,base)[0];c=plan['config']
    centers,halves=batch.collector.geometry(c);goals=c['goal_xyz']
    exact=batch.scene_geometry_hash(centers,halves,goals,.001)
    assert batch.physical_layout_hash(centers,halves,goals,[0,0,.865],.001)!=batch.physical_layout_hash(centers,halves,goals,[.005,0,.865],.001)
    assert exact==batch.scene_geometry_hash(centers,halves,goals,.001)
    rows=[dict(parent_id='a',geometry=exact),dict(parent_id='b',geometry=exact)]
    assert batch.duplicate_groups(rows,'geometry')==[['a','b']]
    changed=centers.copy();changed[0,0]+=.005
    assert exact!=batch.scene_geometry_hash(changed,halves,goals,.001)


def test_interrupted_parent_retains_denominator_and_never_claims_exact_inflight_count(tmp_path):
    reg,base=fixtures();plan=batch.build_plan(reg,base)[0]
    records=[dict(input_id=plan['parent_id']+'_target0',attempt=i,success=False) for i in range(2)]
    (tmp_path/'attempts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    row=batch.parent_evidence(tmp_path,plan,None,None,.001)
    assert row['status']=='interrupted_no_retry' and (row['attempted_lower'],row['attempted_upper'])==(2,3)
    assert (row['unattempted_lower'],row['unattempted_upper'])==(24,25)
    assert row['worker_seconds'] is None


def test_closed_setup_failure_accounts_all_27_unattempted(tmp_path):
    reg,base=fixtures();plan=batch.build_plan(reg,base)[0]
    batch.write(tmp_path/'summary.json',dict(status='setup_failed',requested_route_proposals=27,route_attempts=0))
    row=batch.parent_evidence(tmp_path,plan,2.,0,.001)
    assert row['status']=='setup_failed' and row['attempted_upper']==0 and row['unattempted_lower']==27


def test_precheck_failure_is_not_resampled_or_replaced(tmp_path):
    reg,base=fixtures();plan=batch.build_plan(reg,base)[0]
    plan['geometry_precheck']=dict(passed=False,error='fixture')
    row=batch.parent_evidence(tmp_path,plan,0.,None,.001)
    assert row['status']=='geometry_precheck_failed' and row['unattempted_lower']==27


def test_resume_rejects_changed_registration_before_any_worker(tmp_path):
    reg,base=fixtures();regpath=tmp_path/'reg.json';basepath=tmp_path/'base.json'
    batch.write(regpath,reg);batch.write(basepath,base)
    out=tmp_path/'out';out.mkdir();batch.write(out/'registration.json',{'different':'source'})
    with pytest.raises(ValueError,match='identical'):
        batch.run(regpath,basepath,out,tmp_path/'runs',True)


def test_duplicate_attempt_ledger_is_rejected(tmp_path):
    reg,base=fixtures();plan=batch.build_plan(reg,base)[0]
    row=dict(input_id='same',attempt=0)
    (tmp_path/'attempts.jsonl').write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='duplicate route'):
        batch.parent_evidence(tmp_path,plan,1.,1,.001)


def test_closed_failed_parents_resume_without_worker_reexecution(tmp_path,monkeypatch):
    from types import SimpleNamespace
    reg,base=fixtures();regpath=tmp_path/'reg.json';basepath=tmp_path/'base.json'
    batch.write(regpath,reg);batch.write(basepath,base);calls=[]
    def worker(command,check):
        calls.append(command)
        data=Path(command[-1]);data.mkdir(parents=True)
        batch.write(data/'summary.json',dict(status='setup_failed',requested_route_proposals=27,route_attempts=0))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess,'run',worker)
    out=tmp_path/'out';runs=tmp_path/'runs'
    batch.run(regpath,basepath,out,runs)
    assert len(calls)==4
    batch.run(regpath,basepath,out,runs,True)
    assert len(calls)==4
    summary=json.loads((out/'summary.json').read_text())
    assert summary['closed_parents']==4 and summary['requested_routes']==108
    assert summary['unattempted_routes_lower']==summary['unattempted_routes_upper']==108
    assert summary['accepted_references']==0 and not summary['training_authorized']


@pytest.mark.parametrize('protocol,selection,budget',[
    (batch.collector.LOWER_POST_PROTOCOL,[0,1,2],27),
    (batch.collector.LAYOUT_PROTOCOL,[1],9),
    (batch.collector.LOWER_POST_PROTOCOL,None,9),
    (batch.collector.LAYOUT_PROTOCOL,None,27)])
def test_explicit_v4_v5_target_gate_rejects_wrong_selection_even_with_matching_budget(protocol,selection,budget):
    _,config=fixtures();config['protocol']=protocol;config['requested_route_proposals']=budget
    if selection is None:config.pop('selected_target_indices')
    else:config['selected_target_indices']=selection
    with pytest.raises(ValueError,match='target'):
        batch.collector.validate_config(config)


@pytest.mark.parametrize('phase',['starting','running'])
def test_live_worker_is_not_marked_interrupted_or_restarted(tmp_path,monkeypatch,phase):
    reg,base=fixtures();regpath=tmp_path/'reg.json';basepath=tmp_path/'base.json'
    batch.write(regpath,reg);batch.write(basepath,base);runs=tmp_path/'runs'
    batch.write(runs/'parent_283100.status.json',dict(status=phase,pid=123))
    monkeypatch.setattr(batch.os,'kill',lambda pid,signal:None)
    def forbidden(*args,**kwargs):raise AssertionError('duplicate worker start')
    monkeypatch.setattr(batch.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='still running'):
        batch.run(regpath,basepath,tmp_path/'out',runs)
    assert not (tmp_path/'out'/'closures'/'283100.json').exists()
