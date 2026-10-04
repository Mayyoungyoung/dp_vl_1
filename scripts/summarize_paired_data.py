"""Finite-reference data summary and actual pair identity, scoped to new data."""
import argparse
from collections import Counter
import numpy as np
from scripts.paired_modes_data import DATA,RUN
from scripts.run_observed_probability import read,write,sha


def summarize():
    registration=read(DATA/'registration.json');plans=registration['parent_plan'];families={};stats={}
    for plan in plans:
        folder=DATA/'parents'/plan['role']/plan['parent_id'];s=read(folder/'summary.json')
        if s['status']=='error':raise RuntimeError('Unresolved worker failure: '+plan['parent_id'])
        key=plan['role']+'/'+plan['variant'];counts=stats.setdefault(key,Counter())
        counts.update(scenes=1,requested_slots=27,attempted=s['route_attempts'],accepted=s['accepted_routes'],initialization_passed=int(s['initialization']['passed']))
        if not s['initialization']['passed']:continue
        d=dict(np.load(folder/plan['parent_id']/'observation.npz'))
        truth=dict(np.load(folder/plan['parent_id']/'verification_only.npz'))
        record=dict(observation=d,targets=truth['target_centers'],colors=s['initialization']['actual_target_colors'])
        families.setdefault(plan['family_id'],[]).append(record)
    checked=0
    for family,items in families.items():
        if len(items)!=3:continue
        first=items[0]
        for item in items[1:]:
            for key in ('gripper_pose','gripper_open','camera_intrinsics','camera_extrinsics'):
                np.testing.assert_array_equal(item['observation'][key],first['observation'][key],err_msg=family+'/'+key)
            np.testing.assert_array_equal(item['targets'],first['targets'],err_msg=family+'/targets')
            assert item['colors']==first['colors'],family+'/colors'
        checked+=1
    size=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    assert size<registration['policy']['new_data_budget_bytes']
    report=dict(registration_sha256=sha(DATA/'registration.json'),stats={k:dict(v) for k,v in stats.items()},
        pair_identity_families_passed=checked,requested_families=len(plans)//3,new_data_bytes=size,
        reference_kind=registration['reference_kind'],robot_execution_traces=0,
        all_requested_scenes=len(plans),all_requested_slots=len(plans)*27,
        accepted_geometric_paths=sum(v['accepted'] for v in stats.values()),
        mode_frequency_claim=False,feasible_set_exhaustive=False)
    write(RUN/'data_summary.json',report);print(__import__('json').dumps(report),flush=True)


if __name__=='__main__':summarize()
