import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_two_row_route_conditioning as audit


def rows():
    return [dict(id=i,parent_id=i.rsplit('_target',1)[0],split='TRAIN',image='/'+i+'/front.png',instruction='test') for i in audit.IDS]


def test_exact_selection_ignores_nonselected_payload(tmp_path):
    path=tmp_path/'rows.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows())+'\n'+
        '{"id":"DEV_must_not_decode","payload": deliberately invalid json}\n')
    assert [r['id'] for r in audit.selected_rows(path)]==audit.IDS


@pytest.mark.parametrize('mutation',['missing','duplicate','role','parent'])
def test_selection_fail_closed(tmp_path,mutation):
    value=rows()
    if mutation=='missing':value.pop()
    elif mutation=='duplicate':value.append(value[0])
    elif mutation=='role':value[0]['split']='DEV_MODEL'
    else:value[0]['parent_id']='other'
    path=tmp_path/'rows.jsonl';path.write_text('\n'.join(json.dumps(r) for r in value))
    with pytest.raises(ValueError):audit.selected_rows(path)


def test_donor_is_fixed_within_parent_cycle():
    for parent in audit.PARENTS:
        ids=[parent+'_target%d'%t for t in range(3)]
        assert [audit.donor_id(i) for i in ids]==ids[1:]+ids[:1]
    with pytest.raises(ValueError):audit.donor_id('locked')


def test_endpoint_clamp_and_prefix_are_not_index_half():
    normal=np.zeros((4,24,3));normal[:,:,0]=np.r_[np.linspace(0,.1,20),.4,.7,.9,1.]
    changed=normal+.2;fixed=audit.clamp_endpoint(changed,normal)
    assert np.array_equal(fixed[:,-1],normal[:,-1])
    assert np.array_equal(fixed[:,1:-1],changed[:,1:-1])
    record=audit.displacement(normal,fixed)[0]
    assert 20 in record['prefix_vertex_indices']
    assert 21 not in record['prefix_vertex_indices']
    assert len(record['per_vertex_distance_m'])==24
    assert record['fixed_endpoint_distance_m']==0


def test_no_search_guard_restores_after_error():
    a=lambda:1;b=lambda:2
    planner=SimpleNamespace(astar_virtual=a,v1=SimpleNamespace(astar=b))
    with pytest.raises(RuntimeError):
        with audit.forbid_search(planner):planner.astar_virtual()
    assert planner.astar_virtual is a and planner.v1.astar is b


def test_reference_support_preserves_reference_and_predicted_goal():
    path=np.array([[0.,0.,0.],[1.,0.,0.]]);seen=[]
    planner=SimpleNamespace(v1=SimpleNamespace(path_observed_clear=lambda p,f,l:(False,3)),
        visible_proxy=lambda p,n,r:(seen.append((p.copy(),r['endpoint'].copy())) or dict(passed=True)))
    goal=np.array([2.,0.,0.]);graph=dict(free=np.ones((2,2,2)),lower=np.zeros(3),endpoint=goal,
        nonselected=np.empty((0,3)),rays=dict(endpoint=goal),attachments_available=False)
    result=audit.reference_support(planner,path,graph)
    assert not result['supported_by_both_existing_proxies']
    assert result['reference_end_to_predicted_goal_m']==1
    assert np.array_equal(seen[0][0],path) and np.array_equal(seen[0][1],goal)
    assert audit.reference_support(planner,path,None)['supported'] is None


def test_registered_config():
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/two_row_route_conditioning_v1.json').read_text())
    assert config['ids']==audit.IDS and config['arms']==list(audit.ARMS)
    assert config['max_forward_requests']==36 and config['max_complete_path_states']==144
    assert config['checkpoint_sha256']=='cd1af0b572b80f9e7002ce866b7f3f062fed5afd6ab45ac16290db0c8b67c880'
    assert set(config['cache_sha256'])==set(audit.IDS)
    assert config['search_calls']==config['new_qwen_encodings']==0


def test_original_train_pool_reproduction_and_rejection(tmp_path):
    ids=[('two_row_reach_%d_target%d'%(s,t)) for s in range(283200,283264) if s!=283220 for t in range(3)]
    xyz=np.zeros((189,4,24,3),np.float32);opened=np.ones((189,4,24),np.float32)
    file=tmp_path/'predictions.npz';np.savez(file,scene_ids=ids,paths=xyz,gripper_open=opened)
    pools={('normal',i):(xyz[0].copy(),opened[0].copy()) for i in audit.IDS}
    assert audit.compare_saved_normal(pools,file)['passed']
    pools[('normal',audit.IDS[0])][0][0,1,0]=.001
    with pytest.raises(ValueError):audit.compare_saved_normal(pools,file)


def test_torch_real_hook_identity_and_exception_cleanup():
    torch=pytest.importorskip('torch')
    module=torch.nn.Linear(3,4).eval();x=torch.ones(1,3)
    original=module(x).detach();before=audit.state_digest(module)
    count=len(module._forward_hooks)
    with audit.direct_override(module,original,[]) as calls:
        assert torch.equal(module(x),original)
    assert calls==[1] and len(module._forward_hooks)==count
    with pytest.raises(RuntimeError):
        with audit.direct_override(module,original,[]):raise RuntimeError('injected')
    assert len(module._forward_hooks)==count and audit.state_digest(module)==before


def test_torch_complete_36_call_probe(tmp_path):
    torch=pytest.importorskip('torch')
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.head=torch.nn.Module();self.head.feature_encoder=torch.nn.Linear(4,3)
            self.calls=0
        def forward(self,features,current):
            self.calls+=1
            direct=self.head.feature_encoder(features)
            paths=(current[:,:3]+direct)[:,None,None].expand(-1,4,24,-1).clone()
            paths[:,:,0]=current[:,:3,None].transpose(1,2)
            opened=torch.ones((1,4,24))
            return paths,opened,dict(anchor_xyz=current[:,:3].clone(),context=features.clone(),attention=features.clone())
    torch.manual_seed(9);model=Toy().eval()
    for p in model.parameters():p.requires_grad_(False)
    inputs={identifier:dict(batch=dict(features=torch.full((1,4),float(i)),current=torch.zeros(1,8)),current=np.zeros(8))
        for i,identifier in enumerate(audit.IDS)}
    pools,records=audit.generate_probe(model,inputs,tmp_path)
    assert model.calls==36 and len(records)==36 and len(pools)==36
    assert len(model.head.feature_encoder._forward_hooks)==0
    for identifier in audit.IDS:
        normal=pools[('normal',identifier)][0];changed=pools[('cyclic_direct_swap',identifier)][0]
        assert np.array_equal(normal,pools[('identity',identifier)][0])
        assert np.array_equal(normal[:,-1],changed[:,-1])
        assert not np.array_equal(normal[:,1:-1],changed[:,1:-1])
    assert json.loads((tmp_path/'model_state_guard.json').read_text())['exact_unchanged']
    assert json.loads((tmp_path/'probe_generation_seal.json').read_text())['labels_opened'] is False


def test_torch_geometry_change_rejected():
    torch=pytest.importorskip('torch')
    with pytest.raises(ValueError):audit.exact_geometry({'anchor':torch.zeros(3)},{'anchor':torch.ones(3)})
