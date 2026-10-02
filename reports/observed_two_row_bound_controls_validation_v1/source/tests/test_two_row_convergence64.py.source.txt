import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_observed_two_row_convergence64 as convergence

ROOT=Path(__file__).resolve().parents[1]


def policy():return json.loads((ROOT/'configs/observed_two_row_convergence64_v1.json').read_text())


def checkpoint():
    return dict(model={'weight':np.array([1.,2.],dtype=np.float32)},optimizer={'state':{'step':2}},scheduler={'last_epoch':2},
        scaler=None,rng={'numpy':np.array([1,2]),'python':(1,None)},sampler_state={'state':42},
        sample_stream_audit={'index_chain_sha256':'exact'},step=1500,trajectory_exposures=192000,best=1.,
        history=[dict(step=1500,loss=.01)],config={key:'frozen' for key in convergence.CONFIG_FIELDS},elapsed_s=100.)


def metadata():return {key:'fixed' for key in convergence.POLICY_FIELDS}


def test_predeclared_prefix76_budget_not_original1500_budget():
    p=policy();convergence.validate_policy(p)
    assert p['total_steps']*p['batch_size']*p['candidates']==1536000
    assert p['stage1500_steps']==p['original_steps']==1500
    for key,value in [('total_steps',8000),('registered_train_parents',32),('lr',.001),('data','different')]:
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):convergence.validate_policy(changed)


def test_all_budget_identities_and_approximate_exposure_comparison():
    p=policy();b=convergence.budget_receipt(p)
    assert (b['new_run_reproduction_steps'],b['new_run_postproof_steps'],b['actual_total_steps'])==(1500,10500,12000)
    assert (b['new_run_reproduction_observation_draws'],b['new_run_postproof_observation_draws'])==(48000,336000)
    assert (b['new_run_reproduction_path_states'],b['new_run_postproof_path_states'])==(192000,1344000)
    for suffix,total in [('steps','actual_total_steps'),('observation_draws','actual_training_observation_draws'),
            ('path_states','actual_training_candidate_path_states')]:
        assert b['new_run_reproduction_'+suffix]+b['new_run_postproof_'+suffix]==b[total]
    assert b['actual_training_observation_draws']==384000
    assert b['actual_training_candidate_path_states']==1536000
    assert b['fixed_last_train_reserved_requests']==189 and b['fixed_last_train_reserved_path_states']==756
    assert b['dev_selection_opportunities']==48 and b['original_reference_dev_selection_opportunities']==6
    assert b['comparison_reference']['dev_selection_opportunities']==24
    assert b['per_actual_train_input_draws']==384000/189
    assert b['comparison_reference']['per_actual_train_input_draws']==192000/93
    assert b['per_input_draw_ratio_to_comparison']==pytest.approx(62/63)
    assert b['total_training_budget_ratio_to_comparison']==2
    assert b['original_reference_training_path_states']==192000 # preserved original, not charged to the new run


@pytest.mark.parametrize('field', ['new_training_observation_draws','new_training_candidate_path_states',
    'dev_selection_opportunities','actual_train_inputs','stage1500_steps'])
def test_rejects_corrupt_budget_claims(field):
    p=policy();p[field]+=1
    with pytest.raises(ValueError):convergence.budget_receipt(p)


def test_rejects_changed_comparison_denominators():
    p=policy();p['comparison_reference']['actual_train_inputs']=96
    with pytest.raises(ValueError,match='comparison'):convergence.budget_receipt(p)


def completed_budget_state():
    p=policy()
    result=dict(last_step=12000,trajectory_exposures=1536000,sample_stream_audit=dict(batches=12000,
        observation_draws=384000,initial_model_sha256=p['initialization_sha256']))
    checkpoint=dict(step=12000,trajectory_exposures=1536000,config=dict(steps=12000),
        history=[dict(step=i) for i in range(250,12001,250)])
    return p,result,checkpoint


def test_finished_budget_checks_actual_history_not_only_config():
    p,result,checkpoint=completed_budget_state()
    assert convergence.validate_completed_budget(p,result,checkpoint)==convergence.budget_receipt(p)
    checkpoint['history']=checkpoint['history'][:-1]
    with pytest.raises(ValueError,match='DEV selection'):convergence.validate_completed_budget(p,result,checkpoint)


@pytest.mark.parametrize('target,field,bad', [('result','last_step',6000),('result','trajectory_exposures',768000),
    ('checkpoint','step',6000),('checkpoint','trajectory_exposures',5712000),
    ('stream','batches',6000),('stream','observation_draws',192000),('stream','initial_model_sha256','wrong')])
def test_finished_budget_rejects_old_or_inconsistent_actual_counts(target,field,bad):
    p,result,checkpoint=completed_budget_state()
    obj={'result':result,'checkpoint':checkpoint,'stream':result['sample_stream_audit']}[target];obj[field]=bad
    with pytest.raises(ValueError):convergence.validate_completed_budget(p,result,checkpoint)


def test_exact_state_audit_ignores_only_nontraining_metadata():
    a=checkpoint();b=copy.deepcopy(a);b['elapsed_s']=200.;b['config'].update(steps=12000,output='fresh',code_commit='new')
    assert convergence.compare_training_state(a,b)['passed']
    for key in ('model','optimizer','scheduler','rng','sampler_state','sample_stream_audit','best','history'):
        changed=copy.deepcopy(b)
        if key=='model':changed[key]['weight'][0]+=.00001
        else:changed[key]='not equivalent'
        assert not convergence.compare_training_state(a,changed)['passed']


@pytest.mark.parametrize('key',convergence.STATE_FIELDS)
def test_missing_state_on_both_sides_never_counts_as_equal(key):
    a=checkpoint();b=copy.deepcopy(a);del a[key];del b[key]
    assert not convergence.compare_training_state(a,b)['passed']


def test_missing_training_identity_on_both_sides_rejected():
    a=checkpoint();b=copy.deepcopy(a);del a['config']['train_examples'];del b['config']['train_examples']
    assert not convergence.compare_training_state(a,b)['passed']


def test_scoped_schedule_restores_historical_entry_on_failure():
    calls=[]
    def original(args):calls.append(vars(args));raise RuntimeError('unit interruption')
    ordinary=SimpleNamespace(base=SimpleNamespace(train=original))
    args=SimpleNamespace(steps=1500,resume=False,output='fresh')
    with pytest.raises(RuntimeError):
        with convergence.schedule_adapter(ordinary,metadata()):ordinary.base.train(args)
    assert ordinary.base.train is original and args.steps==1500
    assert calls[0]['steps']==12000 and calls[0]['convergence_selection_budget_steps']==1500


def test_resume_rejects_cross_policy_source_and_original1500_checkpoint():
    config=dict(metadata(),steps=12000,convergence_selection_budget_steps=1500,convergence_actual_total_steps=12000,
        convergence_lr_policy='unchanged constant LambdaLR=1');convergence.validate_resume_policy(config,dict(config))
    for key in convergence.POLICY_FIELDS:
        changed=dict(config);changed[key]='different'
        with pytest.raises(ValueError):convergence.validate_resume_policy(config,changed)
    with pytest.raises(ValueError):convergence.validate_resume_policy(config,dict(config,steps=1500))
    with pytest.raises(ValueError):convergence.validate_resume_policy(config,{'steps':12000})
    changed=dict(config);del changed['convergence_actual_total_steps']
    with pytest.raises(ValueError,match='explicit convergence schedule'):convergence.validate_resume_policy(config,changed)


def test_stage_boundaries_no_automatic_finish_or_reproduction_replay(tmp_path):
    convergence.validate_stage('stage1500',False,tmp_path)
    with pytest.raises(ValueError):convergence.validate_stage('finish',False,tmp_path,1000)
    convergence.validate_stage('stage1500',True,tmp_path,1500) # metadata-only sealing recovery
    convergence.validate_stage('finish',False,tmp_path,1500)
    with pytest.raises(ValueError):convergence.validate_stage('finish',False,tmp_path,3000)
    convergence.validate_stage('finish',True,tmp_path,3000)
    with pytest.raises(ValueError):convergence.validate_stage('finish',True,tmp_path,12001)


def test_snapshot_bound_to_passing_proof_and_policy(tmp_path):
    folder=tmp_path/'peak_seed0';folder.mkdir()
    for name in ('last.pt','best.pt','config.json','history.json','status.json','source_hashes.json'):(folder/name).write_bytes(b'fixed test bytes')
    convergence.snapshot1500(tmp_path)
    p=policy();audit=dict(passed=True,reference_sha256=p['reference_last_sha256'],reproduced_sha256=convergence.digest(folder/'last.pt'),metadata=metadata(),step=1500)
    convergence.write(tmp_path/'stage1500_audit.json',audit)
    convergence.verify_stage1500_gate(tmp_path,p,metadata())
    convergence.snapshot1500(tmp_path) # identical metadata-only reseal is idempotent
    (tmp_path/'stage1500_snapshot/last.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='snapshot changed'):convergence.verify_stage1500_gate(tmp_path,p,metadata())


def test_partial_snapshot_recovers_missing_files_without_replacing_existing(tmp_path):
    model=tmp_path/'peak_seed0';model.mkdir();snapshot=tmp_path/'stage1500_snapshot';snapshot.mkdir()
    for name in ('last.pt','best.pt','config.json','history.json','status.json','source_hashes.json'):(model/name).write_bytes(name.encode())
    (snapshot/'last.pt').write_bytes((model/'last.pt').read_bytes())
    convergence.snapshot1500(tmp_path);assert (snapshot/'best.pt').exists()
    (snapshot/'last.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Never replace'):convergence.snapshot1500(tmp_path)


def test_completed_diagnostic_metadata_resume_never_regenerates_pool(tmp_path):
    output=tmp_path/'fixed_last_train';staging=tmp_path/'fixed_last_train.staging';staging.mkdir()
    (staging/'predictions.npz').write_bytes(b'sealed original predictions')
    receipt=dict(checkpoint_sha256='fixed-model',source_export_manifest_sha256='fixed-data',
        artifact_sha256={'predictions.npz':convergence.digest(staging/'predictions.npz')},new_forward_requests=189)
    convergence.write(staging/'diagnostic_receipt.json',receipt)
    assert convergence.recover_sealed_diagnostic(output,'fixed-model','fixed-data')==receipt
    assert output.exists() and not staging.exists()
    assert convergence.recover_sealed_diagnostic(output,'fixed-model','fixed-data')==receipt
    (output/'predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='pool SHA changed'):convergence.recover_sealed_diagnostic(output,'fixed-model','fixed-data')


def test_unsealed_diagnostic_preserves_budget_and_refuses_automatic_replay(tmp_path):
    output=tmp_path/'fixed_last_train';staging=tmp_path/'fixed_last_train.staging';staging.mkdir()
    (staging/'request_receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='No automatic new forward'):convergence.recover_sealed_diagnostic(output,'model','data')
    assert staging.exists()


def test_final_checkpoint_without_ordinary_receipt_does_not_replay_final_forward(tmp_path):
    (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='No automatic final forward'):
        convergence.verify_ordinary_completion(tmp_path,{},lambda _:pytest.fail('Do not touch pools before the seal exists'))


def test_ordinary_completion_requires_all_original_pool_and_identity_hashes(tmp_path):
    (tmp_path/'summary.json').write_text('{}')
    config=dict(two_row_driver_protocol='ordinary',two_row_driver_sha256='source',two_row_export_sha256='export',two_row_quality_audit_sha256='quality')
    pool={'train/predictions.npz':dict(path='fixed',sha256='unchanged',bytes=42)}
    receipt=dict(protocol='ordinary',source_sha256='source',export_sha256='export',quality_audit_sha256='quality',
        actual_training_summary_sha256=convergence.digest(tmp_path/'summary.json'),training_candidate_path_states=1536000,prediction_artifacts=pool)
    convergence.write(tmp_path/'two_row_driver_receipt.json',receipt)
    assert convergence.verify_ordinary_completion(tmp_path,config,lambda _:pool)==receipt
    with pytest.raises(ValueError,match='pool seal differs'):
        convergence.verify_ordinary_completion(tmp_path,config,lambda _:{})
    with pytest.raises(ValueError,match='identity differs'):
        convergence.verify_ordinary_completion(tmp_path,{},lambda _:pool)


def test_actual_constant_lr_loop_short_prefix_and_resume_exact(tmp_path,monkeypatch):
    """Real tiny CPU loop: the production data gate is not exercised here."""
    torch=pytest.importorskip('torch')
    from scripts import train_observed_two_row as ordinary
    from test_two_row_observation_training import fixture
    data,geometry,_,sources,labels=fixture(tmp_path,monkeypatch)
    train_ids=np.array(['t_target0','t_target1','t_target2'])
    data={key:(np.concatenate([value,value]) if isinstance(value,np.ndarray) else value*2) for key,value in data.items()}
    data['scene_ids'][:3]=train_ids;data['parent_ids'][:3]='t';data['splits']=np.array(['TRAIN']*3+['DEV_MODEL']*3)
    data.update(tasks=['reach']*6,source_hashes={},cache_config={},skipped=[],unreferenced=[],evaluation_protocol='observation_eval_v2')
    train_labels=[dict(row,id=str(train_ids[i]),parent_id='t',split='TRAIN') for i,row in enumerate(labels)]
    sources[1].write_text(''.join(json.dumps(r)+'\n' for r in train_labels+labels))
    geometry.update(index=np.zeros(6,dtype=int),fingerprint='fixture',metadata={'source_hashes':{}})
    monkeypatch.setattr(ordinary.base,'load_observed_dataset',lambda *a,**kw:data)
    monkeypatch.setattr(ordinary.base,'load_geometry',lambda *a,**kw:geometry)
    monkeypatch.setattr(ordinary.base,'measure_latency',lambda *a,**kw:{})
    values=dict(observations=str(sources[0]),supervision=str(sources[1]),cache_dir='fixture',output=str(tmp_path/'original'),
        steps=2,batch_size=2,candidates=2,horizon=2,width=16,depth=1,pooling='both',geometry_pooling='spatial',
        anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',sampling_mode='uniform',metric_aggregation='instruction',
        refinement_mode='none',refinement_sigma=None,refinement_prefix_fraction=None,refinement_bound=None,
        checkpoint_selection='tip_unique_valid',point_width=8,pixel_stride=2,endpoint_residual_bound=.05,
        grounding_weight=.02,grounding_sigma=.025,grounding_target='endpoint',event_scale=.2,lr=3e-4,
        seed=0,eval_every=2,threads=1,device='cpu',resume=False,stop_after=None,sample_stream_audit=True)
    values.update({key:'fixture' for key in ('two_row_driver_sha256','two_row_export_sha256','two_row_quality_audit_sha256','two_row_selection_sha256')})
    with ordinary.evaluation_adapter(sources):ordinary.base.train(SimpleNamespace(**values))
    original=torch.load(tmp_path/'original/last.pt',weights_only=False)
    values.update(steps=1500,output=str(tmp_path/'split'),stop_after=2)
    with convergence.schedule_adapter(ordinary,metadata(),total_steps=4),ordinary.evaluation_adapter(sources):ordinary.base.train(SimpleNamespace(**values))
    prefix=torch.load(tmp_path/'split/last.pt',weights_only=False)
    assert convergence.compare_training_state(original,prefix)['passed']
    values.update(resume=True,stop_after=None)
    with convergence.schedule_adapter(ordinary,metadata(),total_steps=4),ordinary.evaluation_adapter(sources):ordinary.base.train(SimpleNamespace(**values))
    resumed=torch.load(tmp_path/'split/last.pt',weights_only=False)
    values.update(resume=False,output=str(tmp_path/'full'))
    with convergence.schedule_adapter(ordinary,metadata(),total_steps=4),ordinary.evaluation_adapter(sources):ordinary.base.train(SimpleNamespace(**values))
    full=torch.load(tmp_path/'full/last.pt',weights_only=False)
    for key in convergence.STATE_FIELDS:
        if key!='history':assert not convergence.state_differences(full[key],resumed[key],key)
    assert resumed['config']['steps']==4 and resumed['config']['convergence_selection_budget_steps']==1500
