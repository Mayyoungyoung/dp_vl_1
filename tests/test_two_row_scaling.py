import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_observed_two_row_scaling as runner

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def selection(count):
    return json.loads((ROOT/f'configs/observed_two_row_prefix{count+12}_selection_v1.json').read_text())


def test_scaling_paths_do_not_accept_arbitrary_parent_counts():
    for count in (32,64):
        paths=runner.stage_paths('/project','/source',count)
        assert f'prefix{count+12}' in str(paths['data'])
        assert paths['training']!=paths['preparation']
    for count in (0,16,31,48,65,True):
        with pytest.raises(ValueError):runner.stage_paths('/project','/source',count)


def test_fixed_initial_model_and_equal_exposure_audit_not_same_draw_stream():
    summary=dict(last_step=1500,trajectory_exposures=192000,sample_stream_audit=dict(
        batches=1500,observation_draws=48000,initial_model_sha256=runner.EXPECTED_INITIAL_MODEL_SHA256,index_chain_sha256='different-N'))
    assert runner.training_audit(summary)['passed']
    for field,value in [('last_step',1499),('trajectory_exposures',192004)]:
        changed=copy.deepcopy(summary);changed[field]=value;assert not runner.training_audit(changed)['passed']
    changed=copy.deepcopy(summary);changed['sample_stream_audit']['initial_model_sha256']='changed'
    assert not runner.training_audit(changed)['passed']


@pytest.mark.parametrize('count',[32,64])
def test_prepare_stage_never_launches_cache_or_train(tmp_path,monkeypatch,count):
    source=tmp_path/('a'*40);source.mkdir();monkeypatch.chdir(source)
    monkeypatch.setenv('CODE_COMMIT',source.name);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:{0},raising=False)
    paths=runner.stage_paths(tmp_path,source,count);write(paths['selection'],selection(count))
    monkeypatch.setattr(runner.exporter,'closed_prefix',lambda *args:(None,None,None,None,[]))
    monkeypatch.setattr(runner,'source_record',lambda *args:{})
    monkeypatch.setattr(runner,'verify_fixed_dev',lambda *args:{'conditions':36})
    calls=[]
    def record(project,output,run_id,environment,module,args,resume):
        calls.append((run_id,environment,module,args,resume))
        if run_id=='export':write(paths['data']/'export_manifest.json',{'test':True})
        if run_id=='train_quality':write(paths['preparation']/'train_quality.json',{'capacity_gate_passed':True})
    monkeypatch.setattr(runner,'run_record',record)
    runner.run(tmp_path,source,count,'prepare',source/'launcher')
    assert [r[0] for r in calls]==['export','train_quality']
    assert all(r[1]=='.venv/bin/python' and 'cuda' not in r[3] for r in calls)
    assert not paths['training'].exists() and not (paths['data']/'qwen_cache').exists()
    receipt=json.loads((paths['preparation']/'preparation_receipt.json').read_text())
    assert receipt['gpu_stage_started'] is False
    with pytest.raises(FileExistsError):runner.run(tmp_path,source,count,'prepare',source/'launcher')


def test_not_closed_prefix_refuses_before_creating_preparation(tmp_path,monkeypatch):
    source=tmp_path/('b'*40);source.mkdir();monkeypatch.chdir(source)
    monkeypatch.setenv('CODE_COMMIT',source.name);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:{0},raising=False)
    paths=runner.stage_paths(tmp_path,source,32);write(paths['selection'],selection(32))
    monkeypatch.setattr(runner.exporter,'closed_prefix',lambda *args:(None,None,None,None,['two_row_reach_283231']))
    with pytest.raises(ValueError,match='remains open'):runner.run(tmp_path,source,32,'prepare',source/'launcher')
    assert not paths['preparation'].exists() and not paths['data'].exists()


def dev_fixture(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new';rows=[];labels=[];hashes={}
    for index in range(64,76):
        parent=f'two_row_reach_{283200+index}'
        raw=tmp_path/'parents/DEV_MODEL'/parent;raw.mkdir(parents=True)
        files={}
        for name in ('image','observation','verification','config','route'):
            path=raw/name;path.write_text(name);files[name]=str(path);hashes[str(path)]=runner.exporter.digest(path)
        for target in range(3):
            identifier=parent+f'_target{target}'
            rows.append(dict(id=identifier,parent_id=parent,split='DEV_MODEL',image=files['image'],instruction='same actual language'))
            labels.append(dict(id=identifier,parent_id=parent,split='DEV_MODEL',observation=files['observation'],
                verification_only=files['verification'],route_config=files['config'],routes=[files['route']]))
    for root,count in ((old,16),(new,32)):
        root.mkdir();output={}
        for name,values in (('observations.jsonl',rows),('supervision.jsonl',labels)):
            (root/name).write_text(''.join(json.dumps(r)+'\n' for r in values));output[name]=runner.exporter.digest(root/name)
        write(root/'export_manifest.json',dict(selection=selection(count),output_files_sha256=output,source_files_sha256=hashes))
    return old,new,rows,labels


def test_dev_identity_requires_original_all36_language_labels_and_raw_hashes(tmp_path):
    old,new,rows,_=dev_fixture(tmp_path)
    receipt=runner.verify_fixed_dev(new,old);assert receipt['conditions']==36 and receipt['raw_source_hashes_equal']
    rows[0]['instruction']='changed target'
    (new/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=json.loads((new/'export_manifest.json').read_text())
    manifest['output_files_sha256']['observations.jsonl']=runner.exporter.digest(new/'observations.jsonl');write(new/'export_manifest.json',manifest)
    with pytest.raises(ValueError,match='identity differs'):runner.verify_fixed_dev(new,old)


def test_fixed_dev_raw_mutation_rejected_even_if_metadata_rows_same(tmp_path):
    old,new,_,labels=dev_fixture(tmp_path)
    Path(labels[0]['routes'][0]).write_text('changed trajectory')
    with pytest.raises(ValueError,match='SHA changed'):runner.verify_fixed_dev(new,old)


def test_dev_features_compare_arrays_by_id_not_archive_hash(tmp_path):
    roots=[tmp_path/'old',tmp_path/'new'];identifier='two_row_reach_283264_target0'
    for index,root in enumerate(roots):
        root.mkdir();path=root/'features.npz'
        np.savez(path,mean_hidden=np.ones(2048),last_hidden=np.ones(2048),id=identifier,parent_id='two_row_reach_283264',
            split='DEV_MODEL',image_sha256='fixed-image',input_tokens=100,extra_source_marker=index)
        (root/'samples.jsonl').write_text(json.dumps(dict(id=identifier,file=path.name,sha256=runner.exporter.digest(path)))+'\n')
    result=runner.compare_dev_features(roots[1],roots[0],[identifier])
    assert result['passed'] and result['per_condition'][0]['old_sha256']!=result['per_condition'][0]['new_sha256']
    path=roots[1]/'features.npz'
    with np.load(path) as a:arrays={key:a[key].copy() for key in a.files}
    arrays['mean_hidden'][0]+=.001;np.savez(path,**arrays)
    (roots[1]/'samples.jsonl').write_text(json.dumps(dict(id=identifier,file=path.name,sha256=runner.exporter.digest(path)))+'\n')
    result=runner.compare_dev_features(roots[1],roots[0],[identifier])
    assert not result['passed'] and result['per_condition'][0]['feature_max_abs_difference']['mean_hidden']>0


def test_quality_allows_missing_parent_and_zero_reference_actual_input(tmp_path,monkeypatch):
    config=selection(32);data=tmp_path/'data';data.mkdir();rows=[];labels=[]
    # Parent20 failed before saving an image. Never fabricate its three inputs.
    for index in list(range(20))+list(range(21,32))+list(range(64,76)):
        parent=f'two_row_reach_{283200+index}';role='TRAIN' if index<64 else 'DEV_MODEL'
        for target in range(3):
            identifier=parent+f'_target{target}'
            rows.append(dict(id=identifier,parent_id=parent,split=role,image='already checked',instruction='actual language'))
            labels.append(dict(id=identifier,parent_id=parent,split=role,routes=[] if index==0 else ['checked positive']))
    for name,values in (('observations.jsonl',rows),('supervision.jsonl',labels)):
        (data/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
    manifest=dict(selection=config,actual_inputs=129,requested_inputs=132,source_files_sha256={})
    write(data/'export_manifest.json',manifest)
    monkeypatch.setattr(runner.exporter,'verify_export',lambda root:(manifest,{}))
    train=[r for r in labels if r['split']=='TRAIN'];q=tmp_path/'quality.json'
    quality=dict(protocol='two_row_train_endpoint_capacity_and_reference_quality_v1',capacity_gate_passed=True,
        dev_raw_arrays_opened=False,locked_raw_opened=False,registered_train_parents=32,registered_train_indices=list(range(32)),
        source_export_manifest_sha256=runner.exporter.digest(data/'export_manifest.json'),
        train_input_ids=sorted(r['id'] for r in train),train_reference_counts={r['id']:len(r['routes']) for r in train},source_files_sha256={})
    write(q,quality);assert len(runner.validate_quality(data,config,q)[2])==129
    quality['train_input_ids'].pop();write(q,quality)
    with pytest.raises(ValueError,match='Complete fixed TRAIN'):runner.validate_quality(data,config,q)


def test_gpu_stage_caches_then_trains_then_audits_initialization(tmp_path,monkeypatch):
    source=tmp_path/('c'*40);source.mkdir();monkeypatch.chdir(source)
    monkeypatch.setenv('CODE_COMMIT',source.name);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:{0},raising=False)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *args,**kwargs:runner.GPU_UUID)
    paths=runner.stage_paths(tmp_path,source,32);write(paths['selection'],selection(32))
    quality=paths['preparation']/'train_quality.json';write(quality,{'checked':True})
    write(paths['data']/'export_manifest.json',{'checked':True})
    dev=dict(input_ids=['fixed36']);write(paths['preparation']/'fixed_dev_identity.json',dev)
    receipt=dict(source_export_manifest_sha256=runner.exporter.digest(paths['data']/'export_manifest.json'),
        quality_audit_sha256=runner.exporter.digest(quality),fixed_dev_identity_sha256=runner.exporter.digest(paths['preparation']/'fixed_dev_identity.json'),
        selection_sha256=runner.exporter.digest(paths['selection']),capacity_gate_passed=True)
    write(paths['preparation']/'preparation_receipt.json',receipt)
    monkeypatch.setattr(runner,'validate_quality',lambda *args:({}, {}, [{'id':'actual'}]))
    monkeypatch.setattr(runner,'verify_fixed_dev',lambda *args:dev)
    monkeypatch.setattr(runner,'source_record',lambda *args:{})
    monkeypatch.setattr(runner,'verify_cache',lambda *args:{'checked':True})
    monkeypatch.setattr(runner,'compare_dev_features',lambda *args:{'passed':True})
    monkeypatch.setattr(runner.exporter,'verify_export',lambda *args:({},{}))
    calls=[]
    def record(project,output,run_id,environment,module,args,resume):
        calls.append(run_id)
        if run_id=='peak_seed0':
            write(output/'peak_seed0/summary.json',dict(last_step=1500,trajectory_exposures=192000,
                sample_stream_audit=dict(batches=1500,observation_draws=48000,initial_model_sha256=runner.EXPECTED_INITIAL_MODEL_SHA256)))
    monkeypatch.setattr(runner,'run_record',record)
    runner.run(tmp_path,source,32,'gpu',source/'launcher')
    assert calls==['qwen_cache','peak_seed0']
    assert json.loads((paths['training']/'initialization_and_exposure_audit.json').read_text())['passed']
