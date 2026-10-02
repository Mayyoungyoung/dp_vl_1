"""CPU-only finite-call and failure-budget tests with a fake generator."""
import json

import numpy as np
import pytest

from routeset.vlm_sft_evaluation import generate_comparison,request_seed


def samples():
    return [dict(id='scene%d'%n,parent_id='parent%d'%n,split='DEV_MODEL',instruction='reach red',
                 image_path='unread.png',observation_path='unread.npz') for n in range(2)]


def answer(k):return json.dumps([[[100,200,300,1],[400,500,600,1]]]*k)


def test_four_complete_independent_requests_and_whole_share_k4_without_repeat_pooling(tmp_path):
    calls=[]
    def generator(sample,k,seed,tokens):
        calls.append((sample['id'],k,seed,tokens))
        return dict(text=answer(k),tokens=dict(prompt_tokens=10,output_tokens=5*k))
    groups=generate_comparison(samples(),generator,tmp_path/'out',seed=0,repeats=2,horizon=2)
    assert len(calls)==20 and len(groups)==4
    assert [k for _,k,_,_ in calls[:10]]==[1]*8+[4]*2
    assert all(tokens==512*k for _,k,_,tokens in calls)
    assert all(group['requested_candidate_slots']==group['charged_candidate_slots']==8 for group in groups)
    assert all(not group['candidate_pooling_between_repeats'] for group in groups)
    for repeat in (0,1):
        for method in ('independent4','whole4'):
            with np.load(tmp_path/f'out/repeat{repeat}/{method}/predictions.npz') as saved:
                assert saved['paths'].shape==(2,4,2,3) and np.isfinite(saved['paths']).all()
    assert calls[0][2]!=calls[10][2]
    assert request_seed(0,0,'scene0','independent4',0)==calls[0][2]


def test_errors_missing_routes_and_overgeneration_are_charged_without_topk(tmp_path):
    count=0
    def generator(sample,k,seed,tokens):
        nonlocal count
        count+=1
        if count==1:return dict(text=answer(2),tokens=dict(prompt_tokens=10,output_tokens=20))
        if count==6:raise RuntimeError('synthetic request failure')
        if k==4:return dict(text='[]',tokens=dict(prompt_tokens=10,output_tokens=2))
        return dict(text=answer(1),tokens=dict(prompt_tokens=10,output_tokens=10))
    results=generate_comparison(samples(),generator,tmp_path/'out',horizon=2)
    independent,whole=results
    assert independent['charged_candidate_slots']==9 and independent['budget_exceeded_examples']==1
    assert independent['failed_requests']==1 and count==10
    with np.load(tmp_path/'out/repeat0/independent4/predictions.npz') as saved:
        assert np.isnan(saved['paths'][0]).all()  # entire >K scene fails fixed-K4
        assert np.isfinite(saved['nominal_parsed_paths'][0,1:]).all()
        assert np.isnan(saved['paths'][1,1]).all() and np.isfinite(saved['paths'][1,[0,2,3]]).all()
    assert whole['strict_finite_slots']==0 and whole['charged_candidate_slots']==8
    records=[json.loads(line) for line in (tmp_path/'out/repeat0/independent4/requests.jsonl').read_text().splitlines()]
    assert records[0]['text']==answer(2) and records[5]['failure']['type']=='RuntimeError'


def test_generation_boundary_rejects_supervision_and_never_overwrites(tmp_path):
    seen=[]
    bad=samples();bad[0]['routes']=['answer.npz']
    with pytest.raises(ValueError,match='observation metadata'):
        generate_comparison(bad,lambda *args:seen.append(args),tmp_path/'bad',horizon=2)
    assert not seen
    output=tmp_path/'existing';output.mkdir()
    with pytest.raises(FileExistsError):
        generate_comparison(samples(),lambda *args:seen.append(args),output,horizon=2)


def test_postprocess_reparses_raw_text_before_labels_and_rejects_changed_arrays(tmp_path):
    from scripts.analyze_vlm_route_sft import load_group
    def generator(sample,k,seed,tokens):
        route=[[100,200,300,1]]*24
        return dict(text=json.dumps([route]*k),tokens=dict(prompt_tokens=10,output_tokens=30*k))
    root=tmp_path/'out';generate_comparison(samples(),generator,root,horizon=24)
    folder=root/'repeat0/independent4';ids=[row['id'] for row in samples()]
    paths,_,_,_=load_group(folder,dict(seed=0),'independent4',0,ids)
    assert paths.shape==(2,4,24,3)
    with np.load(folder/'predictions.npz') as saved:arrays={key:saved[key].copy() for key in saved.files}
    arrays['paths'][0,0,2,0]+=.1
    np.savez_compressed(folder/'predictions.npz',**arrays)
    with pytest.raises(ValueError,match='strict parse'):
        load_group(folder,dict(seed=0),'independent4',0,ids)


def test_repeat_metric_average_does_not_add_unique_counts_or_invent_selected_valid():
    from scripts.analyze_vlm_route_sft import average
    rows=[dict(unique=1.,selected=None,coverage=.5),dict(unique=2.,selected=None,coverage=None)]
    assert average(rows,fields=('unique','selected','coverage'))==dict(unique=1.5,selected=None,coverage=.5)


def test_deep_malformed_json_is_a_charged_failure_in_generation_and_postprocessing(tmp_path):
    from scripts.analyze_vlm_route_sft import load_group
    def generator(sample,k,seed,tokens):
        return dict(text='['*1100+']'*1100,tokens=dict(prompt_tokens=10,output_tokens=1200))
    root=tmp_path/'out';groups=generate_comparison(samples(),generator,root,horizon=24)
    assert all(group['strict_finite_slots']==0 and group['charged_candidate_slots']==8 for group in groups)
    for method in ('independent4','whole4'):
        paths,_,_,_=load_group(root/f'repeat0/{method}',dict(seed=0),method,0,[r['id'] for r in samples()])
        assert np.isnan(paths).all()


def test_analysis_rejects_replaced_training_index_or_changed_export_geometry(tmp_path):
    import hashlib
    from scripts.analyze_vlm_route_sft import validate_source_index,verify_exported_file
    source={'fixture':'a'*64}
    expected=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    validate_source_index(source,expected)
    with pytest.raises(ValueError,match='index changed'):
        validate_source_index({'fixture':'b'*64},expected)
    path=(tmp_path/'verification.npz').resolve();path.write_bytes(b'original fixture')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    exported=dict(source_parents={'parent':dict(reserved_role='DEV_MODEL',source_files_sha256={str(path):digest})})
    assert verify_exported_file(path,'parent',exported)==digest
    path.write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='changed since'):
        verify_exported_file(path,'parent',exported)
