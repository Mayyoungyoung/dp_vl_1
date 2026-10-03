"""Saved-evidence failures and arithmetic, no Torch/model/raw-data requirement."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from routeset.observed_qwen_continuation import RequestJournal, digest
from scripts import analyze_observed_two_row_diffusion as a


def candidate(k, finite=True, mode=None):
    return dict(candidate=k, finite_xyz=finite, finite_event_values=finite,
        semantic_goal_correct=finite, starts_at_current_state=finite,
        tip_segments_clear=finite, event_state_sequence_correct=finite,
        TipValid=finite, declared_passage_type=mode,
        classified_tip_valid=finite and mode is not None)


def rows(n=36):
    result=[]
    for i in range(n):
        parent='two_row_reach_%d'%(283264+i//3)
        cs=[candidate(0,mode=['middle','middle']),candidate(1,mode=['middle','middle']),
            candidate(2),candidate(3,False)]
        metrics=dict(evaluation_protocol='observed_two_row_tip_eval_v1',TipValidAtK=.75,
            AnyTipValidAtK=1.,UniqueClassifiedTipValidAtK=1.,UnknownTypeTipValidCount=1.,
            DuplicateClassifiedTipValidCount=1.,KnownReferenceTypeCoverageAtK=.5,
            semantic_goal_accuracy=.75,TipClearAtK=.75,EventSequenceCorrectAtK=.75)
        result.append(dict(scene_id=parent+'_target'+str(i%3),parent_id=parent,tip_candidates=cs,
            tip_evaluation=metrics,reference_count=3,candidate_matched_ADE_m=.1,reference_matched_ADE_m=.2))
    return result


def history():
    return [dict(step=s,score=.0375+1,pool='dev/step%05d'%s,
        metrics=dict(UniqueClassifiedTipValidAtK=1,TipValidAtK=.75,selection_score=1.0375))
        for s in range(250,12001,250)]


def test_saved_first_tie_not_later_reselection():
    h=history()
    assert a.original_best(h)['step']==250
    h[-1]['metrics']['TipValidAtK']=1
    with pytest.raises(ValueError,match='score'):a.original_best(h)


def test_48_choice_schedule_is_fixed():
    with pytest.raises(ValueError,match='48'):a.original_best(history()[:-1])


def test_repeats_are_metric_average_never_candidate_union():
    pools={r:dict(rows=rows()) for r in range(3)}
    result=a.repeated_summary(pools)
    assert result['mean']['UniqueClassifiedTipValidAtK']==1
    assert result['mean']['UnknownTypeTipValidCount']==1
    assert all(x['candidate_slots']==144 for x in result['per_repeat'].values())
    assert result['pools_merged'] is False and result['training_seeds']==1
    pools[1]['rows']=rows(72)
    with pytest.raises(ValueError,match='K4'):a.repeated_summary(pools)


def test_unknown_and_failed_slots_are_not_dropped():
    value=a.aggregate(rows())
    assert value['candidate_slots']==144 and value['tip_valid_candidates']==108
    assert value['valid_unknown_total']==36 and value['classified_duplicate_total']==36
    assert value['semantic_collision_four_cells']['wrong_goal__collision_or_invalid']==36


def test_all_stages_gate_reads_only_status_before_reject(tmp_path):
    folder=tmp_path/'independent';folder.mkdir()
    a.write(folder/'status.json',dict(status='running',arm='independent',stage='train'))
    evidence=a.Evidence()
    with pytest.raises(ValueError,match='complete'):a.require_completed(tmp_path,evidence)
    assert len(evidence.hashes)==1 and next(iter(evidence.hashes)).endswith('status.json')


def test_exact_journal_rejects_reordering_with_same_counts(tmp_path):
    journal=RequestJournal(tmp_path/'requests.jsonl')
    journal.issue('train_denoise','1');journal.issue('train_geometry','1')
    with pytest.raises(ValueError,match='order'):
        a.check_journal(tmp_path,dict(actual_calls=journal.snapshot()),
                        [('train_geometry','1'),('train_denoise','1')],a.Evidence())


@pytest.fixture
def pool(tmp_path):
    folder=tmp_path/'pool';folder.mkdir()
    config=dict(eval_noise_seeds=[300000,300001,300002]);data=rows(1)
    ids=[data[0]['scene_id']];noise,noise_id=a.evaluation_noise(ids,300000)
    schedule=dict(sampling_indices=np.linspace(99,0,40).round().astype(int).tolist())
    identity=dict(protocol=a.DRIVER_PROTOCOL,config=config,step=250,kind='dev',repeat=0,
        ids=ids,requests=1,candidates=4,final_path_states=4,denoiser_calls=40,
        intermediate_path_states_including_final=160,model_sha256='a'*64,schedule=schedule,noise=noise_id)
    journal=RequestJournal(tmp_path/'requests.jsonl');before=journal.snapshot()
    for kind,key in a.evaluation_keys(identity):journal.issue(kind,key)
    paths=np.zeros((1,4,24,3),np.float32);paths[:,3]=np.nan
    events=np.zeros((1,4,24),np.float32);events[:,3]=np.nan
    np.savez_compressed(folder/'predictions.npz',paths=paths,gripper_open=events,
        scene_ids=np.asarray(ids),parent_ids=np.asarray([data[0]['parent_id']]))
    np.savez_compressed(folder/'initial_noise.npz',ids=np.asarray(ids),initial_noise=noise)
    metrics={k:a.aggregate(data)[k] for k in a.METRICS}
    metrics.update(candidate_matched_ADE_m=.1,reference_matched_ADE_m=.2,examples=1,candidates=4)
    a.write(folder/'metrics.json',metrics);a.write(folder/'per_scene.json',data)
    generation=dict(identity=identity,noise=noise_id,predictions_sha256=digest(folder/'predictions.npz'),
        all_predictions_sealed_before_labels=True,requests=[dict(id=ids[0],noise_sha256=a.array_digest(noise[0]),
            actual_calls=[[i,t] for i,t in enumerate(schedule['sampling_indices'])],
            geometry_and_input_seconds=.1,denoising_seconds=.2,cached_sampler_seconds=.3,
            finite_candidates=[True,True,True,False])])
    a.write(folder/'generation.json',generation)
    receipt=dict(identity=identity,journal_before=before,journal_after=journal.snapshot(),metrics=metrics,
        artifacts={p.name:digest(p) for p in folder.iterdir()})
    a.write(folder/'pool_receipt.json',receipt)
    return folder,config,ids,schedule,journal


def verify(fixture):
    folder,config,ids,schedule,journal=fixture
    return a.verify_pool(folder,config,ids,250,'dev',0,schedule,journal,a.Evidence())


def reseal(folder,name):
    receipt=a.read(folder/'pool_receipt.json');receipt['artifacts'][name]=digest(folder/name)
    a.write(folder/'pool_receipt.json',receipt)


def test_valid_sealed_pool_retains_nan_failure(pool):
    result=verify(pool)
    assert result['rows'][0]['tip_evaluation']['TipValidAtK']==.75
    assert np.isnan(result['arrays']['paths'][0,3]).all()


def test_noise_relabel_even_if_rehashed_is_rejected(pool):
    folder,config,ids,schedule,journal=pool
    noise,_=a.evaluation_noise(ids,300001)
    np.savez_compressed(folder/'initial_noise.npz',ids=np.asarray(ids),initial_noise=noise)
    reseal(folder,'initial_noise.npz')
    with pytest.raises(ValueError,match='noise'):verify(pool)


def test_missing_40th_call_not_hidden_by_receipt(pool):
    folder=pool[0];generation=a.read(folder/'generation.json')
    generation['requests'][0]['actual_calls'].pop()
    a.write(folder/'generation.json',generation);reseal(folder,'generation.json')
    with pytest.raises(ValueError,match='40 timestep'):verify(pool)


def test_label_decision_cannot_upgrade_nan_candidate(pool):
    folder=pool[0];data=a.read(folder/'per_scene.json');data[0]['tip_candidates'][3]['finite_xyz']=True
    a.write(folder/'per_scene.json',data);reseal(folder,'per_scene.json')
    with pytest.raises(ValueError,match='finite'):verify(pool)


def test_unsealed_or_extra_artifact_rejected(pool):
    folder=pool[0];receipt=a.read(folder/'pool_receipt.json');del receipt['artifacts']['predictions.npz']
    a.write(folder/'pool_receipt.json',receipt)
    with pytest.raises(ValueError,match='Incomplete'):verify(pool)


def test_pool_receipt_boundary_cannot_hide_extra_issued_call(pool):
    folder,config,ids,schedule,journal=pool
    receipt=a.read(folder/'pool_receipt.json');receipt['journal_before']['sha256']='f'*64
    a.write(folder/'pool_receipt.json',receipt)
    with pytest.raises(ValueError,match='boundary'):verify(pool)


def test_no_model_or_torch_loaded_by_import():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',
        'import sys; import scripts.analyze_observed_two_row_diffusion; '
        'assert "torch" not in sys.modules; '
        'assert "routeset.observed_route_diffusion" not in sys.modules'],check=True,cwd=a.PROJECT)
