import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_multitask_landmark_pair import analyze, sha


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path):
    # Saved-metric protocol fixture only, no synthetic experiment claims.
    roots={arm:tmp_path/arm for arm in ('ordinary','auxiliary')}
    fields=('candidate_matched_ADE_m','reference_matched_ADE_m','candidate_endpoint_error_m','best_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')
    metric={field:.1 for field in fields}
    metric.update(examples=48,parents=12,reference_evaluation_examples=48,semantic_evaluation_examples=0,examples_without_reference=0,semantic_goal_accuracy=None,ValidAtK=None,
        task_parent_reference_metrics=dict(per_task=[dict(task='task',**{field:.1 for field in fields})],
            per_parent=[dict(parent_id='positive',task='task',**{field:.1 for field in fields}),dict(parent_id='unreferenced',task='task',**{field:None for field in fields})]))
    config={key:1 for key in ('observations','supervision','cache_dir','steps','batch_size','candidates','horizon','width','depth','pooling','geometry_pooling','anchor_mode','endpoint_mode','sampling_mode','metric_aggregation','refinement_mode','checkpoint_selection','point_width','pixel_stride','endpoint_residual_bound','grounding_sigma','event_scale','lr','seed','eval_every','feature_dim','dataset_fingerprint','train_examples','dev_examples','train_parents','dev_parents','checkpoint_selection_protocol','evaluation_protocol')}
    config.update(steps=2,batch_size=2,candidates=4)
    audit=dict(protocol='actual_observation_indices_sha256_chain_v1',batches=2,observation_draws=4,index_chain_sha256='a'*64,initial_model_sha256='b'*64)
    for arm,root in roots.items():
        root.mkdir();write(root/'config.json',dict(config,grounding_weight=0. if arm=='ordinary' else .02,grounding_target='endpoint' if arm=='ordinary' else 'event_supported'))
        write(root/'source_hashes.json',{'input':'hash'})
        summary=dict(metrics=metric,last_metrics=metric,train_metrics=metric,trajectory_exposures=16,parameters=1231965,sample_stream_audit=audit,elapsed_s=1.,data_load_preprocess_s=1.,gpu_hours_reserved=0.,peak_cuda_memory_mb=0.,best_step=2,last_step=2)
        for name,key in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('dev_model/predictions.npz','prediction_sha256'),('last_dev_model/predictions.npz','last_prediction_sha256')]:
            path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(name.encode());summary[key]=sha(path)
        write(root/'summary.json',summary)
    return roots


def test_actual_paired_audit_retains_null_reference_and_full_deltas(tmp_path):
    r=fixture(tmp_path);result=analyze(r['ordinary'],r['auxiliary'])
    assert result['status']=='passed' and len(result['per_parent'])==6
    assert result['per_parent'][1]['delta_auxiliary_minus_ordinary']['candidate_matched_ADE_m'] is None


@pytest.mark.parametrize('change',('stream','source','binary','denominator'))
def test_reject_unmatched_actual_evidence(tmp_path,change):
    r=fixture(tmp_path);path=r['auxiliary']/'summary.json';summary=json.loads(path.read_text())
    if change=='stream':summary['sample_stream_audit']['index_chain_sha256']='c'*64;write(path,summary)
    elif change=='source':write(r['auxiliary']/'source_hashes.json',{'input':'different'})
    elif change=='binary':(r['auxiliary']/'best.pt').write_bytes(b'changed')
    else:summary['metrics']['examples']=47;write(path,summary)
    with pytest.raises(ValueError):analyze(r['ordinary'],r['auxiliary'])
