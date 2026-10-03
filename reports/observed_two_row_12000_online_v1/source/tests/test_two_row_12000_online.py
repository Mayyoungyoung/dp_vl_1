import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_two_row_12000_online as entry


def completed(tmp_path):
    run=tmp_path/'peak_seed0';run.mkdir()
    (run/'best.pt').write_bytes(b'fixed original best')
    (run/'last.pt').write_bytes(b'fixed12000')
    spec=entry.ARMS['no_direct']
    cfg=dict(steps=12000,eval_every=250,lr=.0003,batch_size=32,seed=0,train_examples=189,dev_examples=36,
        two_row_export_sha256=entry.EXPORT_SHA,code_commit=spec['source'],no_direct_protocol=spec['protocol'])
    summary=dict(last_step=12000,parameters=682845,trajectory_exposures=1536000,best_step=250,
        best_checkpoint_sha256=entry.digest(run/'best.pt'))
    (run/'summary.json').write_text(json.dumps(summary))
    (run/'history.json').write_text(json.dumps([{'step':i} for i in range(250,12001,250)]))
    receipt=dict(protocol=spec['protocol'],summary_sha256=entry.digest(run/'summary.json'),passed=True,
        shared_initialization_exact=True,checkpoint_sha256={p.name:entry.digest(p) for p in (run/'best.pt',run/'last.pt')})
    (tmp_path/spec['receipt']).write_text(json.dumps(receipt))
    return run,cfg,summary,dict(status='completed',exit_code=0,step=12000)


def test_fixed_completed_original_best_and72_total_budget(tmp_path):
    run,cfg,summary,status=completed(tmp_path)
    receipt=entry.validate_training_identity('no_direct',run,cfg,summary,status)
    assert receipt['original_best_step']==250 and receipt['selection_opportunities']==48
    assert not receipt['reselection'] and len(entry.DEV_IDS)*len(entry.ARMS)==72
    assert len(entry.DEV_IDS)*len(entry.ARMS)*4==288


@pytest.mark.parametrize('field,value',[('steps',1500),('code_commit','other'),('dev_examples',48),
    ('lr',1e-4),('train_examples',96),('two_row_export_sha256','changed')])
def test_completion_rejects_other_source_budget_dataset_or_protocol(tmp_path,field,value):
    run,cfg,summary,status=completed(tmp_path);cfg[field]=value
    with pytest.raises(ValueError):entry.validate_training_identity('no_direct',run,cfg,summary,status)


def test_reselection_and_changed_receipt_artifacts_are_rejected(tmp_path):
    run,cfg,summary,status=completed(tmp_path)
    (run/'best.pt').write_bytes(b'another checkpoint')
    with pytest.raises(ValueError,match='receipt'):entry.validate_training_identity('no_direct',run,cfg,summary,status)
    with pytest.raises(ValueError):entry.validate_training_identity('no_direct',run,cfg,summary,dict(status='running',step=12000))


def test_scoped_online_adapter_restores_preflight_and_model_without_model_calls(monkeypatch):
    class Old:pass
    class New:pass
    module=SimpleNamespace(ObservedGeometryRouteHead=Old)
    original=entry.online.preflight
    monkeypatch.setattr(entry,'preflight',lambda args,arm:arm)
    with pytest.raises(RuntimeError):
        with entry.entry_adapter('no_direct',module,New):
            assert module.ObservedGeometryRouteHead is New
            assert entry.online.preflight(None)=='no_direct'
            raise RuntimeError('fixture interruption')
    assert entry.online.preflight is original and module.ObservedGeometryRouteHead is Old


def test_saved_feature_byte_comparison_preserves_mismatch_without_any_model(tmp_path):
    import hashlib
    run=tmp_path/'run';run.mkdir();output=tmp_path/'online';(output/'requests').mkdir(parents=True)
    cache=tmp_path/'cache';cache.mkdir()
    row=dict(id=entry.DEV_IDS[0],parent_id=entry.online.DEV_PARENTS[0],split='DEV_MODEL',image='/fixed/front.png',instruction='touch')
    obs=tmp_path/'observations.jsonl';obs.write_text(json.dumps(row)+'\n')
    (run/'config.json').write_text(json.dumps(dict(cache_dir=str(cache),observations=str(obs))))
    key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
    a=dict(mean_hidden=np.zeros(2048,np.float32),last_hidden=np.ones(2048,np.float32),input_tokens=np.array(3))
    path=cache/(key+'.npz');np.savez(path,**a)
    (run/'source_hashes.json').write_text(json.dumps({str(path.resolve()):entry.digest(path)}))
    value=dict(a,mean_hidden=a['mean_hidden']+.01)
    pred=output/'requests'/(row['id']+'.npz');np.savez(pred,**value)
    pred.with_suffix('.seal.json').write_text(json.dumps({'prediction_sha256':entry.digest(pred)}))
    (output/'cache_comparison.json').write_text(json.dumps({'per_scene':[{'id':row['id'],'candidate_decisions_identical':False}]}))
    result=entry.add_exact_saved_comparison(output,run)
    assert result['additional_forward_requests']==result['additional_qwen_encodings']==0
    assert not result['all_feature_bytes_exact'] and not result['all_candidate_decisions_identical']
    assert result['per_scene'][0]['feature_exact_bytes']==dict(mean_hidden=False,last_hidden=True,input_tokens=True)
    pred.write_bytes(b'changed')
    with pytest.raises(ValueError,match='seal changed'):entry.add_exact_saved_comparison(output,run)
