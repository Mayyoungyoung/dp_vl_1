"""Actual shared tiny-model continuation/resume, no Qwen claims."""
from copy import deepcopy
import json

import pytest
torch = pytest.importorskip('torch')

from routeset.common import sha256
from routeset.vlm_sft_data import fixed_development_plan,load_sft_data
from scripts.train_vlm_route_sft import digest_json,run_training,source_receipt
from test_vlm_sft_data import make_fixture,fixture_resample
from test_vlm_sft_training import TinyCausal,TorchProcessor,tensor_tree_equal


def test_continuous6_equals_completed2_continued_to4_then_strict_resume6(tmp_path):
    torch.set_num_threads(1)
    obs,sup=make_fixture(tmp_path/'data');data=load_sft_data(obs,sup,4,fixture_resample)
    config=dict(steps=6,seed=13,lr=.001,weight_decay=0.,horizon=4,chunk_size=64,gradient_clip=1.,
        eval_every=2,checkpoint_every=2,log_every=2,dev_plan_seed=313,data_fingerprint=data['fingerprint'],
        source_sha256=source_receipt())
    config['dev_plan_sha256']=digest_json(fixed_development_plan(data['samples'],313))
    def model():torch.manual_seed(13);return TinyCausal()
    full=run_training(model(),TorchProcessor(),data,config,tmp_path/'full',device='cpu')
    short=deepcopy(config);short['steps']=2
    origin=run_training(model(),TorchProcessor(),data,short,tmp_path/'original',device='cpu')
    before={p.name:sha256(p) for p in (tmp_path/'original').iterdir() if p.is_file()}
    for key,value in [('lr',.002),('data_fingerprint','changed'),('eval_every',4)]:
        wrong=deepcopy(config);wrong[key]=value
        with pytest.raises(ValueError):
            run_training(model(),TorchProcessor(),data,wrong,tmp_path/('wrong_'+key),device='cpu',
                         continue_from=tmp_path/'original/last.pt')
    # Normal resume must not silently extend the finished original budget.
    with pytest.raises(ValueError):
        run_training(model(),TorchProcessor(),data,config,tmp_path/'original',device='cpu',resume=True)
    half=run_training(model(),TorchProcessor(),data,config,tmp_path/'continued',device='cpu',stop_after=4,
                      continue_from=tmp_path/'original/last.pt')
    assert half['status']=='stopped_early' and half['step']==4
    assert half['continuation_accounting']['added_exposure']['TRAIN']['candidate_slots']==5
    done=run_training(model(),TorchProcessor(),data,config,tmp_path/'continued',device='cpu',resume=True)
    assert done['status']==full['status']=='completed'
    first=torch.load(tmp_path/'full/last.pt',weights_only=False)
    second=torch.load(tmp_path/'continued/last.pt',weights_only=False)
    for key in ('adapters','optimizer','scheduler','rng','sampler_state','step','history',
                'best_step','best_nll','exposure','gradient_audit'):
        tensor_tree_equal(first[key],second[key])
    assert done['continuation_accounting']['inherited_exposure']['TRAIN']['candidate_slots']==5
    assert done['continuation_accounting']['added_exposure']['TRAIN']['candidate_slots']==10
    assert done['continuation_accounting']['cumulative_exposure']['TRAIN']['candidate_slots']==15
    assert done['continuation_accounting']['inherited_elapsed_seconds']==origin['elapsed_seconds']
    assert all(row['changed_from_initial'] and row['ever_nonzero_gradient'] for row in done['added_gradient_audit'].values())
    assert before=={p.name:sha256(p) for p in (tmp_path/'original').iterdir() if p.is_file()}
    assert sha256(tmp_path/'continued/inherited_best.pt')==before['best.pt']
    assert json.loads((tmp_path/'continued/config.json').read_text())['steps']==6
    assert json.loads((tmp_path/'original/config.json').read_text())['steps']==2
    rows=[json.loads(x) for x in (tmp_path/'continued/requests.jsonl').read_text().splitlines()]
    assert [r['k'] for r in rows if r['kind']=='train_request']==[1,4,1,4,1,4]
    assert not (tmp_path/'continued/train.lock').exists()
