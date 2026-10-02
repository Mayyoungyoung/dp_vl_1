import json
from pathlib import Path

import pytest
torch=pytest.importorskip('torch')

from routeset.common import sha256
from scripts import evaluate_legacy_multitask_transfer as transfer


@pytest.fixture
def checkpoint(tmp_path,monkeypatch):
    config=dict(code_commit='a'*40,seed=0,endpoint_mode='free_offset',refinement_mode='none',
        candidates=4,horizon=24,steps=1500,metric_aggregation='task_parent',grounding_weight=0.)
    run=tmp_path/'runs/family/ordinary_seed0';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(config));(run/'summary.json').write_text('{}')
    torch.save(dict(config=config,step=750,model={'weight':torch.zeros(2)}),run/'best.pt')
    row=dict(run=str(run),training_commit='a'*40,seed=0,arm='ordinary',step=750,checkpoint='best.pt',
        checkpoint_sha256=sha256(run/'best.pt'),config_sha256=sha256(run/'config.json'),summary_sha256=sha256(run/'summary.json'))
    monkeypatch.setattr(transfer,'audit_model_source',lambda *args:{'verified':'separately pure tested'})
    return row,run,config


def test_actual_loaded_checkpoint_step_and_config_are_checked(checkpoint):
    row,run,config=checkpoint
    assert transfer.inspect_checkpoint(row)[1]['step']==750
    torch.save(dict(config=config,step=1500,model={}),run/'best.pt')
    row['checkpoint_sha256']=sha256(run/'best.pt')
    with pytest.raises(ValueError,match='Checkpoint state disagrees'):transfer.inspect_checkpoint(row)
    torch.save(dict(config=dict(config,seed=1),step=750,model={}),run/'best.pt')
    row['checkpoint_sha256']=sha256(run/'best.pt')
    with pytest.raises(ValueError,match='Checkpoint state disagrees'):transfer.inspect_checkpoint(row)


def test_checkpoint_bytes_and_registered_architecture_are_checked(checkpoint):
    row,run,config=checkpoint
    (run/'best.pt').write_bytes((run/'best.pt').read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Frozen checkpoint/config/summary changed'):transfer.inspect_checkpoint(row)
    row['checkpoint_sha256']=sha256(run/'best.pt')
    config['candidates']=8;(run/'config.json').write_text(json.dumps(config));row['config_sha256']=sha256(run/'config.json')
    with pytest.raises(ValueError,match='architecture/seed/protocol'):transfer.inspect_checkpoint(row)
