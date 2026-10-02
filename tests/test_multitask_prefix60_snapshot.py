"""Only the registered expanded TRAIN prefix may be exported; no success selection."""
import copy
import json
from pathlib import Path
import pytest

from scripts.observation_collect_multitask import registration
from scripts.snapshot_multitask_observations import build,digest,select_prefix,write_json
from test_multitask_snapshot import make_corpus


@pytest.fixture
def expanded(tmp_path):
    template=Path(__file__).resolve().parents[1]/'configs/multitask_prefix60_registration_v1.json'
    selection=json.loads(template.read_text(encoding='utf-8'))
    plan=registration(282000,'validated-five-v1','interleaved-early-dev-v1')
    selected=select_prefix(plan,selection)
    source,other,_=make_corpus(tmp_path,selected)
    selection.update(source_dataset=str(source),compare_sources=[str(other)],
        source_partition_manifest_sha256=digest(source/'partition_manifest.json'))
    path=tmp_path/'registered.json';write_json(path,selection)
    return source,other,selected,path


def test_expanded_exact_request_keeps_setup_failures_zero_refs_and_original_dev(expanded,tmp_path):
    source,other,selected,path=expanded;out=tmp_path/'observation_multitask_prefix60_v1'
    result=build(source,[other],out,path)
    assert result['requested_parents']==60 and result['requested_attempts']==180
    assert result['actual_attempt_records']==177 and result['unattempted_slots']==3
    assert result['setup_failed_parents']==1 and result['parents_with_zero_reference']==1
    assert result['successful_reference_routes']==58 and result['observations']==119
    assert result['requested_parent_counts']=={'TRAIN':48,'DEV_MODEL':12}
    default=select_prefix(registration(282000,'validated-five-v1','interleaved-early-dev-v1'))
    assert [r for r in selected if r['split']=='DEV_MODEL']==[r for r in default if r['split']=='DEV_MODEL']
    assert result['selection_registration_sha256']==digest(path)
    assert all(digest(out/name)==expected for name,expected in result['snapshot_files_sha256'].items())


def test_expanded_requires_last_requested_parent_closed_without_substitution(expanded,tmp_path):
    source,other,selected,path=expanded;last=selected[-1]
    (source/last['split']/'parents'/last['parent_id']/'closed.json').rename(source/'held_closure.json')
    with pytest.raises(ValueError,match='not fully closed'):
        build(source,[other],tmp_path/'observation_multitask_prefix60_v1',path)
    assert not (tmp_path/'observation_multitask_prefix60_v1').exists()


def test_expanded_registration_cannot_change_requested_roles_or_source_hash(expanded,tmp_path):
    source,other,_,path=expanded;original=json.loads(path.read_text())
    changed=copy.deepcopy(original);changed['source_partition_manifest_sha256']='0'*64;write_json(path,changed)
    with pytest.raises(ValueError,match='hash mismatch'):build(source,[other],tmp_path/'observation_multitask_prefix60_v1',path)
    changed=copy.deepcopy(original);changed['selected_requested_parents'][-1]['split']='TEST_LOCKED';write_json(path,changed)
    with pytest.raises(ValueError,match='Requested parent list'):build(source,[other],tmp_path/'observation_multitask_prefix60_v1',path)
