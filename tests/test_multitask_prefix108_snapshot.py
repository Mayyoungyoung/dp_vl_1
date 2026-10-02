import copy
import json
from pathlib import Path
import pytest
from scripts.observation_collect_multitask import registration
from scripts.snapshot_multitask_observations import build,digest,select_prefix,write_json
from test_multitask_snapshot import make_corpus


@pytest.fixture
def final_prefix(tmp_path):
    selection=json.loads((Path(__file__).resolve().parents[1]/'configs/multitask_prefix108_registration_v1.json').read_text())
    selected=select_prefix(registration(282000,'validated-five-v1','interleaved-early-dev-v1'),selection)
    source,other,_=make_corpus(tmp_path,selected)
    selection.update(source_dataset=str(source),compare_sources=[str(other)],source_partition_manifest_sha256=digest(source/'partition_manifest.json'))
    path=tmp_path/'registration.json';write_json(path,selection)
    return source,other,selected,path


def test_all_108_fixed_requests_retain_failure_denominators_and_same_dev(final_prefix,tmp_path):
    source,other,selected,path=final_prefix
    result=build(source,[other],tmp_path/'observation_multitask_prefix108_v1',path)
    assert result['requested_parents']==108 and result['requested_attempts']==324
    assert result['requested_parent_counts']=={'TRAIN':96,'DEV_MODEL':12}
    assert result['setup_failed_parents']==1 and result['parents_with_zero_reference']==1
    assert result['actual_attempt_records']==321 and result['unattempted_slots']==3
    assert result['successful_reference_routes']==106
    old=select_prefix(registration(282000,'validated-five-v1','interleaved-early-dev-v1'))
    assert [r for r in selected if r['split']=='DEV_MODEL']==[r for r in old if r['split']=='DEV_MODEL']


def test_missing_final_requested_parent_cannot_be_replaced(final_prefix,tmp_path):
    source,other,selected,path=final_prefix;last=selected[-1]
    (source/last['split']/'parents'/last['parent_id']/'closed.json').rename(source/'withheld_closure.json')
    with pytest.raises(ValueError,match='not fully closed'):build(source,[other],tmp_path/'observation_multitask_prefix108_v1',path)


def test_extended_registration_cannot_add_future_roles_or_disguise_smaller_prefix(final_prefix,tmp_path):
    source,other,_,path=final_prefix;selection=json.loads(path.read_text())
    wrong=copy.deepcopy(selection);wrong['train_parent_indices']=list(range(8));write_json(path,wrong)
    with pytest.raises(ValueError,match='Unrecognized'):build(source,[other],tmp_path/'observation_multitask_prefix108_v1',path)
    wrong=copy.deepcopy(selection);wrong['selected_requested_parents'][-1]['split']='TEST_LOCKED';write_json(path,wrong)
    with pytest.raises(ValueError,match='Requested parent list'):build(source,[other],tmp_path/'observation_multitask_prefix108_v1',path)
