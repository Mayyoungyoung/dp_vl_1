import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.two_row_anchor import ANCHOR_FILES, PARENT, exact_readback, load_anchor


def fixture_anchor():
    observed=SimpleNamespace(front_rgb=np.zeros((2,2,3),dtype=np.uint8),front_depth=np.ones((2,2)),
        gripper_pose=np.arange(7,dtype=float),gripper_open=np.array(1.),misc={
            "front_camera_intrinsics":np.eye(3),"front_camera_extrinsics":np.eye(4)})
    return dict(source="fixture",hashes={},world={"inventory":[["robot",1,0]],"state":{
        "_robot":{"arm_joints":[0.,1.],"arm_velocity":[.0005,0.],"gripper_joints":[.04,.04]},
        "target":{"pose":[0.,0.,0.,0.,0.,0.,1.],"velocity":[0.]*6}}},observation=observed)


def test_exact_recorded_state_allows_only_zero_difference_and_normalizes_inventory_container():
    anchor=fixture_anchor();world=copy.deepcopy(anchor["world"])
    world["inventory"]=[tuple(row) for row in world["inventory"]]
    result=exact_readback(anchor,world,copy.deepcopy(anchor["observation"]))
    assert result["passed"] and result["world_max_abs"]==0
    assert result["arbitrary_hidden_dynamic_state_equivalence"] is None
    assert "joint forces/torques" in result["unrecorded_v1_fields"]


@pytest.mark.parametrize("field",["velocity","depth","rgb","calibration","inventory","missing_field"])
def test_any_recorded_state_difference_closes_gate(field):
    anchor=fixture_anchor();world=copy.deepcopy(anchor["world"]);observation=copy.deepcopy(anchor["observation"])
    if field=="velocity":world["state"]["_robot"]["arm_velocity"][0]=0.
    elif field=="depth":observation.front_depth[0,0]+=1e-12
    elif field=="rgb":observation.front_rgb[0,0,0]=1
    elif field=="calibration":observation.misc["front_camera_intrinsics"][0,0]+=1e-12
    elif field=="inventory":world["inventory"].append(["new_object",2,0])
    else:del world["state"]["target"]["velocity"]
    result=exact_readback(anchor,world,observation)
    assert not result["passed"]
    json.dumps(result,allow_nan=False)


def test_anchor_loader_freezes_source_and_uses_precanonical_preparation_joints(tmp_path):
    folder=tmp_path/PARENT;folder.mkdir()
    base=json.loads((Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v1.json").read_text())
    (tmp_path/"manifest.json").write_text(json.dumps({"config":base}))
    anchor=fixture_anchor();(folder/"restore_reference.json").write_text(json.dumps(anchor["world"]))
    Image.fromarray(anchor["observation"].front_rgb).save(folder/"front.png")
    np.savez(folder/"observation.npz",depth=anchor["observation"].front_depth,gripper_pose=np.arange(7),gripper_open=1.,
        camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4))
    np.savez(folder/"preparation_trace.npz",arm_joint_positions=np.stack([np.zeros(7),np.arange(7)]),
        gripper_joint_positions=np.array([[0.,0.],[.04,.04]]))
    config=copy.deepcopy(base);config["initial_state_anchor"]={"protocol":"reconstruct_v1_prep_terminal_then_exact_readback_v1",
        "dataset":str(tmp_path),"files_sha256":{name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in ANCHOR_FILES}}
    loaded=load_anchor(config)
    assert np.array_equal(loaded["preparation_arm_joints"],np.arange(7))
    (folder/"restore_reference.json").write_text("{}")
    with pytest.raises(ValueError,match="SHA mismatch"):load_anchor(config)
