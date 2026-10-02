"""Read-only v1 anchor loading and exact comparison of recorded state fields.

v1 did not persist native configuration-tree buffers. This is a reconstruct-
then-verify protocol, not restoration of arbitrary hidden simulator state.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


PARENT = "two_row_reach_283000"
ANCHOR_FILES = ("manifest.json", PARENT+"/preparation_trace.npz", PARENT+"/restore_reference.json",
                PARENT+"/observation.npz", PARENT+"/front.png")
UNRECORDED = ["joint target positions read back at canonical initial state", "joint target velocities read back",
              "gripper joint measured velocities", "joint forces/torques", "per-link dynamic velocities",
              "contact/solver warm-start state", "controller internal state", "native task/arm/gripper configuration-tree buffers"]


def load_anchor(config):
    declaration = config["initial_state_anchor"]
    if declaration["protocol"] != "reconstruct_v1_prep_terminal_then_exact_readback_v1":
        raise ValueError("unknown anchor protocol")
    root = Path(declaration["dataset"])
    expected = declaration["files_sha256"]
    if set(expected) != set(ANCHOR_FILES):
        raise ValueError("anchor must freeze exactly the five required source files")
    actual = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ANCHOR_FILES}
    if actual != expected:
        raise ValueError("frozen v1 anchor artifact SHA mismatch")
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["config"]["protocol"] != "observed_two_row_low_posts_v1":
        raise ValueError("anchor source is not the recorded v1 setting")
    if any(config.get(key) != value for key,value in manifest["config"].items() if key != "protocol"):
        raise ValueError("physical/source seed/acceptance configuration changed from v1")
    with np.load(root/PARENT/"preparation_trace.npz",allow_pickle=False) as archive:
        arm = np.asarray(archive["arm_joint_positions"][-1],dtype=float)
        gripper = np.asarray(archive["gripper_joint_positions"][-1],dtype=float)
    if arm.shape!=(7,) or gripper.shape!=(2,) or not np.isfinite(np.r_[arm,gripper]).all():
        raise ValueError("invalid preparation terminal joint state")
    with np.load(root/PARENT/"observation.npz",allow_pickle=False) as archive:
        observation=SimpleNamespace(front_rgb=np.asarray(Image.open(root/PARENT/"front.png")),
            front_depth=archive["depth"].copy(),gripper_pose=archive["gripper_pose"].copy(),
            gripper_open=archive["gripper_open"].copy(),misc={
                "front_camera_intrinsics":archive["camera_intrinsics"].copy(),
                "front_camera_extrinsics":archive["camera_extrinsics"].copy()})
    world=json.loads((root/PARENT/"restore_reference.json").read_text())
    return dict(source=str(root),hashes=actual,world=world,observation=observation,
                preparation_arm_joints=arm,preparation_gripper_joints=gripper)


def exact_readback(anchor, current_world, current_observation):
    expected=anchor["world"]
    inventory_equal=[list(x) for x in expected["inventory"]]==[list(x) for x in current_world["inventory"]]
    missing=[];maximum=0.;worst=None;all_numeric=True
    for name in sorted(set(expected["state"])|set(current_world["state"])):
        if name not in expected["state"] or name not in current_world["state"]:
            missing.append(name);continue
        old,new=expected["state"][name],current_world["state"][name]
        for field in sorted(set(old)|set(new)):
            if field not in old or field not in new:
                missing.append(name+"."+field);continue
            a,b=np.asarray(old[field],dtype=float),np.asarray(new[field],dtype=float)
            if a.shape!=b.shape:
                missing.append(name+"."+field+" shape");continue
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                all_numeric=False;continue
            difference=float(np.abs(a-b).max())
            if difference>maximum:maximum,worst=difference,name+"."+field
    arrays={}
    for name in ("front_rgb","front_depth","gripper_pose","gripper_open"):
        a,b=np.asarray(getattr(anchor["observation"],name)),np.asarray(getattr(current_observation,name))
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        arrays[name]=dict(equal=bool(np.array_equal(a,b) and finite),max_abs_difference=
            float(np.abs(a.astype(float)-b.astype(float)).max()) if a.shape==b.shape and finite else None)
    for name in ("front_camera_intrinsics","front_camera_extrinsics"):
        a,b=np.asarray(anchor["observation"].misc[name]),np.asarray(current_observation.misc[name])
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        arrays[name]=dict(equal=bool(np.array_equal(a,b) and finite),max_abs_difference=
            float(np.abs(a.astype(float)-b.astype(float)).max()) if a.shape==b.shape and finite else None)
    passed=inventory_equal and not missing and all_numeric and maximum==0 and all(v["equal"] for v in arrays.values())
    return dict(protocol="strict_v1_recorded_state_readback_v1",passed=bool(passed),source=anchor["source"],
        source_files_sha256=anchor["hashes"],inventory_equal=inventory_equal,missing_or_mismatched_fields=missing,
        all_world_values_finite=all_numeric,world_max_abs=maximum,worst_world_field=worst,observation_fields=arrays,
        arbitrary_hidden_dynamic_state_equivalence=None,unrecorded_v1_fields=UNRECORDED,
        boundary="Exact equality of every recorded field only; zero is required, including small nonzero measured arm velocities")


def apply_recorded_task_state(task, posts, anchor):
    """Restore saved task/obstacle fields; robot links follow joint kinematics."""
    from pyrep.const import ObjectType
    state=anchor["world"]["state"]
    for obj in task._task.get_base().get_objects_in_tree(exclude_base=False):
        values=state[obj.get_name()]
        obj.set_pose(values["pose"])
        if obj.get_type()==ObjectType.SHAPE:obj.set_color(values["color"])
        if obj.get_type()==ObjectType.JOINT:obj.set_joint_position(values["joint_position"][0])
    for post in posts:
        values=state["_extra_"+post.get_name()]
        post.set_pose(values["pose"]);post.set_color(values["color"])
        post.set_collidable(bool(values["flags"][0]));post.set_respondable(bool(values["flags"][1]));post.set_dynamic(bool(values["flags"][2]))
