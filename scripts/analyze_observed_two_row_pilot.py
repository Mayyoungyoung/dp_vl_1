"""Read-only analysis of all registered two-row pilot slots, including failures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts import collect_observed_two_row_pilot as collector
except ImportError:
    import collect_observed_two_row_pilot as collector


def row_crossings(xyz, config):
    """Every actual crossing with sample/arc progress; no guide classification."""
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    result=[]
    for index,(a,b) in enumerate(zip(xyz[:-1],xyz[1:])):
        for row,x in enumerate(config["row_x"]):
            if not min(a[0],b[0]) <= x <= max(a[0],b[0]):continue
            fraction=0. if abs(b[0]-a[0])<=1e-12 else float((x-a[0])/(b[0]-a[0]))
            points=[a,b] if abs(b[0]-a[0])<=1e-12 else [a+fraction*(b-a)]
            for point in points:
                margin=config["tip_clearance_m"];base=config["post_base_z"];top=base+config["post_size_xyz"][2]
                low,high=config["post_y"][row];half=config["post_size_xyz"][1]/2
                if point[2]>top+margin:label="over"
                elif not base+margin<=point[2]<=top-margin:label="ambiguous_height"
                elif point[1]<low-half-margin:label="negative_y"
                elif low+half+margin<point[1]<high-half-margin:label="middle"
                elif point[1]>high+half+margin:label="positive_y"
                else:label="blocked_or_boundary_y"
                result.append(dict(row=row,sample_segment=index,fraction=fraction,xyz=point.tolist(),
                    direction=int(np.sign(b[0]-a[0])),arc_progress=float((lengths[index]+fraction*(lengths[index+1]-lengths[index]))/max(lengths[-1],1e-12)),
                    passage_relation=label))
    return result


def segment_diagnostics(xyz, planning_segments, config):
    result=[];offset=0
    for segment in planning_segments:
        steps=segment["simulated_steps"]
        if steps==0:continue
        part=xyz[offset:offset+steps+1];offset+=steps
        goal=np.asarray(segment["goal_supervision_only"]);delta=goal-part[0]
        fraction=np.clip((part-part[0])@delta/max(float(delta@delta),1e-12),0.,1.)
        deviation=np.linalg.norm(part-(part[0]+fraction[:,None]*delta),axis=1)
        result.append(dict(segment=segment["segment"],simulation_status=segment["simulation_status"],
            first_sample=offset-steps,last_sample=offset,goal=goal.tolist(),actual_end=part[-1].tolist(),
            endpoint_error_m=float(np.linalg.norm(part[-1]-goal)),maximum_deviation_from_straight_guide_segment_m=float(deviation.max()),
            min_z_m=float(part[:,2].min()),max_z_m=float(part[:,2].max()),
            arc_length_m=float(np.linalg.norm(np.diff(part,axis=0),axis=1).sum()),
            actual_row_crossings=row_crossings(part,config)))
    if offset!=len(xyz)-1:raise ValueError("recorded simulated steps do not match trajectory samples")
    return result


def compare_initials(data, baseline):
    """Same seed/config is insufficient: compare the actual exported states."""
    current_config=json.loads((data/"manifest.json").read_text())["config"]
    old_config=json.loads((baseline/"manifest.json").read_text())["config"]
    current=data/("two_row_reach_%06d"%current_config["seed"])
    old=baseline/("two_row_reach_%06d"%old_config["seed"])
    names=("front.png","observation.npz","restore_reference.json")
    if not all((root/name).exists() for root in (current,old) for name in names):
        return dict(baseline=str(baseline),strict_initial_match=None,reason="one version lacks a completed initial observation")
    a=np.asarray(collector.Image.open(old/"front.png"));b=np.asarray(collector.Image.open(current/"front.png"))
    difference=collector.legacy.compare_restore(json.loads((old/"restore_reference.json").read_text()),
        json.loads((current/"restore_reference.json").read_text()),a,b)
    arrays={}
    with np.load(old/"observation.npz",allow_pickle=False) as first, np.load(current/"observation.npz",allow_pickle=False) as second:
        for key in sorted(set(first.files)|set(second.files)):
            matching=key in first and key in second and np.array_equal(first[key],second[key])
            arrays[key]=dict(equal=bool(matching),max_abs_difference=float(np.abs(first[key].astype(float)-second[key].astype(float)).max())
                if key in first and key in second and first[key].shape==second[key].shape else None)
    expected_keys=set(old_config)|set(current_config)
    physical_config_same=all(old_config.get(k)==current_config.get(k) for k in expected_keys if k not in ("protocol","row_plane_guides"))
    strict=(difference["global_inventory_equal"] and difference["max_abs"]==0 and difference["rgb_max_difference"]==0 and
            all(value["equal"] for value in arrays.values()) and physical_config_same)
    return dict(baseline=str(baseline),strict_initial_match=bool(strict),physical_config_equal=physical_config_same,
        world_and_rgb=difference,observation_fields=arrays,
        interpretation="Strictly paired initial condition" if strict else "Same requested geometry only; do not claim a strictly paired robot initial condition",
        initial_artifact_hashes={label:{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
                                for label,root in (("baseline",old),("current",current))})


def analyze(data, output, baseline=None):
    manifest = json.loads((data/"manifest.json").read_text())
    config = manifest["config"]
    centers, halves = collector.validate_config(config)
    target_indices = collector.registered_target_indices(config)
    summary = json.loads((data/"summary.json").read_text())
    records = [json.loads(line) for line in (data/"attempts.jsonl").read_text().splitlines()] if (data/"attempts.jsonl").exists() else []
    output.mkdir(parents=True, exist_ok=False)
    parent = data/("two_row_reach_%06d" % config["seed"])
    setup = json.loads((parent/"setup.json").read_text()) if (parent/"setup.json").exists() else {}
    rows, paths, hash_checks = [], {}, []
    for record in records:
        key = (record["input_id"], record["attempt"])
        trace = record.get("trajectory") or record.get("failed_partial_trajectory")
        first_clearance_failure = None
        actual_partial_signature = None;crossings=[];segment_details=[]
        if trace:
            path = parent/trace["file"]
            checked = hashlib.sha256(path.read_bytes()).hexdigest() == trace["sha256"]
            hash_checks.append(checked)
            if not checked:
                raise ValueError("trajectory hash mismatch: " + str(path))
            with np.load(path, allow_pickle=False) as archive:
                xyz = archive["gripper_pose"][:,:3]
            paths[key] = xyz
            actual_partial_signature = collector.crossing_signature(xyz, config)
            crossings=row_crossings(xyz,config)
            segment_details=segment_diagnostics(xyz,record.get("planning_segments",[]),config)
            for i, (a, b) in enumerate(zip(xyz[:-1], xyz[1:])):
                failed_posts = [j for j in range(len(centers)) if not collector.legacy.tip_polyline_clear(
                    np.stack([a,b]), centers[j:j+1], halves[j:j+1], config["tip_clearance_m"])]
                if failed_posts:
                    first_clearance_failure = dict(segment=i, post_indices=failed_posts, segment_start=a.tolist(), segment_end=b.tolist())
                    break
        segments = record.get("planning_segments", [])
        failure = next((dict(segment=s["segment"],stage=stage) for s in segments for stage in ("planning","simulation")
                        if s[stage+"_status"] == "failed"), None)
        rows.append(dict(id=record["input_id"],attempt=record["attempt"],success=record["success"],
            proposed=record["proposed_type_supervision_only"],actual_accepted_type=record.get("actual_route_type"),
            partial_path_signature_diagnostic_only=actual_partial_signature,
            first_failed_execution_stage=failure,error=record.get("error"),collision_label=record.get("collision_pair"),
            planning_calls=sum(s["get_path_calls"] for s in segments), planning_seconds=sum(s["planning_seconds"] for s in segments),
            simulation_seconds=sum(s["simulation_seconds"] for s in segments),worker_slot_seconds=record["seconds"],
            endpoint_error_m=record.get("endpoint_error_m"),length_m=record.get("length_m"),
            first_partial_tip_clearance_failure=first_clearance_failure,actual_row_crossings=crossings,
            guide_segment_diagnostics=segment_details))
    def count(values):
        return dict(Counter(values))
    result = dict(source=str(data),source_collector_sha256=manifest["sources_sha256"],analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        collection_summary=summary,setup_success=setup.get("success"),setup_seconds=setup.get("seconds"),
        registered_target_indices=target_indices,registered_route_slots=config["requested_route_proposals"],
        all_trajectory_hashes_match=all(hash_checks),trajectory_hashes_checked=len(hash_checks),
        success_count=sum(r["success"] for r in records),
        failure_counts=count(r.get("error") for r in records if not r["success"]),
        first_failed_stage_counts=count(str(r["first_failed_execution_stage"]) for r in rows if r["first_failed_execution_stage"]),
        partial_paths_with_tip_clearance_failure=sum(r["first_partial_tip_clearance_failure"] is not None for r in rows),
        accepted_unknown_types=sum(r["success"] and r.get("actual_route_type") is None for r in records),
        accepted_with_over_type=sum(r["success"] and r.get("actual_route_type") is not None and "over" in r["actual_route_type"] for r in records),
        accepted_routes_with_any_over_crossing=sum(r["success"] and any(c["passage_relation"]=="over" for c in r["actual_row_crossings"]) for r in rows),
        maximum_accepted_guide_endpoint_error_m=max((s["endpoint_error_m"] for r in rows if r["success"] for s in r["guide_segment_diagnostics"]),default=None),
        maximum_accepted_segment_deviation_m=max((s["maximum_deviation_from_straight_guide_segment_m"] for r in rows if r["success"] for s in r["guide_segment_diagnostics"]),default=None),
        strict_restore_passes=sum(r.get("restore",{}).get("max_abs")==0 and r.get("restore",{}).get("rgb_max_difference")==0 and
            r.get("restore",{}).get("global_inventory_equal",False) and bool(r.get("observation_restore")) and
            all(r.get("observation_restore",{}).values()) for r in records),
        generated_robot_validity=None,route_type_total=None,
        inference="All registered attempts are retained. Partial-tip AABB intersections are diagnostic only; arm_environment does not identify a particular robot link or external body.")
    if baseline is not None:
        result["cross_version_initial_comparison"]=compare_initials(data,baseline)
    collector.write_json(output/"analysis.json", result)
    collector.write_json(output/"all_attempt_diagnostics.json", rows)
    if not paths:
        return result
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(target_indices),2,figsize=(13,5*len(target_indices)),squeeze=False)
    palette = plt.get_cmap("tab10")
    all_xyz=np.concatenate(list(paths.values()))
    goals=np.asarray(config["goal_xyz"])
    for target_row,target in enumerate(target_indices):
        identifier=parent.name+"_target%d" % target
        for column,(horizontal,vertical) in enumerate(((0,1),(0,2))):
            axis=axes[target_row,column]
            for center,half in zip(centers,halves):
                axis.add_patch(Rectangle((center[horizontal]-half[horizontal],center[vertical]-half[vertical]),
                    2*half[horizontal],2*half[vertical],facecolor="gray",alpha=.35))
            for row in rows:
                if row["id"]!=identifier or (identifier,row["attempt"]) not in paths:continue
                xyz=paths[(identifier,row["attempt"])]
                color=palette(row["attempt"])
                axis.plot(xyz[:,horizontal],xyz[:,vertical],color=color,lw=1.3,alpha=.85,
                    linestyle="-" if row["success"] else "--",label="%d %s"%(row["attempt"],"PASS" if row["success"] else "FAIL"))
                axis.scatter(xyz[-1,horizontal],xyz[-1,vertical],s=20,marker="x",color=color)
            axis.scatter(goals[target,horizontal],goals[target,vertical],s=100,marker="*",c="black",label="goal label")
            axis.set_xlabel("xyz"[horizontal]+" (m)");axis.set_ylabel("xyz"[vertical]+" (m)")
            axis.set_xlim(min(all_xyz[:,horizontal].min(),centers[:,horizontal].min())-.04,max(all_xyz[:,horizontal].max(),goals[:,horizontal].max())+.04)
            axis.set_ylim(min(all_xyz[:,vertical].min(),centers[:,vertical].min()-halves[:,vertical].max())-.04,
                          max(all_xyz[:,vertical].max(),centers[:,vertical].max()+halves[:,vertical].max())+.04)
            axis.set_aspect("equal",adjustable="box");axis.grid(alpha=.2)
            axis.set_title("Target %d — %s projection, all 9 registered slots"%(target,"XY" if column==0 else "XZ"))
            if column==0:axis.legend(fontsize=7,ncol=2)
    fig.suptitle("Two-row pilot: solid=accepted, dashed=failed partial/full trajectory\nGray boxes are projections; end-tip markers are not robot collision contact locations",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/("all_%d_attempts.png"%config["requested_route_proposals"]),dpi=150);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compare-initial-with",type=Path)
    args=parser.parse_args()
    print(json.dumps(analyze(args.data,args.output,args.compare_initial_with)))


if __name__=="__main__":main()
