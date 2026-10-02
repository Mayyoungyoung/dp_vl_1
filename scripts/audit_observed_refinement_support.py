"""Post-hoc visible-point support at saved draft prefixes; no safety claim.

Reads only current observed RGB-D/camera and saved predictions. Does not open
verification boxes, semantic targets, segmentation masks or reference routes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from scripts.export_observation_roles import selected_rows,digest


def support_at_draft(draft, final, xyz, mask, sigma, prefix_fraction):
    draft,final,xyz=np.asarray(draft),np.asarray(final),np.asarray(xyz)
    mask=np.asarray(mask,dtype=bool)
    if draft.shape!=final.shape or draft.ndim!=3 or draft.shape[-1]!=3 or draft.shape[1]<3:
        raise ValueError('matching saved K,H,3 draft and final paths required')
    if sigma<=0 or not 0<prefix_fraction<=1 or mask.shape!=xyz.shape[:1] or not mask.any():
        raise ValueError('valid mask and configured positive scales required')
    if not np.isfinite(draft).all() or not np.isfinite(final).all() or not np.isfinite(xyz[mask]).all():
        raise ValueError('finite submitted paths and valid observations required')
    if not np.array_equal(draft[:,[0,-1]],final[:,[0,-1]]):
        raise ValueError('start or endpoint changed within draft update')
    lengths=np.linalg.norm(np.diff(draft,axis=1),axis=-1)
    fractions=np.cumsum(lengths,axis=-1)[:,:-1]/np.maximum(lengths.sum(-1,keepdims=True),1e-8)
    eligible=fractions<=prefix_fraction
    # Float32 roundoff at the exact eligibility boundary is reported explicitly;
    # it never changes stored paths or the model's training/inference decision.
    borderline=np.abs(fractions-prefix_fraction)<=1e-6
    if not np.array_equal(draft[:,1:-1][~eligible & ~borderline],final[:,1:-1][~eligible & ~borderline]):
        raise ValueError('non-prefix point changed outside boundary roundoff band')
    tree=cKDTree(xyz[mask]);query=draft[:,1:-1].reshape(-1,3)
    nearest=tree.query(query,k=1)[0].reshape(eligible.shape)
    inside=tree.query_ball_point(query,sigma,return_length=True).reshape(eligible.shape)
    double=tree.query_ball_point(query,2*sigma,return_length=True).reshape(eligible.shape)
    delta=final[:,1:-1]-draft[:,1:-1]
    records=[]
    for candidate in range(len(draft)):
        for waypoint in range(draft.shape[1]-2):
            records.append(dict(candidate=candidate,waypoint_index=waypoint+1,
                draft_normalized_arc=float(fractions[candidate,waypoint]),eligible=bool(eligible[candidate,waypoint]),
                boundary_roundoff_band=bool(borderline[candidate,waypoint]),
                nearest_visible_point_m=float(nearest[candidate,waypoint]),
                visible_points_within_sigma=int(inside[candidate,waypoint]),
                visible_points_within_2sigma=int(double[candidate,waypoint]),
                actual_update_l2_m=float(np.linalg.norm(delta[candidate,waypoint])),
                actual_update_coordinate_max_m=float(np.abs(delta[candidate,waypoint]).max())))
    return records


def aggregate(records):
    chosen=[row for row in records if row['eligible']]
    result=dict(all_interior_queries=len(records),eligible_queries=len(chosen),
        boundary_roundoff_queries=sum(row['boundary_roundoff_band'] for row in records))
    for key in ('nearest_visible_point_m','visible_points_within_sigma','visible_points_within_2sigma','actual_update_l2_m','actual_update_coordinate_max_m'):
        values=[row[key] for row in chosen]
        result[key]=dict(min=float(np.min(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(np.max(values))) if values else None
    result.update(eligible_without_points_within_sigma=sum(not row['visible_points_within_sigma'] for row in chosen),
                  eligible_without_points_within_2sigma=sum(not row['visible_points_within_2sigma'] for row in chosen))
    return result


def read_saved_stage(path,by_id,candidates,horizon,expected_sha):
    if digest(path)!=expected_sha:
        raise ValueError('saved prediction SHA differs from original summary')
    with np.load(path,allow_pickle=False) as archive:
        identifiers=list(map(str,archive['scene_ids']));parents=list(map(str,archive['parent_ids']))
        drafts,finals=archive['draft_paths'],archive['paths']
    if (len(identifiers)!=len(set(identifiers)) or set(identifiers)!=set(by_id) or
            drafts.shape!=finals.shape or drafts.shape!=(len(identifiers),candidates,horizon,3)):
        raise ValueError('all DEV and exact saved K/H draft/final shapes required')
    if len(parents)!=len(identifiers) or any(parents[i]!=by_id[identifier]['parent_id'] for i,identifier in enumerate(identifiers)):
        raise ValueError('saved prediction parent identity mismatch')
    return identifiers,drafts,finals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from scripts.train_observed_geometry import read_geometry
    started=time.perf_counter()
    config=json.loads((args.run/'config.json').read_text())
    summary=json.loads((args.run/'summary.json').read_text())
    if config.get('refinement_mode') not in ('local','global'):
        raise ValueError('saved one-update run required')
    observations,supervision=Path(config['observations']),Path(config['supervision'])
    export=json.loads((observations.parent/'export_manifest.json').read_text())
    if set(export['requested_roles'])!={'TRAIN','DEV_MODEL'}:
        raise ValueError('development-only export required')
    if digest(observations)!=export['input_manifest_sha256'] or digest(supervision)!=export['supervision_manifest_sha256']:
        raise ValueError('source export changed')
    parents=set(export['requested_parent_ids'])
    rows=[row for row in selected_rows(observations,parents) if row['split']=='DEV_MODEL']
    chosen_parents={row['parent_id'] for row in rows}
    # Only the current-observation pointer/identity are used from these rows.
    labels={row['id']:row for row in selected_rows(supervision,chosen_parents)}
    clouds={};hashes={};by_id={row['id']:row for row in rows}
    expected_hashes=config['geometry_preprocessing']['source_hashes']
    for row in rows:
        if row['parent_id'] in clouds:continue
        image=(observations.parent/row['image']).resolve()
        observation=(supervision.parent/labels[row['id']]['observation']).resolve()
        for path in (image,observation):
            hashes[str(path)]=digest(path)
            if hashes[str(path)]!=expected_hashes[str(path)]:raise ValueError('observed source changed')
        clouds[row['parent_id']]=read_geometry(image,observation,config['pixel_stride'])
    result=dict(protocol='observed_draft_visible_support_v1',refinement_mode=config['refinement_mode'],
        sigma_m=config['refinement_sigma'],prefix_arc_fraction=config['refinement_prefix_fraction'],
        coordinate_update_bound_m=config['refinement_bound'],source_hashes=hashes,
        scope='Only sampled visible points at the original pixel stride; low distance/high count does not certify free space, hidden solids, collisions or robot feasibility.',
        label_use='No verification, target, segmentation or reference trajectory accessed',stages={})
    for stage,folder in (('best','dev_model'),('last','last_dev_model')):
        file=args.run/folder/'predictions.npz'
        expected_sha=summary['prediction_sha256' if stage=='best' else 'last_prediction_sha256']
        identifiers,drafts,finals=read_saved_stage(file,by_id,config['candidates'],config['horizon'],expected_sha)
        records=[]
        for identifier,draft,final in zip(identifiers,drafts,finals):
            row=by_id[identifier];cloud=clouds[row['parent_id']]
            queries=support_at_draft(draft,final,cloud['world_xyz'],cloud['valid_mask'],
                config['refinement_sigma'],config['refinement_prefix_fraction'])
            records.extend(dict(scene_id=identifier,parent_id=row['parent_id'],**query) for query in queries)
        result['stages'][stage]=dict(aggregate=aggregate(records),all_queries=records,prediction_sha256=digest(file))
    result.update(cpu_wall_s=time.perf_counter()-started,config_sha256=digest(args.run/'config.json'),script_sha256=digest(__file__))
    if args.output.exists():raise ValueError('fresh output required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:stage['aggregate'] for key,stage in result['stages'].items()},indent=2))


if __name__=='__main__':main()
