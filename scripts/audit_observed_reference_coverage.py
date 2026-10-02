"""TRAIN-only saved-prediction audit; no model loading, search, fitting or GPU.

Reference types and physical boxes are evaluation-only. Interpolated references
are diagnostic probes, never added to the K4 prediction pool. Unknown types are
not equivalent to one another and do not become invented distinct types.
"""
import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_observed_obstacles import scene_metrics, PROTOCOL
from scripts.export_observation_roles import selected_rows, digest

ALPHAS = (.25, .5, .75)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def constant_event_reference(poses, opened, horizon):
    """Exact constant-phase specialization of training resample_event_segments.

Refuse event-changing references instead of silently changing their supervision.
The test compares against that actual source function without importing torch.
"""
    xyz = np.asarray(poses, dtype=np.float64)[:, :3]
    opened = np.asarray(opened).reshape(-1) > .5
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or
            len(xyz) != len(opened) or not np.isfinite(xyz).all() or
            not np.all(opened == opened[0])):
        raise ValueError('finite constant-event reach reference required')
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    if distance[-1] < 1e-10:
        sampled = np.repeat(xyz[:1], horizon, axis=0)
    else:
        query = np.linspace(0., distance[-1], horizon)
        sampled = np.stack([np.interp(query, distance, xyz[:, axis]) for axis in range(3)], axis=1)
        sampled[0], sampled[-1] = xyz[0], xyz[-1]
    return sampled.astype(np.float32), np.full(horizon, opened[0], dtype=np.float32)


def assignment_audit(cost):
    """Exact saturation optimum and second distinct assignment, diagnostic only.

M<K covers every reference and lets remaining candidates use any reference.
M>=K gives K distinct references. Matches train_v2.positive_assignment_loss.
"""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != 4 or not np.isfinite(cost).all():
        raise ValueError('finite K4 by M assignment cost required')
    k, m = cost.shape
    if not m:
        return None
    if m < k:
        options = [(float(cost[np.arange(k), indices].mean()), tuple(indices))
                   for indices in itertools.product(range(m), repeat=k)
                   if len(set(indices)) == m]
        options.sort()
        best, chosen = options[0]
        runner_up = options[1][0] if len(options) > 1 else None
    else:
        rows, columns = linear_sum_assignment(cost)
        chosen = tuple(map(int, columns[np.argsort(rows)]))
        best = float(cost[np.arange(k), chosen].mean())
        alternatives = []
        for row, column in enumerate(chosen):
            excluded = cost.copy(); excluded[row, column] = np.inf
            other_rows, other_columns = linear_sum_assignment(excluded)
            alternatives.append(float(cost[other_rows, other_columns].mean()))
        runner_up = min(alternatives)
    return dict(reference_indices=list(chosen), best_cost=best,
                second_assignment_cost=runner_up,
                second_minus_best=None if runner_up is None else runner_up-best,
                scope='continuous MSE ambiguity; not a route-type or validity label')


def passage(value):
    return None if value is None else tuple(value)


def coverage_audit(reference_rows, candidate_rows):
    reference_types = {passage(row['declared_passage_type']) for row in reference_rows
                       if row['TipValid'] and row['declared_passage_type'] is not None}
    valid_types = Counter(passage(row['declared_passage_type']) for row in candidate_rows
                          if row['TipValid'] and row['declared_passage_type'] is not None)
    missing = reference_types-set(valid_types)
    duplicates = sum(valid_types.values())-len(valid_types)
    return dict(known_valid_reference_types=[list(t) for t in sorted(reference_types)],
                valid_generated_types=[list(t) for t in sorted(valid_types)],
                missing_supported_known_types=[list(t) for t in sorted(missing)],
                valid_classified_duplicate_candidates=duplicates,
                unknown_valid_reference_count=sum(r['TipValid'] and r['declared_passage_type'] is None for r in reference_rows),
                unknown_valid_candidate_count=sum(r['TipValid'] and r['declared_passage_type'] is None for r in candidate_rows),
                duplicate_replacement_reference_oracle_gain=min(duplicates, len(missing)),
                oracle_scope='replace classified duplicate with known positive; label-assisted opportunity, not achievable model gain')


def validate_predictions(ids, parents, paths, events, rows, horizon):
    expected = {r['id']: r['parent_id'] for r in rows}
    if (len(ids) != len(set(ids)) or set(ids) != set(expected) or len(parents) != len(ids) or
            paths.shape != (len(rows), 4, horizon, 3) or events.shape != paths.shape[:-1]):
        raise ValueError('all TRAIN inputs including no-reference rows and exact K4/H budget required')
    if any(parent != expected[identifier] for identifier, parent in zip(ids, parents)):
        raise ValueError('prediction parent mismatch')
    return {identifier: index for index, identifier in enumerate(ids)}


def checked_file(data, value, parent, source, hashes, recorded=None):
    path = (data/value).resolve()
    if path.parent != source.resolve()/parent:
        raise ValueError('raw path escaped explicitly selected TRAIN parent')
    hashes[str(path)] = digest(path)
    if recorded is not None and recorded.get(str(path)) != hashes[str(path)]:
        raise ValueError('TRAIN raw artifact changed since training: '+str(path))
    return path


def audit_scene(paths, events, refs, ref_events, current, geometry, label, event_scale):
    tip, candidates = scene_metrics(paths, events, current, geometry, label['semantic_targets'],
                                   label.get('route_types', []), clearance=.02)
    reference_rows = []
    for path, event in zip(refs, ref_events):
        _, details = scene_metrics(path[None], event[None], current, geometry,
                                   label['semantic_targets'], [], clearance=.02)
        reference_rows.append(details[0])
    result = dict(reference_count=len(refs), tip_metrics=tip, candidates=candidates,
                  references=reference_rows, coverage=coverage_audit(reference_rows, candidates),
                  assignment=None, nearest_reference_ambiguity=[], interpolation_probes=[])
    if not len(refs):
        return result
    # Same start exclusion, event scale and per-coordinate average as the trainer.
    predicted = np.concatenate([paths[:, 1:], events[:, 1:, None]*event_scale], -1)
    targets = np.concatenate([refs[:, 1:], ref_events[:, 1:, None]*event_scale], -1)
    costs = np.square(predicted[:, None]-targets[None]).mean((-1, -2))
    if np.isfinite(costs).all():
        result['assignment'] = assignment_audit(costs)
        result['assignment']['cost_matrix'] = costs.tolist()
        for row in costs:
            order = np.argsort(row, kind='stable')
            types = [passage(reference_rows[i]['declared_passage_type']) for i in order[:2]]
            result['nearest_reference_ambiguity'].append(dict(nearest_indices=order[:2].tolist(),
                cost_gap=float(row[order[1]]-row[order[0]]) if len(order)>1 else None,
                different_known_types=len(types)==2 and None not in types and types[0]!=types[1]))
    for a, b in itertools.combinations(range(len(refs)), 2):
        ta, tb = [passage(reference_rows[i]['declared_passage_type']) for i in (a,b)]
        relation = 'unknown' if ta is None or tb is None else ('same_known' if ta==tb else 'different_known')
        for alpha in ALPHAS:
            path = (1-alpha)*refs[a]+alpha*refs[b]
            event = (1-alpha)*ref_events[a]+alpha*ref_events[b]
            _, details = scene_metrics(path[None], event[None], current, geometry,
                                       label['semantic_targets'], [], clearance=.02)
            result['interpolation_probes'].append(dict(reference_indices=[a,b],alpha=alpha,
                reference_type_relation=relation,both_references_valid=all(reference_rows[i]['TipValid'] for i in (a,b)),
                tip_check=details[0]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reservation',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-prediction-sha256',required=True)
    args = p.parse_args(); started = time.perf_counter()
    if args.output.exists(): raise FileExistsError('new audit output required')
    config, summary = read(args.run/'config.json'), read(args.run/'summary.json')
    recorded = read(args.run/'source_hashes.json')
    if (config.get('refinement_mode','none')!='none' or config['anchor_mode']!='straight_through_peak' or
            config['candidates']!=4 or config['horizon']!=24 or config['objective']!='saturation' or
            read(args.run/'status.json')['status']!='completed'):
        raise ValueError('completed ordinary peak K4/H24 saturation source required')
    if digest(args.run/'best.pt')!=summary['best_checkpoint_sha256']:
        raise ValueError('original best checkpoint changed')
    if Path(config['observations']).resolve()!=(args.data/'observations.jsonl').resolve():
        raise ValueError('source dataset mismatch')
    exported = read(args.data/'export_manifest.json'); reservation = read(args.reservation)
    if digest(args.reservation)!=exported['reservation_sha256']:
        raise ValueError('prospective reservation changed')
    source = Path(exported['source_dataset'])
    record = [r for r in reservation['collections'] if Path(r['source']).name==source.name]
    if len(record)!=1: raise ValueError('unique matching parent reservation required')
    record=record[0];lo,hi=record['roles_inclusive']['TRAIN']
    parents={record['parent_prefix']+str(i) for i in range(lo,hi+1)}
    if not parents.issubset(set(exported['requested_parent_ids'])):
        raise ValueError('all reserved TRAIN parents must be in export')
    hashes={}
    for filename,key in [('observations.jsonl','input_manifest_sha256'),('supervision.jsonl','supervision_manifest_sha256')]:
        path=args.data/filename;hashes[str(path)]=digest(path)
        if hashes[str(path)]!=exported[key] or recorded.get(str(path))!=hashes[str(path)]:
            raise ValueError('original training manifest changed')
    rows=selected_rows(args.data/'observations.jsonl',parents)
    label_rows=selected_rows(args.data/'supervision.jsonl',parents);labels={r['id']:r for r in label_rows}
    if (len(rows)!=3*len(parents) or len(labels)!=len(label_rows) or set(labels)!={r['id'] for r in rows} or
            {r['parent_id'] for r in rows}!=parents): raise ValueError('full TRAIN identity coverage required')
    for row in rows:
        label=labels[row['id']]
        if (set(row)!={'id','parent_id','split','image','instruction'} or row['split']!='TRAIN' or
                label['split']!='TRAIN' or label['parent_id']!=row['parent_id'] or
                label['semantic_targets']['tolerance']!=.03): raise ValueError('TRAIN identity or fixed semantic standard violated')
    if read(source/'manifest.json')['acceptance']['tip_polyline_clearance_m']!=.02:
        raise ValueError('fixed clearance standard violated')
    prediction=args.run/'train/predictions.npz'
    if digest(prediction)!=args.expected_prediction_sha256: raise ValueError('saved TRAIN prediction changed')
    with np.load(prediction,allow_pickle=False) as archive:
        ids=list(map(str,archive['scene_ids']));pp=list(map(str,archive['parent_ids']))
        paths,events=archive['paths'],archive['gripper_open']
    order=validate_predictions(ids,pp,paths,events,rows,24)
    # Prediction pool fully fixed and identity-checked before physical labels open.
    results=[]
    for row in rows:
        label=labels[row['id']];parent=row['parent_id'];refs=[];ref_events=[]
        file=checked_file(args.data,label['observation'],parent,source,hashes,recorded)
        with np.load(file,allow_pickle=False) as archive:
            current={k:archive[k].astype(np.float32) for k in ('gripper_pose','gripper_open')}
        file=checked_file(args.data,label['verification_only'],parent,source,hashes)
        with np.load(file,allow_pickle=False) as archive:
            geometry={k:archive[k] for k in ('obstacle_centers','obstacle_halfsizes')}
        for filename in label['routes']:
            file=checked_file(args.data,filename,parent,source,hashes,recorded)
            with np.load(file,allow_pickle=False) as archive:
                xyz,opened=constant_event_reference(archive['gripper_pose'],archive['gripper_open'],24)
                if not np.allclose(xyz,archive['xyz_24'],atol=1e-6,rtol=0):
                    raise ValueError('constant-phase specialization differs from saved collector reference')
            refs.append(xyz);ref_events.append(opened)
        index=order[row['id']]
        result=audit_scene(paths[index],events[index],np.asarray(refs),np.asarray(ref_events),current,geometry,label,config['event_scale'])
        if [passage(r['declared_passage_type']) for r in result['references']]!=[passage(t) for t in label['route_types']]:
            raise ValueError('recomputed H24 reference types differ from collection')
        result.update(scene_id=row['id'],parent_id=parent);results.append(result)
    candidates=[c for r in results for c in r['candidates']];references=[c for r in results for c in r['references']]
    probes=[c for r in results for c in r['interpolation_probes']]
    output=dict(protocol='observed_train_reference_coverage_audit_v1',tip_evaluation_protocol=PROTOCOL,
        split='TRAIN',parents=len(parents),observations=len(rows),submitted_candidate_slots=len(candidates),
        reference_count_histogram=dict(Counter(str(r['reference_count']) for r in results)),
        references=len(references),valid_references=sum(r['TipValid'] for r in references),
        known_reference_type_frequency=dict(Counter('|'.join(r['declared_passage_type']) for r in references if r['declared_passage_type'] is not None)),
        unknown_references=sum(r['declared_passage_type'] is None for r in references),
        same_scene_known_duplicate_reference_count=sum(sum(t['declared_passage_type'] is not None for t in r['references'])-len({passage(t['declared_passage_type']) for t in r['references'] if t['declared_passage_type'] is not None}) for r in results),
        TipValidAtK=float(np.mean([r['TipValid'] for r in candidates])),
        UniqueClassifiedTipValidAtK=float(np.mean([r['tip_metrics']['UniqueClassifiedTipValidAtK'] for r in results])),
        duplicate_valid_candidates=sum(r['coverage']['valid_classified_duplicate_candidates'] for r in results),
        duplicate_and_supported_missing_scenes=sum(r['coverage']['duplicate_replacement_reference_oracle_gain']>0 for r in results),
        duplicate_replacement_reference_oracle_gain=sum(r['coverage']['duplicate_replacement_reference_oracle_gain'] for r in results),
        interpolation=dict(alphas=ALPHAS,probes=len(probes),candidate_pool_use=False,
            groups={kind:dict(probes=sum(p['reference_type_relation']==kind and p['both_references_valid'] for p in probes),
                colliding=sum(p['reference_type_relation']==kind and p['both_references_valid'] and not p['tip_check']['tip_segments_clear'] for p in probes)) for kind in ('different_known','same_known','unknown')}),
        by_reference_count={str(n):dict(candidates=sum(r['reference_count']==n for r in results)*4,
            collisions=sum(not c['tip_segments_clear'] for r in results if r['reference_count']==n for c in r['candidates']),
            semantic_correct_collisions=sum(c['semantic_goal_correct'] and not c['tip_segments_clear'] for r in results if r['reference_count']==n for c in r['candidates'])) for n in sorted({r['reference_count'] for r in results})},
        source=dict(run=str(args.run),best_step=summary['best_step'],checkpoint_sha256=summary['best_checkpoint_sha256'],
            prediction_sha256=digest(prediction),config_sha256=digest(args.run/'config.json'),training_source_commit=config['code_commit']),
        source_commit=os.environ.get('CODE_COMMIT'),script_sha256=digest(__file__),
        gpu_hours=0,model_forward_calls=0,raw_dev_or_locked_files_opened=False,
        limitations='Single development-trained source. Unknown types are not grouped or counted as new. Interpolation collisions and matching ambiguity are associations, not training-causality evidence. Oracle reference replacement is privileged diagnostic only. Tip checks exclude full arm, table, IK and execution.',
        cpu_wall_s=time.perf_counter()-started)
    args.output.mkdir(parents=True)
    write(args.output/'summary.json',output);write(args.output/'per_scene.json',results)
    write(args.output/'input_hashes.json',hashes)
    script_root=Path(__file__).resolve().parents[1]
    names=['scripts/audit_observed_reference_coverage.py','scripts/evaluate_observed_obstacles.py','scripts/collect_obstacle_reach.py','scripts/export_observation_roles.py','scripts/observation_collect_rlbench.py','routeset/observed_route_head.py']
    write(args.output/'source_hashes.json',{name:digest(script_root/name) for name in names})
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
