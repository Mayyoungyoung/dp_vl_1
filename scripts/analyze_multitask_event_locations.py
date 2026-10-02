"""Saved-output diagnostic: event location, without changing checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_close(opened):
    states = np.asarray(opened) > .5
    transitions = np.flatnonzero(states[:-1] & ~states[1:]) + 1
    return int(transitions[0]) if len(transitions) else None


def compare(paths, opened, references, events):
    """Match by existing path ADE, never by favorable event position."""
    paths, opened, references, events = map(np.asarray, (paths, opened, references, events))
    if paths.ndim != 3 or paths.shape[-1] != 3 or opened.shape != paths.shape[:2]:
        raise ValueError('Candidate path/event shape mismatch')
    if not all(np.isfinite(x).all() for x in (paths, opened, references, events)):
        raise ValueError('Nonfinite saved predictions must not be silently removed')
    if len(references) and (references.shape[1:] != paths.shape[1:] or events.shape != references.shape[:2]):
        raise ValueError('Reference path/event shape mismatch')
    nearest = (np.linalg.norm(paths[:, None]-references[None], axis=-1).mean(-1).argmin(1)
               if len(references) else [None] * len(paths))
    result = []
    for slot, ref in enumerate(nearest):
        pred_index = first_close(opened[slot])
        ref_index = first_close(events[ref]) if ref is not None else None
        expected = ref_index is not None
        distance = (float(np.linalg.norm(paths[slot, pred_index]-references[ref, ref_index]))
                    if expected and pred_index is not None else None)
        result.append(dict(slot=slot,reference_index=int(ref) if ref is not None else None,
            predicted_close_index=pred_index,reference_close_index=ref_index,
            reference_available=ref is not None,reference_has_close=expected if ref is not None else None,
            presence_match=(expected == (pred_index is not None)) if ref is not None else None,
            close_location_error_m=distance,
            close_within_3cm=(distance is not None and distance <= .03) if expected else None,
            absolute_close_index_error=abs(pred_index-ref_index) if distance is not None else None))
    return result


def mean_present(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def validate_prediction_shapes(saved, examples, candidates, horizon):
    expected = dict(paths=(examples,candidates,horizon,3), gripper_open=(examples,candidates,horizon),
                    scene_ids=(examples,), parent_ids=(examples,))
    if any(np.asarray(saved[key]).shape != shape for key,shape in expected.items()):
        raise ValueError('Exact saved scene/candidate/horizon budget required')


def grouped_metrics(rows):
    fields = ('presence_match','close_location_error_m','close_within_3cm','absolute_close_index_error')
    tasks = {}
    for task in sorted({r['task'] for r in rows}):
        parents = {}
        for parent in sorted({r['parent_id'] for r in rows if r['task'] == task}):
            local = [r for r in rows if r['parent_id'] == parent and r['task'] == task]
            parents[parent] = {key:mean_present([mean_present([c[key] for c in r['candidates']]) for r in local]) for key in fields}
        tasks[task] = {key:mean_present([r[key] for r in parents.values()]) for key in fields}
        tasks[task]['parents'] = parents
    return dict(per_task=tasks,macro={key:mean_present([r[key] for r in tasks.values()]) for key in fields})


def run(ordinary, auxiliary, output):
    if output.exists():
        raise FileExistsError('Fresh analysis output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('CPU only diagnostic must hide GPUs')
    import torch
    from routeset.observed_route_head import resample_event_segments
    torch.set_num_threads(1)
    started = time.perf_counter()
    configs = [json.loads((p/'config.json').read_text()) for p in (ordinary, auxiliary)]
    common = ('observations','supervision','cache_dir','horizon','candidates','seed','dataset_fingerprint',
              'steps','batch_size','endpoint_mode','checkpoint_selection','metric_aggregation')
    if any(configs[0][key] != configs[1][key] for key in common):
        raise ValueError('Same data and registered control required')
    config = configs[0]
    if config['endpoint_mode'] != 'free_offset' or config['checkpoint_selection'] != 'reference_ADE':
        raise ValueError('Generic reference-only diagnostic required')
    sources = json.loads((ordinary/'source_hashes.json').read_text())
    observed_path, label_path = Path(config['observations']), Path(config['supervision'])
    for path in (observed_path, label_path):
        if sources.get(str(path)) != sha(path):
            raise ValueError('Original manifest changed')
    # Identity/role filtering precedes opening reference route content.
    observed = [json.loads(line) for line in observed_path.read_text().splitlines()]
    labels = {r['id']:r for r in map(json.loads,label_path.read_text().splitlines())}
    dev = {r['id']:r for r in observed if r['split'] == 'DEV_MODEL'}
    if len(dev) != 48 or len({r['parent_id'] for r in dev.values()}) != 12:
        raise ValueError('Unchanged twelve-parent DEV required')
    references, used = {}, {str(observed_path):sha(observed_path),str(label_path):sha(label_path)}
    for identifier, obs in dev.items():
        label = labels[identifier]
        if label['split'] != 'DEV_MODEL' or label['parent_id'] != obs['parent_id']:
            raise ValueError('Reference identity/role mismatch')
        paths, events = [], []
        for name in label['routes']:
            path = (label_path.parent/name).resolve()
            if sources.get(str(path)) != sha(path):
                raise ValueError('Original positive reference changed')
            used[str(path)] = sha(path)
            with np.load(path,allow_pickle=False) as archive:
                xyz, state = resample_event_segments(archive['gripper_pose'],archive['gripper_open'],config['horizon'])
            paths.append(xyz);events.append(state)
        references[identifier] = (np.asarray(paths),np.asarray(events))
    results = {}
    for name, directory in (('ordinary',ordinary),('auxiliary',auxiliary)):
        summary = json.loads((directory/'summary.json').read_text())
        used[str(directory/'summary.json')] = sha(directory/'summary.json')
        for selection, sub, hash_key in (('best','dev_model','prediction_sha256'),('last','last_dev_model','last_prediction_sha256')):
            path = directory/sub/'predictions.npz'
            if sha(path) != summary[hash_key]:
                raise ValueError('Saved prediction changed')
            used[str(path)] = sha(path)
            with np.load(path,allow_pickle=False) as saved:
                validate_prediction_shapes(saved,len(dev),config['candidates'],config['horizon'])
                ids = saved['scene_ids'].astype(str).tolist()
                if len(ids) != len(set(ids)) or set(ids) != set(dev):
                    raise ValueError('Every original DEV instruction required')
                rows = []
                for i, identifier in enumerate(ids):
                    if saved['parent_ids'][i] != dev[identifier]['parent_id']:
                        raise ValueError('Saved prediction parent mismatch')
                    ref, event = references[identifier]
                    rows.append(dict(scene_id=identifier,parent_id=dev[identifier]['parent_id'],task=labels[identifier]['task'],
                        candidates=compare(saved['paths'][i],saved['gripper_open'][i],ref,event)))
            candidates = [c for r in rows for c in r['candidates']]
            results[name+'_'+selection] = dict(rows=rows,**grouped_metrics(rows),
                total_slots=len(candidates),reference_available_slots=sum(c['reference_available'] for c in candidates),
                expected_close_slots=sum(c['reference_has_close'] is True for c in candidates),
                expected_close_missing_prediction=sum(c['reference_has_close'] is True and c['predicted_close_index'] is None for c in candidates),
                unexpected_close_slots=sum(c['reference_has_close'] is False and c['predicted_close_index'] is not None for c in candidates))
    output.mkdir(parents=True)
    report=dict(status='completed',protocol='multitask_saved_first_close_location_v1',code_commit=os.environ.get('CODE_COMMIT'),
        results=results,input_sha256=used,script_sha256=sha(__file__),cpu_seconds=time.perf_counter()-started,
        scope='Post-selection descriptive DEV diagnosis. Same ADE-nearest positive reference rule; no new predictions, refitting or checkpoint selection. '
              'Conditional location error excludes missing predicted closes but missing counts and within3cm denominator retain them. '
              'Presence covers all reference-available slots. 3cm is a descriptive location tolerance, not contact, semantics or task success. '
              'Macro language-within-parent, parent-within-task, across-evaluable-tasks; no-close tasks have null location metrics.')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['macro'] for k,v in results.items()}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ordinary-run',type=Path,required=True)
    parser.add_argument('--auxiliary-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.ordinary_run,args.auxiliary_run,args.output)
