"""TRAIN-only direct-feature intervention and zero-search observed-graph audit.

Counterfactual pools are not normal task predictions and are never MAIN results.
Only registered observation/current/cache fields reach the frozen model. All 36
head calls and all 12 observed graphs are sealed before reference/box access.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np

PROTOCOL = 'two_row_train12_direct_route_conditioning_v1'
ROOT = Path(__file__).resolve().parents[1]
PARENTS = ['two_row_reach_%d' % s for s in range(283200, 283204)]
IDS = [p+'_target%d' % t for p in PARENTS for t in range(3)]
ARMS = ('normal', 'identity', 'cyclic_direct_swap')
INPUT_KEYS = {'id', 'parent_id', 'split', 'image', 'instruction'}
CURRENT_KEYS = {'depth', 'camera_intrinsics', 'camera_extrinsics', 'gripper_pose', 'gripper_open'}
MODEL_SOURCES = ('routeset/observed_geometry.py', 'routeset/observed_route_head.py',
    'routeset/models.py', 'routeset/common.py', 'routeset/observed_path_refinement.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def selected_rows(path):
    """Decode only ID-matched rows, including for the separate label table."""
    selected = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            match = re.search(r'"id"\s*:\s*("(?:[^"\\]|\\.)*")', line)
            if match is None: raise ValueError('Missing JSONL identity')
            identifier = json.loads(match.group(1))
            if identifier not in IDS: continue
            row = json.loads(line)
            if (identifier in selected or row.get('split') != 'TRAIN'
                    or row.get('parent_id') != identifier.rsplit('_target', 1)[0]):
                raise ValueError('Fixed TRAIN identity/role mismatch')
            selected[identifier] = row
    if set(selected) != set(IDS): raise ValueError('All twelve fixed TRAIN inputs required; no replacement')
    return [selected[key] for key in IDS]


def donor_id(identifier):
    if identifier not in IDS: raise ValueError('Unregistered intervention input')
    parent, target = identifier.rsplit('_target', 1)
    return parent+'_target%d' % ((int(target)+1) % 3)


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


@contextmanager
def direct_override(module, replacement, capture):
    """Only the direct feature branch; remove this hook even after exceptions."""
    calls = []
    def hook(unused_module, unused_inputs, value):
        calls.append(1)
        capture.append(value.detach().clone())
        if replacement is None: return value
        if value.shape != replacement.shape or value.dtype != replacement.dtype or value.device != replacement.device:
            raise ValueError('Replacement direct feature contract changed')
        return replacement.detach().clone()
    handle = module.register_forward_hook(hook)
    try:
        yield calls
        if len(calls) != 1: raise ValueError('Exactly one direct feature branch call required')
    finally:
        handle.remove()


def exact_geometry(first, second):
    import torch
    if set(first) != set(second): raise ValueError('Geometry keys changed under direct intervention')
    for key in first:
        if not torch.equal(first[key], second[key]):
            raise ValueError('Geometry/anchor changed under direct intervention: '+key)


def clamp_endpoint(paths, normal):
    value = paths.copy()
    if value.shape != (4, 24, 3) or normal.shape != value.shape: raise ValueError('Exact K4/H24 required')
    value[:, -1] = normal[:, -1]
    return value


def displacement(normal, changed):
    """No alignment/reordering. Prefix is each normal path's <=50% arc length."""
    result = []
    for k, (before, after) in enumerate(zip(normal, changed)):
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(before, axis=0), axis=-1))]
        fraction = arc/arc[-1] if arc[-1] else np.zeros_like(arc)
        distances = np.linalg.norm(after-before, axis=-1)
        body = np.arange(24)[1:-1]
        prefix = body[fraction[body] <= .5]
        result.append(dict(candidate=k, per_vertex_distance_m=distances.tolist(),
            normal_arc_fraction=fraction.tolist(), prefix_vertex_indices=prefix.tolist(),
            prefix_mean_m=float(distances[prefix].mean()) if len(prefix) else None,
            prefix_max_m=float(distances[prefix].max()) if len(prefix) else None,
            full_body_mean_m=float(distances[body].mean()), full_body_max_m=float(distances[body].max()),
            fixed_start_distance_m=float(distances[0]), fixed_endpoint_distance_m=float(distances[-1])))
    return result


def compare_saved_normal(pools, saved_path):
    """Original already-sealed TRAIN prediction pool only, never a new forward."""
    with np.load(saved_path,allow_pickle=False) as archive:
        ids=list(map(str,archive['scene_ids']))
        if len(ids)!=len(set(ids)) or not set(IDS)<=set(ids): raise ValueError('Saved TRAIN pool identity mismatch')
        expected={p+'_target%d'%t for p in ['two_row_reach_%d'%s for s in range(283200,283264)] for t in range(3)}
        if not set(ids)<=expected or len(ids)!=189: raise ValueError('Only original 189 TRAIN prediction pool accepted')
        records=[]
        for identifier in IDS:
            index=ids.index(identifier); paths,events=pools[('normal',identifier)]
            delta_xyz=float(np.max(np.abs(paths-archive['paths'][index])))
            delta_open=float(np.max(np.abs(events-archive['gripper_open'][index])))
            if not np.isfinite(delta_xyz+delta_open) or max(delta_xyz,delta_open)>1e-5:
                raise ValueError('Normal output differs from original saved TRAIN pool beyond1e-5: '+identifier)
            records.append(dict(id=identifier,max_absolute_xyz_difference_m=delta_xyz,max_absolute_event_difference=delta_open))
    return dict(passed=True,absolute_tolerance=1e-5,relative_tolerance=0,records=records,
        input_prediction_sha256=digest(saved_path),new_forward_requests=0)


def checked(path, expected, hashes):
    path = Path(path)
    actual = digest(path)
    if expected.get(str(path)) != actual: raise ValueError('Registered source changed: '+str(path))
    hashes[str(path)] = actual
    return path


def load_inputs(row, manifest, config, hashes, device, expected_cache):
    import torch
    from PIL import Image
    from routeset.observed_geometry import backproject_rgbd
    from routeset.observed_route_head import CACHE_KEYS
    if set(row) != INPUT_KEYS or row['id'] not in IDS or row['split'] != 'TRAIN':
        raise ValueError('Observation-only fixed TRAIN whitelist required')
    image = checked(row['image'], manifest['source_files_sha256'], hashes)
    pointer = checked(image.with_name('observation.npz'), manifest['source_files_sha256'], hashes)
    if image.parent.name != row['parent_id'] or image.name != 'front.png': raise ValueError('Current image identity changed')
    rgb = np.asarray(Image.open(image).convert('RGB'))
    with np.load(pointer, allow_pickle=False) as archive:
        if set(archive.files) != CURRENT_KEYS: raise ValueError('Current input contains undeclared fields')
        observed = {key: archive[key].copy() for key in archive.files}
    key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    cache = Path(config['cache_dir'])/(key+'.npz')
    if cache.name != Path(expected_cache['relative']).name or digest(cache) != expected_cache['sha256']:
        raise ValueError('Predeclared twelve-input cache SHA changed')
    hashes[str(cache)] = digest(cache)
    with np.load(cache, allow_pickle=False) as archive:
        if set(archive.files) != CACHE_KEYS: raise ValueError('Cache whitelist differs')
        if any(str(archive[k].item()) != row[k] for k in ('id', 'parent_id', 'split')):
            raise ValueError('Cache identity differs')
        if str(archive['image_sha256'].item()) != digest(image): raise ValueError('Cache image differs')
        features = np.concatenate([archive[k].reshape(-1) for k in ('mean_hidden', 'last_hidden')]).astype(np.float32)
    current = np.r_[observed['gripper_pose'].reshape(7), observed['gripper_open'].reshape(1)].astype(np.float32)
    points = backproject_rgbd(rgb, observed['depth'], observed['camera_intrinsics'], observed['camera_extrinsics'], pixel_stride=2)
    batch = {key: torch.as_tensor(points[key][None], device=device)
        for key in ('world_xyz', 'rgb', 'uv', 'depth', 'valid_mask')}
    batch.update(features=torch.as_tensor(features[None], device=device), current=torch.as_tensor(current[None], device=device))
    if features.shape != (4096,) or not np.isfinite(features).all() or not np.isfinite(current).all():
        raise ValueError('Nonfinite or malformed model condition')
    return dict(batch=batch, rgb=rgb, observed=observed, current=current)


def generate_probe(model, inputs, output, synchronize=lambda: None):
    """Input dictionary has observations only. Does not accept labels or paths."""
    import torch
    if set(inputs) != set(IDS) or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Fixed twelve inputs and frozen evaluation model required')
    before = state_digest(model)
    normal, direct, geometric, pools, records = {}, {}, {}, {}, []
    try:
        with torch.inference_mode():
            for arm in ARMS:
                for identifier in IDS:
                    donor = identifier if arm != 'cyclic_direct_swap' else donor_id(identifier)
                    replacement = None if arm == 'normal' else direct[donor]
                    captured = []
                    append(output/'forward_ledger.jsonl', dict(event='started', arm=arm, id=identifier, candidate_states=4))
                    synchronize(); started = time.perf_counter()
                    with direct_override(model.head.feature_encoder, replacement, captured):
                        xyz, opened, geometry = model(**inputs[identifier]['batch'])
                    synchronize(); elapsed = time.perf_counter()-started
                    paths, events = xyz[0].cpu().numpy(), opened[0].cpu().numpy()
                    if not np.isfinite(paths).all() or not np.isfinite(events).all(): raise ValueError('Nonfinite probe output')
                    if arm == 'normal':
                        normal[identifier] = (paths.copy(), events.copy())
                        direct[identifier] = captured[0]
                        geometric[identifier] = {k:v.detach().clone() for k,v in geometry.items()}
                    else:
                        exact_geometry(geometric[identifier], geometry)
                        if arm == 'identity' and (not np.array_equal(paths, normal[identifier][0])
                                or not np.array_equal(events, normal[identifier][1])):
                            raise ValueError('Identity replacement did not recover exact original output')
                    fixed = clamp_endpoint(paths, normal[identifier][0])
                    pools[(arm, identifier)] = (fixed, events.copy())
                    file = output/(arm+'_'+identifier+'.npz')
                    with file.open('xb') as stream:
                        np.savez_compressed(stream, paths=fixed, unmodified_forward_paths=paths,
                            gripper_open=events, anchor_xyz=geometry['anchor_xyz'].cpu().numpy(),
                            current=inputs[identifier]['current'], direct_feature_used=(captured[0] if replacement is None else replacement).cpu().numpy())
                    record = dict(arm=arm,id=identifier,direct_feature_donor_id=donor,forward_seconds=elapsed,
                        candidates=4,prediction_sha256=digest(file),file=file.name,
                        geometry_anchor_current_unchanged=True,endpoint_fixed_to_normal=True,
                        displacement=displacement(normal[identifier][0], fixed), labels_opened=False)
                    records.append(record); append(output/'forward_ledger.jsonl', dict(event='completed',**record))
    finally:
        after = state_digest(model)
        write(output/'model_state_guard.json',dict(before=before,after=after,exact_unchanged=before==after))
        if before != after: raise ValueError('Frozen model state changed')
    write(output/'probe_generation_seal.json',dict(protocol=PROTOCOL,requests=36,complete_path_states=144,
        records=records, labels_opened=False, new_qwen_encodings=0, search_calls=0))
    return pools, records


def build_observed_graph(planner, row, value, fit, output):
    """Same original planner graph, component and attachments; no label parameter."""
    began = time.perf_counter()
    observed, current = value['observed'], value['current']
    rgb, xyz, valid = planner.prototype.observed_grid(value['rgb'], observed['depth'],
        observed['camera_intrinsics'], observed['camera_extrinsics'])
    endpoint, localization = planner.prototype.predict(rgb, xyz, valid, row['instruction'], fit['prototype'])
    result = dict(id=row['id'],localization=localization,search_calls=0,labels_opened=False)
    if endpoint is None:
        result.update(status='localization_unavailable',elapsed_seconds=time.perf_counter()-began)
        write(output/(row['id']+'_graph_seal.json'),result); return None,result
    try:
        free, lower, start_permission, target_permission, metadata = planner.v1.build_grid(rgb,xyz,valid,
            observed['depth'],observed['camera_intrinsics'],observed['camera_extrinsics'],current,endpoint,
            row['instruction'],fit['prototype'],localization,fit['workspace'])
    except ValueError as error:
        result.update(status='grid_rejected',error=str(error),elapsed_seconds=time.perf_counter()-began)
        write(output/(row['id']+'_graph_seal.json'),result); return None,result
    component = planner.v1.selected_target_component(rgb,valid,row['instruction'],fit['prototype'],localization)
    nonselected = xyz[valid & ~component]
    rays = dict(depth=observed['depth'],valid=valid,component=component,intrinsics=observed['camera_intrinsics'],
        camera_to_world=observed['camera_extrinsics'],current=current,endpoint=endpoint,
        target_radius=metadata['contact_allowances']['target_radius_m'])
    starts, start_audit = planner.attachments(current[:3],'start',free,lower,
        planner.CONFIG['current_tip_contact_radius_m'],nonselected,rays)
    goals, goal_audit = planner.attachments(endpoint,'goal',free,lower,rays['target_radius'],nonselected,rays)
    path = output/(row['id']+'_observed_graph.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream, free=free,lower=lower,start_permission=start_permission,
            target_permission=target_permission,predicted_goal=endpoint)
    result.update(status='constructed',grid=metadata,start_attachments=start_audit,goal_attachments=goal_audit,
        artifact_sha256=digest(path),elapsed_seconds=time.perf_counter()-began)
    write(output/(row['id']+'_graph_seal.json'),result)
    return dict(free=free,lower=lower,endpoint=endpoint,nonselected=nonselected,rays=rays,
        attachments_available=bool(starts and goals)),result


def reference_support(planner, path, graph):
    if graph is None: return dict(status='unknown_graph_unavailable',supported=None)
    grid_clear, checked_cells = planner.v1.path_observed_clear(path,graph['free'],graph['lower'])
    proxy = planner.visible_proxy(path,graph['nonselected'],graph['rays'])
    return dict(status='evaluated',original_grid_proxy_clear=bool(grid_clear),grid_cells_checked=checked_cells,
        original_visible_proxy=proxy,original_predicted_endpoint_attachments_available=graph['attachments_available'],
        reference_end_to_predicted_goal_m=float(np.linalg.norm(path[-1]-graph['endpoint'])),
        supported_by_both_existing_proxies=bool(grid_clear and proxy['passed']),
        limitation='Reference tested unchanged against predicted component; endpoint mismatch is not repaired. '
                   'Admitted finite references are positive witnesses only, rejection does not prove no other valid graph route.')


@contextmanager
def forbid_search(planner):
    saved=[(planner,'astar_virtual',planner.astar_virtual),(planner.v1,'astar',planner.v1.astar)]
    def forbidden(*unused,**unused_kw):
        raise RuntimeError('No search is authorized in this graph support audit')
    for module,name,unused in saved: setattr(module,name,forbidden)
    try: yield
    finally:
        for module,name,original in saved: setattr(module,name,original)


def run(project, policy_path, output, device):
    import torch
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_route_head import resample_event_segments
    from scripts.evaluate_observed_two_row_online import head_options
    from scripts import observation_spatial_penalty_astar as spatial
    from scripts.evaluate_observed_two_row import scene_metrics
    from scripts.collect_observed_two_row_pilot import crossing_signature
    project, policy_path, output = map(Path,(project,policy_path,output))
    output.mkdir(parents=True,exist_ok=False)
    began = time.perf_counter(); hashes = {}; policy = json.loads(policy_path.read_text())
    if (policy['protocol'],policy['ids'],policy['arms'],policy['max_forward_requests']) != (PROTOCOL,IDS,list(ARMS),36):
        raise ValueError('Fixed protocol/identity/budget changed')
    def pinned(relative,key):
        path=project/relative; actual=digest(path)
        if actual != policy[key]: raise ValueError('Pinned artifact changed: '+relative)
        hashes[str(path)]=actual; return path
    try:
        run_path=project/policy['training_run']; data=project/policy['data']
        config=json.loads(pinned(policy['training_run']+'/config.json','training_config_sha256').read_text())
        summary=json.loads(pinned(policy['training_run']+'/summary.json','training_summary_sha256').read_text())
        checkpoint=pinned(policy['training_run']+'/last.pt','checkpoint_sha256')
        if summary['last_checkpoint_sha256'] != digest(checkpoint) or summary['last_step'] != 12000:
            raise ValueError('Fixed completed last12000 required')
        manifest,gate=spatial.verify_export_metadata(data,dict(export_manifest_sha256=policy['export_manifest_sha256']))
        rows=selected_rows(data/'observations.jsonl')
        if any(set(r)!=INPUT_KEYS for r in rows): raise ValueError('Observation whitelist changed')
        expected_obs=str(data/'observations.jsonl')
        if config['observations'] != expected_obs or config['supervision'] != str(data/'supervision.jsonl'):
            raise ValueError('Training input table identity changed')
        cache_config_path=Path(config['cache_dir'])/'cache_config.json'
        if json.loads(cache_config_path.read_text()) != config['cache_config']:
            raise ValueError('Original frozen Qwen cache metadata changed')
        hashes[str(cache_config_path)]=digest(cache_config_path)
        old_source=project/'research_v2/releases'/policy['training_commit']
        source_receipts={}
        for name in MODEL_SOURCES:
            new,old=ROOT/name,old_source/name
            if new.read_bytes().replace(b'\r\n',b'\n') != old.read_bytes().replace(b'\r\n',b'\n'):
                raise ValueError('Trained model implementation changed: '+name)
            source_receipts[name]=dict(actual_sha256=digest(new),trained_actual_sha256=digest(old),
                canonical_lf_equal=True)
        planner=spatial.control.dependencies()
        fit_path=pinned(policy['planner_fit'],'planner_fit_sha256')
        fit=json.loads(fit_path.read_text())
        if spatial.fit_identity(fit['prototype'],fit['workspace']) != policy['planner_fit_identity']:
            raise ValueError('Original TRAIN-fitted planner parameters changed')
        torch.set_num_threads(1)
        if device=='cuda':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Only authorized GPU1')
            torch.cuda.set_per_process_memory_fraction(.35); torch.cuda.reset_peak_memory_stats()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['step'] != 12000 or saved['config'] != config: raise ValueError('Checkpoint config differs')
        model=ObservedGeometryRouteHead(**head_options(config)).to(device).eval()
        model.load_state_dict(saved['model'],strict=True)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        if set(policy['cache_sha256']) != set(IDS): raise ValueError('Exact twelve cache identities required')
        inputs={row['id']:load_inputs(row,manifest,config,hashes,device,policy['cache_sha256'][row['id']]) for row in rows}
        for parent in PARENTS:
            values=[inputs[parent+'_target%d'%t] for t in range(3)]
            if any(not np.array_equal(values[0]['rgb'],v['rgb']) or
                   not np.array_equal(values[0]['current'],v['current']) for v in values[1:]):
                raise ValueError('Donor changes image or current physical state')
        write(output/'input_source_seal.json',dict(protocol=PROTOCOL,inputs_sha256=hashes,model_sources=source_receipts,
            initial_mechanical_gate=gate,planner_fit_identity=policy['planner_fit_identity'],labels_opened=False,
            new_model_state_sha256=state_digest(model),policy_sha256=digest(policy_path),
            source_files_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ('routeset','scripts') for p in (ROOT/folder).rglob('*.py')},
            launcher_sha256=digest(ROOT/'scripts/launch_two_row_route_conditioning_v1.sh')))
        sync=torch.cuda.synchronize if device=='cuda' else lambda:None
        pools,records=generate_probe(model,inputs,output,sync)
        original_prediction=pinned(policy['saved_train_predictions'],'saved_train_predictions_sha256')
        normal_reproduction=compare_saved_normal(pools,original_prediction)
        write(output/'normal_reproduction.json',normal_reproduction)
        graph_data={}; graph_records={}
        with forbid_search(planner):
            for row in rows:
                graph_data[row['id']],graph_records[row['id']]=build_observed_graph(planner,row,inputs[row['id']],fit,output)
        write(output/'all_inputs_sealed.json',dict(forward_requests=36,graphs=12,search_calls=0,labels_opened=False,
            generation_seal_sha256=digest(output/'probe_generation_seal.json'),
            graph_seals={r['id']:digest(output/(r['id']+'_graph_seal.json')) for r in rows}))
        # No global dataset loader: only these twelve label payloads are decoded.
        labels=selected_rows(data/'supervision.jsonl'); evaluations=[]; supports=[]
        for row,label in zip(rows,labels):
            identifier=row['id']; value=inputs[identifier]
            with np.load(checked(label['verification_only'],manifest['source_files_sha256'],hashes),allow_pickle=False) as archive:
                boxes={key:archive[key].copy() for key in ('obstacle_centers','obstacle_halfsizes')}
            route_config=json.loads(checked(label['route_config'],manifest['source_files_sha256'],hashes).read_text())
            arm_rows={}
            for arm in ARMS:
                paths,events=pools[(arm,identifier)]
                _,candidate_rows=scene_metrics(paths,events,value['observed'],boxes,label['semantic_targets'],label['route_types'],route_config)
                arm_rows[arm]=candidate_rows
            for k in range(4):
                original,changed=arm_rows['normal'][k],arm_rows['cyclic_direct_swap'][k]
                evaluations.append(dict(id=identifier,candidate=k,arms={a:arm_rows[a][k] for a in ARMS},
                    clear_to_collision=bool(original['tip_segments_clear'] and not changed['tip_segments_clear']),
                    collision_to_clear=bool(not original['tip_segments_clear'] and changed['tip_segments_clear']),
                    not_a_task_accuracy_or_causal_generalization_result=True))
            refs=[]
            for i,name in enumerate(label['routes']):
                path=checked(name,manifest['source_files_sha256'],hashes)
                with np.load(path,allow_pickle=False) as archive:
                    raw=archive['gripper_pose'][:,:3].copy()
                    sampled,_=resample_event_segments(archive['gripper_pose'],archive['gripper_open'],24)
                refs.append(dict(reference_index=i,source=name,sha256=digest(path),known_type=label['route_types'][i],
                    raw_recomputed_type=crossing_signature(raw,route_config),
                    h24_recomputed_type=crossing_signature(sampled,route_config),
                    raw=reference_support(planner,raw,graph_data[identifier]),
                    h24=reference_support(planner,sampled,graph_data[identifier])))
            supports.append(dict(id=identifier,known_positive_count=len(refs),graph=graph_records[identifier],references=refs,
                incomplete_reference_set=True,unseen_paths_not_negative=True))
        for name,value in hashes.items():
            if digest(name)!=value: raise ValueError('Source mutated during audit: '+name)
        for record in records:
            if digest(output/record['file'])!=record['prediction_sha256']:
                raise ValueError('Sealed counterfactual pool changed during evaluation')
        report=dict(protocol=PROTOCOL,scope='Fixed TRAIN-only counterfactual diagnostic; not MAIN task quality or novelty',
            ids=IDS,records=records,collision_transitions=evaluations,reference_graph_support=supports,
            actual_forward_requests=36,actual_complete_path_states=144,new_qwen_encodings=0,search_calls=0,
            optimizer_updates=0,identity_output_exact=True,model_state_unchanged=True,ground_truth_model_inputs=False,
            references_opened_only_after_all_input_artifacts_sealed=True,
            actual_elapsed_seconds=time.perf_counter()-began,execution_device=device,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            gpu_hours_reserved=(time.perf_counter()-began)/3600 if device=='cuda' else 0,
            scope_limitations=['Direct-head sensitivity does not establish the cause of DEV generalization failure.',
                'An insensitive direct branch does not exclude the task-conditioned geometry/query branch.',
                'Graph proxy support is neither continuous graph-path existence nor complete valid-mode count.',
                'Planner uses preserved TRAIN32 prototype grounding, neural probe uses TRAIN64 Qwen head; these are separate diagnostics.'],
            source_hashes=hashes)
        report['normal_saved_pool_reproduction']=normal_reproduction
        write(output/'report.json',report)
        write(output/'status.json',dict(status='completed',exit_code=0,actual_forward_requests=36))
    except BaseException as error:
        ledger=[json.loads(line) for line in (output/'forward_ledger.jsonl').read_text().splitlines()] if (output/'forward_ledger.jsonl').exists() else []
        started_count=sum(r['event']=='started' for r in ledger)
        completed_count=sum(r['event']=='completed' for r in ledger)
        write(output/'status.json',dict(status='failed',exit_code=1,error=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-began,
            started_forward_requests=started_count,completed_forward_requests=completed_count,
            uncertain_inflight_forward_requests=started_count-completed_count,unattempted_requests=36-started_count,
            reserved_forward_requests=36,reserved_complete_path_states=144,
            failed_and_unattempted_requests_preserved=True,no_automatic_replay=True))
        raise
    finally:
        write(output/'artifact_index.json',{p.relative_to(output).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size)
            for p in output.rglob('*') if p.is_file() and p.name!='artifact_index.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','policy','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args(); run(args.project,args.policy,args.output,args.device)
