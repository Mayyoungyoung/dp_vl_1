"""Local read-only aggregation of sealed prefix28/44/76 pools; no model or raw data IO."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
OLD = ROOT / 'reports/observed_two_row_prefix28_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def counts(group):
    candidates = [c for row in group for c in row['tip_candidates']]
    grid = Counter((bool(c['semantic_goal_correct']), bool(c['tip_segments_clear'])) for c in candidates)
    finite = sum(c['finite_xyz'] and c['finite_event_values'] for c in candidates)
    assert sum(grid.values()) == len(candidates)
    return dict(conditions=len(group), parents=len({r['parent_id'] for r in group}), slots=len(candidates),
        finite=finite, valid=sum(c['TipValid'] for c in candidates),
        any_valid=sum(any(c['TipValid'] for c in r['tip_candidates']) for r in group),
        semantic=sum(c['semantic_goal_correct'] for c in candidates),
        clear=sum(c['tip_segments_clear'] for c in candidates),
        correct_goal_clear=grid[True, True], correct_goal_collision=grid[True, False],
        wrong_goal_clear=grid[False, True], wrong_goal_collision=grid[False, False],
        classified_valid=sum(c['classified_tip_valid'] for c in candidates),
        unique_type_sum=sum(r['tip_evaluation']['UniqueClassifiedTipValidAtK'] for r in group),
        unknown_valid=sum(r['tip_evaluation']['UnknownTypeTipValidCount'] for r in group),
        duplicate_valid=sum(r['tip_evaluation']['DuplicateClassifiedTipValidCount'] for r in group),
        reference_count=sum(r['reference_count'] for r in group),
        candidate_nearest_reference_ADE_m=float(np.mean([r['candidate_matched_ADE_m'] for r in group])) if group else None,
        reference_nearest_candidate_ADE_m=float(np.mean([r['reference_matched_ADE_m'] for r in group])) if group else None,
        candidate_nearest_reference_endpoint_m=float(np.mean([r['candidate_endpoint_error_m'] for r in group])) if group else None,
        valid_count_histogram=dict(sorted(Counter(sum(c['TipValid'] for c in r['tip_candidates']) for r in group).items())),
        valid_type_histogram=dict(sorted(Counter('|'.join(c['declared_passage_type']) for c in candidates if c['classified_tip_valid']).items())))


def main():
    sync = read(P / 'SYNC_SHA256_INDEX.json')
    by_remote = {row['server_path']: row for row in sync['entries']}
    for row in sync['entries']:
        if row['copied']:
            path = ROOT / row['local_path']
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    verified = []
    for relative in ('diagnosis/analysis/artifact_index.json', 'last_train/analysis/artifact_index.json'):
        folder = Path(relative).parent
        for name, value in read(P / relative).items():
            target = folder / name
            path = P / target
            if not path.exists():
                path = ROOT / 'runs/synced_two_row_prefix76_v1' / target
            sha = value if isinstance(value, str) else value['sha256']
            assert digest(path) == sha
            if isinstance(value, dict): assert path.stat().st_size == value['bytes']
            verified.append(str(target))
    receipt = read(P / 'training/peak_seed0/two_row_driver_receipt.json')
    for row in receipt['prediction_artifacts'].values():
        indexed = by_remote[row['path']]
        assert row['sha256'] == indexed['sha256'] and row['bytes'] == indexed['bytes']
    requests = rows(P / 'online/best/requests.jsonl')
    for row in requests:
        path = ROOT / 'runs/synced_two_row_prefix76_v1/online/best/requests' / row['prediction_file']
        assert digest(path) == row['prediction_sha256']
    summary = read(P / 'training/peak_seed0/summary.json')
    for stage in ('best', 'last'):
        remote = '/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_prefix76_v1/peak_seed0/' + stage + '.pt'
        assert by_remote[remote]['sha256'] == summary[stage + '_checkpoint_sha256']
    folders = {tag: ROOT / ('reports/observed_two_row_' + tag + '_v1') for tag in ('prefix28', 'prefix44', 'prefix76')}
    groups, configs, fits, colors, train_ids = {}, {}, {}, {}, {}
    for prefix, folder in folders.items():
        configs[prefix] = read(folder / 'training/peak_seed0/config.json')
        obs_folder = folders['prefix44'] if prefix == 'prefix28' else folder
        obs = rows(obs_folder / 'export_metadata/observations.jsonl')
        if prefix == 'prefix28':
            obs = [r for r in obs if r['split'] != 'TRAIN' or int(r['parent_id'].split('_')[-1]) < 283216]
        colors[prefix] = {r['id']: r['instruction'].split('touch the ', 1)[1].split(' sphere', 1)[0] for r in obs}
        train_ids[prefix] = {r['id'] for r in obs if r['split'] == 'TRAIN'}
        for tag, sub in [('best_dev', 'dev_model'), ('last_dev', 'last_dev_model'), ('best_train', 'train')]:
            groups[prefix + '_' + tag] = read(folder / 'training/peak_seed0' / sub / 'per_scene.json')
        last_folder = ROOT / 'reports/observed_two_row_last_train_v1/analysis' if prefix == 'prefix28' else folder / 'last_train/analysis'
        groups[prefix + '_last_train'] = read(last_folder / 'last_train/per_scene.json')
        fits[prefix] = read(last_folder / 'report.json')
    keys = ('steps','batch_size','candidates','horizon','width','depth','point_width','lr','seed','eval_every','pooling','pixel_stride','anchor_mode','endpoint_mode','endpoint_residual_bound','grounding_target','grounding_weight','grounding_sigma','event_scale','refinement_mode','sampling_mode','objective','feature_dim','selection_metric')
    differences = {key: {p:c[key] for p,c in configs.items()} for key in keys if len({str(c[key]) for c in configs.values()}) != 1}
    assert not differences, differences
    inits = {p: read(folder / 'training/peak_seed0/summary.json')['sample_stream_audit']['initial_model_sha256'] for p,folder in folders.items()}
    assert len(set(inits.values())) == 1
    dev = {p:{row['id']:row for row in rows((folders['prefix44'] if p=='prefix28' else folder) / 'export_metadata/observations.jsonl') if row['split']=='DEV_MODEL'} for p,folder in folders.items()}
    assert dev['prefix28'] == dev['prefix44'] == dev['prefix76'] and len(dev['prefix76']) == 36
    devrefs = {p:{row['id']:row for row in rows((folders['prefix44'] if p=='prefix28' else folder) / 'export_metadata/supervision.jsonl') if row['split']=='DEV_MODEL'} for p,folder in folders.items()}
    assert devrefs['prefix28'] == devrefs['prefix44'] == devrefs['prefix76']
    for prefix in ('prefix44','prefix76'):
        identity = read(folders[prefix] / 'preparation/fixed_dev_identity.json')
        assert identity['observation_and_supervision_rows_equal'] and identity['raw_source_hashes_equal']
        assert identity['old_export_manifest_sha256'] == '5c14a6494d6f3ba887347f64767c72e8f1aa7c280cd4d43933063f42a53da484'
    current_seen = {p:{colors[p][i] for i in train_ids[p]} for p in folders}
    result = dict(protocol='two_row_prefix76_local_saved_pool_analysis_v1',
        scope='Only synced sealed pools/JSON and TRAIN/DEV instruction metadata. No raw geometry, model, simulator, optimizer, planner or forward call.',
        sync_sha256=digest(P / 'SYNC_SHA256_INDEX.json'),
        verification=dict(copied_files=sum(r['copied'] for r in sync['entries']), original_artifact_index_entries=len(verified),
            driver_artifact_entries=len(receipt['prediction_artifacts']), request_pool_seals=len(requests), checkpoint_server_hashes_verified=2,
            checkpoints_copied=0, hyperparameter_keys_identical=list(keys), initial_model_sha256=inits, fixed36_dev_inputs_and_reference_rows_equal=True, original28_rows_via_bound_server_identity_receipts=True),
        groups={}, target_groups={}, color_groups={}, paired_parents={}, fit_stages={}, jobs={}, timing={},
        colors=dict(training_condition_counts={p:dict(sorted(Counter(colors[p][i] for i in train_ids[p]).items())) for p in folders},
            seen={p:sorted(x) for p,x in current_seen.items()}, definition='Exact instruction color present in TRAIN, not Qwen pretraining or open-vocabulary ability.'))
    for key, group in groups.items():
        result['groups'][key] = counts(group)
        result['target_groups'][key] = {str(t):counts([r for r in group if r['scene_id'].endswith('_target'+str(t))]) for t in range(3)}
        if key.endswith('_dev'):
            prefix = key.split('_')[0]
            result['color_groups'][key] = {status:counts([r for r in group if (colors[prefix][r['scene_id']] in current_seen[prefix]) == (status=='seen')]) for status in ('seen','unseen')}
            result['color_groups'][key]['fixed_original16_partition'] = {status:counts([r for r in group if (colors[prefix][r['scene_id']] in current_seen['prefix28']) == (status=='seen')]) for status in ('seen','unseen')}
            result['color_groups'][key]['per_color'] = {c:counts([r for r in group if colors[prefix][r['scene_id']]==c]) for c in sorted({colors[prefix][r['scene_id']] for r in group})}
    for stage in ('best_dev','last_dev'):
        result['paired_parents'][stage] = []
        for parent in sorted({r['parent_id'] for r in groups['prefix76_'+stage]}):
            record = dict(parent_id=parent)
            for p in folders: record[p] = counts([r for r in groups[p+'_'+stage] if r['parent_id']==parent])
            record['delta_64_minus_16'] = {k:record['prefix76'][k]-record['prefix28'][k] for k in ('valid','any_valid','unique_type_sum','semantic','clear')}
            record['delta_64_minus_32'] = {k:record['prefix76'][k]-record['prefix44'][k] for k in ('valid','any_valid','unique_type_sum','semantic','clear')}
            result['paired_parents'][stage].append(record)
    for prefix,fit in fits.items():
        result['fit_stages'][prefix] = {}
        for key,stage in fit['stages'].items():
            a = np.array([r['per_vertex_distance_m'] for r in stage['per_candidate']])
            result['fit_stages'][prefix][key] = dict(saturation_MSE=stage['original_saturation_loss'], matched_vertex_ADE_m=float(a.mean()), vertex_p95_m=float(np.quantile(a,.95)), vertex_max_m=float(a.max()))
    result['prefix76_train_cohorts'] = {}
    for tag in ('best_train','last_train'):
        group=groups['prefix76_'+tag]
        result['prefix76_train_cohorts'][tag] = {name:counts([r for r in group if lo<=int(r['parent_id'].split('_')[-1])<hi]) for name,lo,hi in [('original16',283200,283216),('next16_registered_15_observed',283216,283232),('newest32',283232,283264)]}
    for path in sorted(P.rglob('*.status.json')):
        state=read(path)
        if 'start_utc' not in state:continue
        assert state['exit_code']==0
        result['jobs'][path.relative_to(P).as_posix()] = dict(status=state['status'],exit_code=state['exit_code'],pid=state['pid'],child_pid=state.get('child_pid'),code_commit=state['code_commit'],elapsed_seconds=(datetime.fromisoformat(state['end_utc'])-datetime.fromisoformat(state['start_utc'])).total_seconds(),status_sha256=digest(path))
    for key in ('continuous_request_wall_ms','generation_ms','prediction_seal_ms','label_and_check_ms'):
        a=np.array([r[key] for r in requests]);result['timing'][key]=dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),first=float(a[0]))
    for key in requests[0]['stages']:
        if key.endswith('_ms'):
            a=np.array([r['stages'][key] for r in requests]);result['timing'][key]=dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),first=float(a[0]))
    comparisons = read(P/'online/best/cache_comparison.json')['per_scene']
    assert len(comparisons)==36 and all(c['candidate_decisions_identical'] for c in comparisons)
    result['online_consistency'] = dict(examples=36, all_decisions_equal=True, max_xyz_abs_diff_m=max(c['xyz_max_abs_difference_m'] for c in comparisons), max_event_abs_diff=max(c['event_max_abs_difference'] for c in comparisons), max_feature_abs_diff=max(max(c['feature_max_abs_difference'].values()) for c in comparisons), new_consistency_forwards=0)
    result['history'] = [dict(step=r['step'],loss=r['loss'],**{k:r['dev_model'][k] for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','semantic_goal_accuracy','selection_score')}) for r in read(P/'training/peak_seed0/history.json')]
    result['script_sha256'] = digest(__file__)
    (P/'OFFLINE_ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verification=result['verification'], groups=result['groups'], fit=result['fit_stages'], jobs=result['jobs']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
