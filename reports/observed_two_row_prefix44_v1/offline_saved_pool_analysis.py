"""Local read-only aggregation of sealed prefix28/44 pools; no model or raw data IO."""
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
                path = ROOT / 'runs/synced_two_row_prefix44_v1' / target
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
        path = ROOT / 'runs/synced_two_row_prefix44_v1/online/best/requests' / row['prediction_file']
        assert digest(path) == row['prediction_sha256']
    summary = read(P / 'training/peak_seed0/summary.json')
    for stage in ('best', 'last'):
        remote = '/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_prefix44_v1/peak_seed0/' + stage + '.pt'
        assert by_remote[remote]['sha256'] == summary[stage + '_checkpoint_sha256']
    groups = {}
    for prefix, folder in [('prefix28', OLD), ('prefix44', P)]:
        for tag, sub in [('best_dev', 'dev_model'), ('last_dev', 'last_dev_model'), ('best_train', 'train')]:
            groups[prefix + '_' + tag] = read(folder / 'training/peak_seed0' / sub / 'per_scene.json')
    groups['prefix28_last_train'] = read(ROOT / 'reports/observed_two_row_last_train_v1/analysis/last_train/per_scene.json')
    groups['prefix44_last_train'] = read(P / 'last_train/analysis/last_train/per_scene.json')
    observations = rows(P / 'export_metadata/observations.jsonl')
    color = {r['id']: r['instruction'].split('touch the ', 1)[1].split(' sphere', 1)[0] for r in observations}
    train32 = [r for r in observations if r['split'] == 'TRAIN']
    train16 = [r for r in train32 if int(r['parent_id'].split('_')[-1]) < 283216]
    seen = {'prefix28': {color[r['id']] for r in train16}, 'prefix44': {color[r['id']] for r in train32}}
    result = dict(protocol='two_row_prefix44_local_saved_pool_analysis_v1',
        scope='Only synced saved pools/JSON and already-exported TRAIN/DEV instruction metadata; no raw geometries, model, GPU, simulator or new forward.',
        sync_sha256=digest(P / 'SYNC_SHA256_INDEX.json'),
        verification=dict(copied_files=len([r for r in sync['entries'] if r['copied']]),
            original_artifact_index_entries=len(verified), driver_artifact_entries=len(receipt['prediction_artifacts']),
            request_pool_seals=len(requests), checkpoints_server_hash_verified=2,
            checkpoints_copied=0),
        colors=dict(training_condition_counts={'prefix28': dict(sorted(Counter(color[r['id']] for r in train16).items())),
            'prefix44': dict(sorted(Counter(color[r['id']] for r in train32).items()))},
            seen={k: sorted(v) for k,v in seen.items()},
            dev_unseen={k: sorted({color[r['id']] for r in observations if r['split']=='DEV_MODEL'} - v) for k,v in seen.items()},
            definition='Exact language color name present among actual TRAIN instructions; no claim about pretrained Qwen exposure or open-vocabulary grounding.'),
        groups={}, target_groups={}, color_groups={}, paired_parents={}, fit_stages={}, jobs={}, timing={})
    for key, group in groups.items():
        result['groups'][key] = counts(group)
        result['target_groups'][key] = {str(t): counts([r for r in group if r['scene_id'].endswith('_target'+str(t))]) for t in range(3)}
        if key.endswith('_dev'):
            prefix = key.split('_')[0]
            result['color_groups'][key] = {status: counts([r for r in group if (color[r['scene_id']] in seen[prefix]) == (status=='seen')]) for status in ('seen', 'unseen')}
            # Keep a common partition under the ORIGINAL16 TRAIN color vocabulary.
            result['color_groups'][key]['fixed_original16_partition'] = {status: counts([r for r in group if (color[r['scene_id']] in seen['prefix28']) == (status=='seen')]) for status in ('seen','unseen')}
            result['color_groups'][key]['per_color'] = {c: counts([r for r in group if color[r['scene_id']]==c]) for c in sorted({color[r['scene_id']] for r in group})}
    for stage in ('best_dev','last_dev'):
        group_a, group_b = groups['prefix28_'+stage], groups['prefix44_'+stage]
        assert {r['scene_id'] for r in group_a} == {r['scene_id'] for r in group_b}
        parents = sorted({r['parent_id'] for r in group_a})
        pairs=[]
        for parent in parents:
            a,b=(counts([r for r in g if r['parent_id']==parent]) for g in (group_a,group_b))
            pairs.append(dict(parent_id=parent, prefix28=a, prefix44=b,
                valid_delta=b['valid']-a['valid'], any_delta=b['any_valid']-a['any_valid'], unique_delta=b['unique_type_sum']-a['unique_type_sum']))
        result['paired_parents'][stage] = pairs
    for prefix in ('prefix28','prefix44'):
        path = ROOT/'reports/observed_two_row_last_train_v1/analysis/report.json' if prefix=='prefix28' else P/'last_train/analysis/report.json'
        fit = read(path)
        result['fit_stages'][prefix]={}
        for key, stage in fit['stages'].items():
            values=np.array([r['per_vertex_distance_m'] for r in stage['per_candidate']])
            result['fit_stages'][prefix][key] = dict(saturation_MSE=stage['original_saturation_loss'],
                matched_vertex_ADE_m=float(values.mean()), vertex_p95_m=float(np.quantile(values,.95)), vertex_max_m=float(values.max()))
    # Original16 and newly added observed15 separate TRAIN cohorts, with no extra predictions.
    result['fixed_last_train_cohorts']={}
    for tag in ('best_train','last_train'):
        group=groups['prefix44_'+tag]
        result['fixed_last_train_cohorts'][tag]={cohort:counts([r for r in group if (int(r['parent_id'].split('_')[-1])<283216)==(cohort=='original16')]) for cohort in ('original16','added15_observed')}
    for path in sorted(P.rglob('*.status.json')):
        state=read(path)
        if 'start_utc' not in state:continue
        result['jobs'][path.relative_to(P).as_posix()] = dict(status=state['status'],exit_code=state.get('exit_code'),
            pid=state['pid'],child_pid=state.get('child_pid'), code_commit=state['code_commit'],
            elapsed_seconds=(datetime.fromisoformat(state['end_utc'])-datetime.fromisoformat(state['start_utc'])).total_seconds(),
            status_sha256=digest(path))
    for key in ('continuous_request_wall_ms','generation_ms','prediction_seal_ms','label_and_check_ms'):
        a=np.array([r[key] for r in requests]);result['timing'][key]=dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),first=float(a[0]))
    for key in requests[0]['stages']:
        if key.endswith('_ms'):
            a=np.array([r['stages'][key] for r in requests]);result['timing'][key]=dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),first=float(a[0]))
    result['history']=[dict(step=r['step'],loss=r['loss'],**{k:r['dev_model'][k] for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','semantic_goal_accuracy','selection_score')}) for r in read(P/'training/peak_seed0/history.json')]
    result['script_sha256']=digest(__file__)
    (P/'OFFLINE_ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('verification','colors','groups','fit_stages','jobs')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
