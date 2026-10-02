"""Compare actual full-Qwen inference outputs against cached-feature evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def semantic(endpoints, label):
    distances = np.linalg.norm(np.asarray(endpoints)[:, None]-np.asarray(label['centers'])[None], axis=-1)
    return (distances.argmin(1) == label['target_index']) & (distances[:, label['target_index']] <= label['tolerance'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-run', type=Path, required=True)
    parser.add_argument('--head-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.head_run/'config.json').read_text())
    labels = {row['id']:row['semantic_targets'] for row in [json.loads(line) for line in Path(config['supervision']).read_text().splitlines() if line.strip()]}
    original_file = args.head_run/'dev_model/predictions.npz'
    full_file = args.full_run/'predictions.npz'
    with np.load(original_file, allow_pickle=False) as archive:
        original = {key:archive[key] for key in archive.files}
    with np.load(full_file, allow_pickle=False) as archive:
        actual = {key:archive[key] for key in archive.files}
    reference_indices = {str(identifier):index for index,identifier in enumerate(original['scene_ids'])}
    if set(reference_indices) != set(map(str, actual['scene_ids'])):
        raise ValueError('complete original and full-inference scene ID sets must match')
    order = [reference_indices[str(identifier)] for identifier in actual['scene_ids']]
    before, after = original['paths'][order], actual['paths']
    open_before, open_after = original['gripper_open'][order], actual['gripper_open']
    if before.shape != after.shape or open_before.shape != open_after.shape:
        raise ValueError('candidate/horizon/output shape mismatch')
    difference = np.abs(after-before)
    rows, original_semantics, actual_semantics = [], [], []
    for index, identifier in enumerate(actual['scene_ids']):
        first = semantic(before[index, :, -1], labels[str(identifier)])
        second = semantic(after[index, :, -1], labels[str(identifier)])
        original_semantics.extend(first.tolist()); actual_semantics.extend(second.tolist())
        rows.append(dict(id=str(identifier), max_xyz_abs_difference_m=float(difference[index].max()),
            candidate_original_goal_success=first.tolist(), candidate_full_inference_goal_success=second.tolist(),
            all_strict_semantic_decisions_equal=bool(np.array_equal(first,second))))
    original_metrics = json.loads((args.head_run/'dev_model/metrics.json').read_text())
    original_accuracy, actual_accuracy = float(np.mean(original_semantics)), float(np.mean(actual_semantics))
    if abs(original_accuracy-original_metrics['semantic_goal_accuracy']) > 1e-12:
        raise ValueError('saved original semantic metric disagrees with its actual predictions')
    report = dict(evaluation_protocol='observation_eval_v2', examples=len(rows), candidates=int(after.shape[1]),
        semantic_evaluation_examples=len(rows), reference_evaluation_examples=original_metrics['reference_evaluation_examples'],
        xyz_shape=list(after.shape), xyz_max_abs_difference_m=float(difference.max()),
        xyz_mean_abs_difference_m=float(difference.mean()), xyz_rms_difference_m=float(np.sqrt(np.square(difference).mean())),
        endpoint_max_distance_between_predictions_m=float(np.linalg.norm(after[:, :, -1]-before[:, :, -1], axis=-1).max()),
        gripper_open_max_abs_difference=float(np.abs(open_after-open_before).max()),
        xyz_allclose_atol_1e_minus6_rtol_1e_minus5=bool(np.allclose(after,before,atol=1e-6,rtol=1e-5)),
        numerical_comparison_tolerance=dict(atol=1e-6,rtol=1e-5),
        semantic_decisions_identical=original_semantics == actual_semantics,
        original_strict_semantic_goal_accuracy=original_accuracy, full_inference_strict_semantic_goal_accuracy=actual_accuracy,
        semantic_rule='unchanged nearest target identity AND original 0.03m tolerance',
        source_files_sha256={str(path.resolve()):digest(path) for path in (original_file,full_file,args.head_run/'best.pt',args.full_run/'summary.json')},
        script_sha256=digest(__file__), per_scene=rows,
        scope='checks inference-path equivalence on actual predictions; no new task generalization or robot execution claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key != 'per_scene'}),flush=True)
    if not report['semantic_decisions_identical']:
        raise RuntimeError('strict semantic outcomes changed; preserve report and investigate')


if __name__ == '__main__':
    main()
