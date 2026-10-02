"""Paired development analysis of conventional observed-point grounding."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--output-name', default='observation_geometry_three_seed')
    parser.add_argument('--report-name', default='OBSERVATION_GROUNDING_THREE_SEED')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = {
        0: {'plain': root/'runs/observed_geometry_v1/seed0',
            'aux': root/'runs/observed_geometry_grounding_v2/seed0'},
        **{seed: {method: root/('runs/observed_geometry_seeds_v2/%s_seed%d' % (method, seed))
                   for method in ['plain', 'aux']} for seed in [1, 2]},
    }
    if args.runs:
        args.runs = args.runs.resolve()
        paths = {seed: {method: args.runs/('%s_seed%d' % (method, seed)) for method in ['plain', 'aux']}
                 for seed in [0, 1, 2]}
    metrics = ['semantic_goal_accuracy', 'AnySemanticGoalAtK', 'candidate_matched_ADE_m',
               'candidate_endpoint_error_m', 'learned_surface_anchor_reference_endpoint_error_m']
    runs, ids, parents = {}, None, None
    for seed, methods in paths.items():
        runs[str(seed)] = {}
        for method, path in methods.items():
            summary = json.loads((path/'summary.json').read_text())
            config = json.loads((path/'config.json').read_text())
            rows = json.loads((path/'dev_model/per_scene.json').read_text())
            assert config['seed'] == seed and config['steps'] == 1000
            assert config.get('grounding_weight', 0.) == (0. if method == 'plain' else .02)
            assert summary['trajectory_exposures'] == 128000
            assert summary['metrics']['evaluation_protocol'] == 'observation_eval_v2'
            assert summary['metrics']['semantic_evaluation_examples'] == 24
            assert summary['metrics']['reference_evaluation_examples'] == 23
            current_ids = [row['scene_id'] for row in rows]
            if ids is None:
                ids = current_ids
                parents = [row['parent_id'] for row in rows]
            assert ids == current_ids
            for key, file in [('best_checkpoint_sha256', 'best.pt'), ('prediction_sha256', 'dev_model/predictions.npz')]:
                assert digest(path/file) == summary[key], (path, key)
            for key in metrics:
                measured = [row[key] for row in rows if row[key] is not None]
                assert abs(np.mean(measured) - summary['metrics'][key]) < 1e-7
            runs[str(seed)][method] = dict(run=str(path.relative_to(root)).replace('\\', '/'),
                code_commit=config['code_commit'], dataset_fingerprint=config['dataset_fingerprint'],
                train_parents=config['train_parents'], train_examples=config['train_examples'],
                selected_step=summary['best_step'], elapsed_s=summary['elapsed_s'],
                gpu_hours=summary['gpu_hours_reserved'], training_exposures=summary['trajectory_exposures'],
                checkpoint_sha256=summary['best_checkpoint_sha256'], prediction_sha256=summary['prediction_sha256'],
                metrics={key: summary['metrics'][key] for key in metrics}, per_scene=rows,
                language_control=summary['metrics']['paired_language_control'])
        assert runs[str(seed)]['plain']['dataset_fingerprint'] == runs[str(seed)]['aux']['dataset_fingerprint']
    parents = np.asarray(parents)
    parent_names = sorted(set(parents))
    results = {}
    rng = np.random.default_rng(20261002)
    draws = rng.integers(len(parent_names), size=(10000, len(parent_names)))
    for key in metrics:
        values = {method: np.asarray([runs[str(seed)][method]['metrics'][key] for seed in [0, 1, 2]])
                  for method in ['plain', 'aux']}
        delta = values['aux'] - values['plain']
        per_parent = []
        for parent in parent_names:
            per_seed = []
            for seed in [0, 1, 2]:
                paired = []
                for idx in np.flatnonzero(parents == parent):
                    left, right = [runs[str(seed)][method]['per_scene'][idx][key] for method in ['plain', 'aux']]
                    if left is not None and right is not None:
                        paired.append(right-left)
                per_seed.append(np.mean(paired))
            per_parent.append(np.mean(per_seed))
        # Equal parent weights for the parent bootstrap. Reference-metric sample means
        # can differ slightly because one parent's instruction lacks a reference.
        ci = np.quantile(np.asarray(per_parent)[draws].mean(1), [.025, .975])
        results[key] = dict(**{method: dict(mean=float(v.mean()), training_seed_sample_sd=float(v.std(ddof=1)),
                                           seed_values=v.tolist()) for method, v in values.items()},
                           paired_seed_differences=delta.tolist(), delta_mean=float(delta.mean()),
                           delta_training_seed_sample_sd=float(delta.std(ddof=1)),
                           fixed_three_models_equal_parent_bootstrap95=ci.tolist())
    target_results = {}
    for suffix in ['target0', 'target1', 'target2']:
        target_results[suffix] = {}
        for method in ['plain', 'aux']:
            target_results[suffix][method] = {
                key: [float(np.mean([r[key] for r in runs[str(seed)][method]['per_scene']
                                     if r['scene_id'].endswith(suffix) and r[key] is not None]))
                      for seed in [0, 1, 2]] for key in metrics}
    result = dict(scope='DEV_MODEL, 8 parents/24 language instructions; selected checkpoints, not locked confirmation',
                  bootstrap='10000 draws of 8 independent parents after averaging three trained models; not training-seed uncertainty',
                  metrics=results, target_breakdown=target_results, runs=runs)
    output = root/'reports'/(args.output_name+'.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    train_parents = runs['0']['plain']['train_parents']
    reproduction = 'python scripts/analyze_geometry_seeds.py'
    if args.runs:
        reproduction += ' --runs '+str(args.runs.relative_to(root)).replace('\\','/')+' --output-name '+args.output_name+' --report-name '+args.report_name
    lines = ['# 观测几何定位的三种子配对实验', '',
        '训练父场景数：%d。%s' % (train_parents,
            '新采集训练布局；验证仍复用原8父DEV，不是独立测试设置。' if args.runs else '原24TRAIN/8DEV开发批。'), '',
        '同一真实冻结Qwen、RGB-D、当前状态与普通集合回归头；仅增加训练时的正参考终点附近像素注意力监督（权重0.02）。'
        '目标坐标和终点标签均不进入模型前向。每次训练1000步、batch32、K4、128000反传目标槽、1231965参数。'
        '同一8父/24指令DEV_MODEL选优；23条有参考路径，全部24条均计语义评价。', '',
        '| 指标 | 普通RGB-D均值±种子SD | 加辅助监督均值±种子SD | 配对差seed0/1/2 |',
        '|---|---:|---:|---|']
    for key, value in results.items():
        a, b = value['plain'], value['aux']
        lines.append('| %s | %.6f ± %.6f | %.6f ± %.6f | %s |' %
            (key, a['mean'], a['training_seed_sample_sd'], b['mean'], b['training_seed_sample_sd'],
             ' / '.join('%+.6f' % x for x in value['paired_seed_differences'])))
    semantic_improved = sum(x > 0 for x in results['semantic_goal_accuracy']['paired_seed_differences'])
    endpoint_improved = sum(x < 0 for x in results['candidate_endpoint_error_m']['paired_seed_differences'])
    ade_improved = sum(x < 0 for x in results['candidate_matched_ADE_m']['paired_seed_differences'])
    lines += ['', ('严格语义正确率改善%d/3种子，终点误差改善%d/3种子，参考ADE改善%d/3种子。' %
                  (semantic_improved, endpoint_improved, ade_improved)) +
        '这项常规监督是增强基线的必要修正，不是路线集合机制的新颖性或机器人执行证据。'
        '每个模型都已在这个很小的开发集上选优；没有把8个父场景扩算成24个独立样本，'
        '也不把固定三个模型的父场景bootstrap区间当作训练种子稳定性。', '',
        '下一步实际采集新的独立父布局，并在新增数据上用同一评价规则重复普通/辅助基线。'
        '接下来必须检验完整路径的碰撞和类型覆盖，不能将终点正确率称作Valid@K。'
        '完整逐场景、语言目标分解、哈希、代码版本和未成功场景均在相邻JSON。', '',
        '复现：`'+reproduction+'`。输入位于本地runs索引对应的服务器运行目录；'
        '脚本验证checkpoint/预测文件SHA、训练预算和逐行聚合。']
    (root/'reports'/(args.report_name+'.md')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({key: {k:v for k,v in value.items() if k not in ['plain','aux']} for key,value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
