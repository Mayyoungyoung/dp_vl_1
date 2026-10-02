"""Versioned CPU reevaluation of existing frozen-cache observation checkpoints."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from routeset.common import sha256, write_json
from routeset.observed_route_head import ObservedRouteHead, load_observed_dataset
from scripts.train_observed_routes import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if (args.output/'metrics.json').exists():
        raise RuntimeError('do not overwrite prior evaluation evidence; choose a fresh output')
    config = json.loads((args.run/'config.json').read_text())
    data = load_observed_dataset(config['observations'], config['supervision'], config['cache_dir'], config['horizon'], config['pooling'])
    train = np.flatnonzero((data['splits'] == 'TRAIN') & data['path_mask'].any(1))
    dev = np.flatnonzero(data['splits'] == 'DEV_MODEL')
    model = ObservedRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'], config['depth'])
    checkpoint = torch.load(args.run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = evaluate(model, data, dev, 'cpu', args.output)
    train_metrics = evaluate(model, data, train, 'cpu', args.output/'train')
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), source_root/'scripts/train_observed_routes.py', source_root/'routeset/observed_route_head.py']
    source_commit = config.get('code_commit', 'unrecorded')
    # Existing run records identify the actual source commit; this path is the
    # project release location, separately recording whether it still exists.
    project_root = args.run.resolve().parents[2]
    original_source = project_root/'research_v2/releases'/source_commit
    provenance = dict(evaluation_protocol='observation_eval_v2', original_run=str(args.run.resolve()),
        original_config=str((args.run/'config.json').resolve()), original_config_sha256=sha256(args.run/'config.json'),
        original_checkpoint=str((args.run/'best.pt').resolve()), original_checkpoint_sha256=sha256(args.run/'best.pt'),
        original_checkpoint_step=checkpoint['step'], original_training_source_commit=source_commit,
        original_training_source_path=str(original_source), original_training_source_path_exists=original_source.exists(),
        evaluation_source_sha256={str(path):sha256(path) for path in source_paths}, dataset_fingerprint=data['fingerprint'],
        train_supervised_examples=len(train), dev_observations=len(dev),
        dev_with_reference=int(data['path_mask'][dev].any(1).sum()), unreferenced=data['unreferenced'],
        metric_change='no-reference observations retained for semantic evaluation; reference-only metrics null for them; 3cm and identity rules unchanged',
        checkpoint_selection='original best checkpoint preserved; reference ADE selection remains the same 23 DEV samples')
    write_json(args.output/'provenance.json', provenance)
    print(json.dumps(dict(metrics=metrics, train_metrics=train_metrics, provenance=provenance)), flush=True)


if __name__ == '__main__':
    main()
