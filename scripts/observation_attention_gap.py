"""Read-only diagnosis of a frozen-Qwen RGB-D checkpoint on every DEV input.

Target centers enter diagnostics only, after normal forward. Peak-attention
locations are explanations, not a substituted predictor or revised benchmark.
The 4cm attention neighborhood is distinct from unchanged 3cm semantic success.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from routeset.common import sha256, write_json
from routeset.observed_route_head import load_observed_dataset, semantic_endpoint_accuracy
from routeset.observed_geometry import ObservedGeometryRouteHead
from scripts.train_observed_geometry import load_geometry, batch_inputs


@torch.no_grad()
def analyze(args):
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError('preserve prior diagnostic: '+str(args.output))
    checkpoint = torch.load(args.run/'best.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    data = load_observed_dataset(config['observations'], config['supervision'], config['cache_dir'], config['horizon'], config['pooling'])
    geometry = load_geometry(data, config['observations'], config['supervision'], config['pixel_stride'])
    if geometry['fingerprint'] != config['dataset_fingerprint']:
        raise ValueError('current dataset differs from checkpoint')
    model = ObservedGeometryRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'],
                                     config['depth'], config['point_width'], config['endpoint_residual_bound']).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    prototype = json.loads(args.prototype.read_text())
    prototype_rows = {row['id']:row for row in prototype['per_scene']}
    dev_ids = np.flatnonzero(data['splits'] == 'DEV_MODEL')
    fixed_train_parents = sorted(set(data['parent_ids'][data['splits']=='TRAIN']))[:8]
    train_ids = np.flatnonzero((data['splits']=='TRAIN') & np.isin(data['parent_ids'],fixed_train_parents))
    output = []
    for idx in np.r_[dev_ids,train_ids]:
        inputs = batch_inputs(data, geometry, np.asarray([idx]), 'cpu')
        paths, _, details = model(**inputs)
        weights = details['attention'][0].numpy()
        xyz = inputs['world_xyz'][0].numpy()
        valid = inputs['valid_mask'][0].numpy()
        anchor, endpoint = details['anchor_xyz'][0].numpy(), paths[0, :, -1].numpy()
        peak = xyz[weights.argmax()]
        # Do not access semantic labels until after the complete normal forward.
        spec = data['semantic_targets'][idx]
        centers, target = np.asarray(spec['centers']), spec['target_index']
        distances = np.linalg.norm(xyz[:, None]-centers[None], axis=-1)
        neighborhoods = valid[:, None] & (distances <= .04)
        masses = (weights[:, None]*neighborhoods).sum(0)
        own = prototype_rows.get(str(data['scene_ids'][idx]))
        proto_endpoint = None if own is None else own['prediction_endpoint']
        proto_error = None if proto_endpoint is None else float(np.linalg.norm(np.asarray(proto_endpoint)-centers[target]))
        if proto_error is not None and abs(proto_error-own['goal_error_m']) > 1e-9:
            raise ValueError('prototype and checkpoint do not share evaluation goal labels')
        output.append(dict(id=str(data['scene_ids'][idx]),parent_id=str(data['parent_ids'][idx]),
                           split=str(data['splits'][idx]),
                           reference_count=int(data['path_mask'][idx].sum()),
                           intended_target_mass_4cm=float(masses[target]),
                           other_target_mass_4cm=float(np.delete(masses,target).sum()),
                           outside_all_target_neighborhoods_mass=float(weights[~neighborhoods.any(1)].sum()),
                           intended_has_greatest_target_mass=bool(masses.argmax()==target),
                           per_target_attention_mass=masses.tolist(),
                           effective_point_count=float(1./np.sum(weights**2)),
                           valid_points=int(valid.sum()),
                           attention_peak_point=peak.tolist(),learned_anchor=anchor.tolist(),
                           peak_goal_error_m=float(np.linalg.norm(peak-centers[target])),
                           peak_semantic_accuracy=float(semantic_endpoint_accuracy(peak[None],spec)[0]),
                           anchor_goal_error_m=float(np.linalg.norm(anchor-centers[target])),
                           anchor_semantic_accuracy=float(semantic_endpoint_accuracy(anchor[None],spec)[0]),
                           candidate_goal_error_m=float(np.linalg.norm(endpoint-centers[target],axis=-1).mean()),
                           candidate_semantic_accuracy=float(semantic_endpoint_accuracy(endpoint,spec).mean()),
                           endpoint_from_anchor_m=float(np.linalg.norm(endpoint-anchor,axis=-1).mean()),
                           prototype_goal_error_m=proto_error,prototype_semantic_accuracy=None if own is None else own['semantic_goal_accuracy']))
    dev_rows = [row for row in output if row['split']=='DEV_MODEL']
    train_rows = [row for row in output if row['split']=='TRAIN']
    if {row['id'] for row in dev_rows} != set(prototype_rows):
        raise ValueError('all prototype and model DEV IDs must match exactly')
    fields = ('intended_target_mass_4cm','other_target_mass_4cm','outside_all_target_neighborhoods_mass',
              'intended_has_greatest_target_mass','effective_point_count','peak_goal_error_m','peak_semantic_accuracy',
              'anchor_goal_error_m','anchor_semantic_accuracy','candidate_goal_error_m','candidate_semantic_accuracy',
              'endpoint_from_anchor_m','prototype_goal_error_m','prototype_semantic_accuracy')
    def summarize(rows):
        return {key:float(np.mean([row[key] for row in rows if row[key] is not None]))
                if any(row[key] is not None for row in rows) else None for key in fields}
    means = summarize(dev_rows)
    result=dict(scope='post-hoc grounding diagnosis, no peak-output replacement or revised selection',
                split='DEV_MODEL',examples=len(dev_rows),semantic_denominator=len(dev_rows),
                reference_examples=sum(row['reference_count']>0 for row in dev_rows),
                fixed_training_diagnostic=dict(selection='lexicographically first eight TRAIN parents, all instructions, never selected by performance',
                    parents=list(fixed_train_parents),examples=len(train_rows),means=summarize(train_rows)),
                semantic_rule='unchanged target identity and <=0.03m; all rows retained',
                diagnostic_neighborhood_m=.04,device='cpu',threads=1,
                frozen_Qwen_cache_used=True,cache_usage='only a genuinely frozen-Qwen checkpoint is inspected; this is not online LoRA or end-to-end latency',
                prototype_output_scope='one surface endpoint',model_output_scope='four complete paths; centers compared only for localization diagnosis',
                checkpoint=str(args.run/'best.pt'),checkpoint_sha256=sha256(args.run/'best.pt'),
                source_training_config=config,prototype_report_sha256=sha256(args.prototype),
                script_sha256=sha256(__file__),means=means,per_scene=output)
    write_json(args.output,result)
    print(json.dumps(dict(examples=len(dev_rows),means=means,fixed_training_means=summarize(train_rows))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prototype',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args)


if __name__=='__main__':
    main()
