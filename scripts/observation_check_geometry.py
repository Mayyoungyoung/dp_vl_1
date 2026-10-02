"""CPU-only real-data geometry pipeline and interrupted-resume audit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from routeset.common import write_json
from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_route_head import load_observed_dataset
from routeset.train_v2 import positive_assignment_loss
from scripts.train_observed_geometry import load_geometry, batch_inputs, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = load_observed_dataset(args.data/'observations.jsonl', args.data/'supervision.jsonl', args.data/'qwen_cache')
    geometry = load_geometry(data, args.data/'observations.jsonl', args.data/'supervision.jsonl')
    model = ObservedGeometryRouteHead(data['features'].shape[1])
    before = {name:value.detach().clone() for name,value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ids = np.array([0, 1])
    started = time.perf_counter()
    xyz, opened, detail = model(**batch_inputs(data, geometry, ids, 'cpu'))
    target = torch.cat([torch.as_tensor(data['paths'][ids, :, 1:]),
                        torch.as_tensor(data['events'][ids, :, 1:, None])*.2], -1)
    prediction = torch.cat([xyz[:, :, 1:], opened[:, :, 1:, None]*.2], -1)
    loss = positive_assignment_loss(prediction, target, data['path_mask'][ids], 'saturation', np.random.default_rng(0))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    changed = {prefix: sum(not torch.equal(before[name], value) for name,value in model.named_parameters() if name.startswith(prefix))
               for prefix in ('geometry.point_encoder', 'geometry.task_query', 'head.output')}
    assert all(changed.values())
    fullcheck = dict(samples=2, point_count=geometry['metadata']['sampled_points'], parameters=model.active_parameter_count(),
                     geometry_parameters=model.geometry.active_parameter_count(), loss=float(loss.detach()),
                     forward_backward_update_cpu_s=time.perf_counter()-started, changed_parameter_tensors=changed)
    common = dict(observations=str(args.data/'observations.jsonl'), supervision=str(args.data/'supervision.jsonl'),
                  cache_dir=str(args.data/'qwen_cache'), steps=4, batch_size=2, candidates=4, horizon=24,
                  width=16, depth=1, pooling='both', geometry_pooling='spatial', point_width=8, pixel_stride=8,
                  endpoint_residual_bound=.05, event_scale=.2, lr=3e-4, seed=71, eval_every=2,
                  threads=1, device='cpu', resume=False, stop_after=None)
    for name, resume, stop in [('full', False, None), ('resumed', False, 2), ('resumed', True, None)]:
        config = dict(common, output=str(args.output/name), resume=resume, stop_after=stop)
        train(argparse.Namespace(**config))
    full = torch.load(args.output/'full'/'last.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(args.output/'resumed'/'last.pt', map_location='cpu', weights_only=False)
    bit_equal = all(torch.equal(value, resumed['model'][name]) for name,value in full['model'].items())
    assert bit_equal and full['step'] == resumed['step'] == 4 and full['trajectory_exposures'] == resumed['trajectory_exposures']
    result = dict(full_resolution_real_data=fullcheck, small_model_resume_bit_equal=bit_equal,
                  check_scope='pipeline and deterministic resumption only; no trained performance claim')
    write_json(args.output/'validation.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
