"""Fixed-TRAIN-batch decomposition of shared geometry parameter gradients.

No optimizer update or checkpoint reselection is performed. Full saturation
assignment is computed once per checkpoint and reused for every loss component.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from routeset.common import sha256
from routeset.observed_geometry import ObservedGeometryRouteHead, positive_endpoint_attention_loss
from routeset.observed_route_head import load_observed_dataset
from routeset.train_v2 import positive_assignment_loss
from scripts.train_observed_geometry import batch_inputs, load_geometry


def assigned_targets(prediction, target, mask):
    costs = (prediction[:, :, None] - target[:, None]).square().mean((-1, -2)).detach().cpu().numpy()
    choices = []
    k = prediction.shape[1]
    for row in range(len(prediction)):
        ids = np.flatnonzero(mask[row])
        if not len(ids):
            raise ValueError("fixed batch must use original positive-reference TRAIN eligibility")
        small = costs[row][:, ids]
        if len(ids) < k:
            nearest = small.argmin(1)
            target_rows, candidate_cols = linear_sum_assignment((small - small.min(1)[:, None]).T)
            nearest[candidate_cols] = target_rows
            choices.append(ids[nearest])
        else:
            candidate_rows, target_cols = linear_sum_assignment(small)
            matched = np.empty(k, dtype=np.int64)
            matched[candidate_rows] = ids[target_cols]
            choices.append(matched)
    indices = torch.as_tensor(np.stack(choices), dtype=torch.long, device=prediction.device)
    matched = target[torch.arange(len(target), device=target.device)[:, None], indices]
    return matched, indices.cpu().tolist()


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1).detach()
                      for parameter, gradient in zip(parameters, gradients)])


def compare_vectors(first, second):
    a, b = float(first.norm()), float(second.norm())
    return dict(first_norm=a, second_norm=b, first_over_second_norm=a / b if b else None,
                cosine=float(torch.dot(first, second) / (a * b)) if a and b else None,
                dot_product=float(torch.dot(first, second)))


def diagnose(run, output, batch_size=6):
    torch.set_num_threads(1)
    run, output = Path(run), Path(output)
    config = json.loads((run / "config.json").read_text())
    if not config.get("grounding_weight", 0.):
        raise ValueError("this diagnostic requires the actual positive-grounding auxiliary run")
    data = load_observed_dataset(config["observations"], config["supervision"], config["cache_dir"], config["horizon"], config["pooling"])
    geometry = load_geometry(data, config["observations"], config["supervision"], config["pixel_stride"])
    if geometry["fingerprint"] != config["dataset_fingerprint"]:
        raise ValueError("checkpoint dataset fingerprint mismatch")
    eligible = np.flatnonzero((data["splits"] == "TRAIN") & data["path_mask"].any(1))
    ids = np.array(sorted(eligible, key=lambda idx: str(data["scene_ids"][idx]))[:batch_size])
    if len(ids) != batch_size:
        raise ValueError("not enough eligible TRAIN examples for fixed batch")
    selected_ids = [str(identifier) for identifier in data["scene_ids"][ids]]
    print(json.dumps(dict(predeclared_batch=selected_ids, selection="lexicographically first eligible TRAIN examples; no DEV, no gradient-based selection")), flush=True)
    records = []
    for name in ("best.pt", "last.pt"):
        started = time.perf_counter()
        checkpoint = torch.load(run / name, map_location="cpu", weights_only=False)
        torch.manual_seed(0)
        model = ObservedGeometryRouteHead(config["feature_dim"], config["horizon"], config["candidates"], config["width"],
            config["depth"], config["point_width"], config["endpoint_residual_bound"])
        model.load_state_dict(checkpoint["model"])
        model.train()
        inputs = batch_inputs(data, geometry, ids, "cpu")
        xyz, opened, details = model(**inputs)
        target_xyz, target_open = torch.as_tensor(data["paths"][ids]), torch.as_tensor(data["events"][ids])
        prediction = torch.cat([xyz[:, :, 1:], opened[:, :, 1:, None] * config["event_scale"]], -1)
        target = torch.cat([target_xyz[:, :, 1:], target_open[:, :, 1:, None] * config["event_scale"]], -1)
        matched, assignments = assigned_targets(prediction, target, data["path_mask"][ids])
        squared_error = (prediction - matched).square()
        denominator = squared_error.numel()
        interior = squared_error[:, :, :-1, :3].sum() / denominator
        endpoint = squared_error[:, :, -1, :3].sum() / denominator
        events = squared_error[..., 3].sum() / denominator
        original = positive_assignment_loss(prediction, target, data["path_mask"][ids], "saturation", np.random.default_rng(0))
        if not torch.allclose(interior + endpoint + events, original, atol=1e-7, rtol=1e-5):
            raise RuntimeError("decomposed losses do not reproduce the original saturation assignment loss")
        grounding = config["grounding_weight"] * positive_endpoint_attention_loss(details["attention"], inputs["world_xyz"],
            inputs["valid_mask"], target_xyz[:, :, -1], torch.as_tensor(data["path_mask"][ids]), config["grounding_sigma"])
        parameters = [(parameter_name, parameter) for parameter_name, parameter in model.named_parameters() if parameter_name.startswith("geometry.")]
        groups = {"all_geometry": parameters,
            "attention_producer": [(parameter_name, parameter) for parameter_name, parameter in parameters if not parameter_name.startswith("geometry.fusion.")]}
        group_results = {}
        for group, named_parameters in groups.items():
            tensors = [parameter for _, parameter in named_parameters]
            vectors = {key: gradient_vector(value, tensors) for key, value in
                       [("interior", interior), ("endpoint", endpoint), ("weighted_grounding", grounding), ("events", events)]}
            group_results[group] = dict(parameter_names=[parameter_name for parameter_name, _ in named_parameters],
                interior_vs_endpoint_plus_grounding=compare_vectors(vectors["interior"], vectors["endpoint"] + vectors["weighted_grounding"]),
                interior_vs_grounding=compare_vectors(vectors["interior"], vectors["weighted_grounding"]),
                endpoint_vs_grounding=compare_vectors(vectors["endpoint"], vectors["weighted_grounding"]),
                component_norms={key: float(vector.norm()) for key, vector in vectors.items()})
        record = dict(checkpoint=name, step=checkpoint["step"], checkpoint_sha256=sha256(run / name),
            fixed_target_assignment_indices=assignments,
            loss_components={key: float(value.detach()) for key, value in
                [("interior", interior), ("endpoint", endpoint), ("events", events), ("weighted_grounding", grounding), ("original_path", original)]},
            loss_reconstruction_abs_difference=float((interior + endpoint + events - original).abs().detach()),
            gradients=group_results, cpu_seconds=time.perf_counter() - started)
        records.append(record)
        print(json.dumps(record), flush=True)
    result = dict(scope="local gradient diagnostic on one predeclared TRAIN batch; not proof of general conflict or improvement",
        run=str(run), selected_training_ids=selected_ids, batch_size=batch_size,
        assignment_policy="original full path+event saturation assignment fixed separately at each checkpoint; reused unchanged for every component",
        loss_scaling="interior xyz, endpoint xyz and events preserve their exact contribution to original mean path loss; grounding uses original training weight",
        gradient_scope="shared learned geometry PARAMETERS, including attention producers; grounding need not depend on the intermediate anchor tensor to update attention",
        model_updated=False, original_ADE_checkpoint_selection_preserved=True,
        script_sha256=sha256(Path(__file__)), dataset_fingerprint=geometry["fingerprint"], records=records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gradient_diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    diagnose(args.run, args.output, args.batch_size)


if __name__ == "__main__":
    main()
