"""All three targets from the first two DEV parents, with real geometry attention."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from routeset.common import sha256, write_json
from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_route_head import load_observed_dataset, semantic_endpoint_accuracy
from scripts.train_observed_geometry import load_geometry, batch_inputs


COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#333333']


def project_world(xyz, intrinsics, camera_to_world):
    camera = (np.asarray(xyz)-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    pixels = camera @ intrinsics.T
    return pixels[..., :2]/pixels[..., 2:], camera[..., 2] > 0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.run/'config.json').read_text())
    if config.get('evaluation_protocol') != 'observation_eval_v2':
        raise ValueError('versioned v2 observation protocol required')
    data = load_observed_dataset(config['observations'], config['supervision'], config['cache_dir'], config['horizon'], config['pooling'])
    geometry = load_geometry(data, config['observations'], config['supervision'], config['pixel_stride'])
    parents = sorted(set(data['parent_ids'][data['splits'] == 'DEV_MODEL']))[:2]
    groups = [np.asarray(sorted(np.flatnonzero((data['splits'] == 'DEV_MODEL') & (data['parent_ids'] == parent)),
                               key=lambda index:str(data['scene_ids'][index]))) for parent in parents]
    if len(groups) != 2 or any(len(group) != 3 for group in groups):
        raise ValueError('fixed visualization requires first two DEV parents with all three instructions')
    ids = np.concatenate(groups)
    model = ObservedGeometryRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'],
                                      config['depth'], config['point_width'], config['endpoint_residual_bound'],
                                      anchor_mode=config.get('anchor_mode','soft'))
    checkpoint = torch.load(args.run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model']); model.eval()
    paths, opened, details = model(**batch_inputs(data, geometry, ids, 'cpu'))
    paths, anchors, attention = paths.numpy(), details['anchor_xyz'].numpy(), details['attention'].numpy()
    vmax = float(attention.max())
    args.output.mkdir(parents=True, exist_ok=True)
    rows, figure_paths = [], []
    for parent_index, (parent, group) in enumerate(zip(parents, groups)):
        fig = plt.figure(figsize=(18, 12), constrained_layout=True)
        for row_number, idx in enumerate(group):
            local = parent_index*3+row_number
            source = geometry['sources'][geometry['index'][idx]]
            with Image.open(source[0]) as image:
                rgb = np.asarray(image.convert('RGB'))
            with np.load(source[1], allow_pickle=False) as current:
                K, E = current['camera_intrinsics'], current['camera_extrinsics']
            specification = data['semantic_targets'][idx]
            goal = np.asarray(specification['centers'][specification['target_index']])  # Visualization/evaluation ONLY.
            endpoint_uv, endpoint_front = project_world(paths[local, :, -1], K, E)
            goal_uv, _ = project_world(goal, K, E)
            anchor_uv, _ = project_world(anchors[local], K, E)
            success = semantic_endpoint_accuracy(paths[local, :, -1], specification)
            errors = np.linalg.norm(paths[local, :, -1]-goal, axis=-1)
            # Labels are used ONLY to summarize already-predicted attention.
            observed_xyz = geometry['points']['world_xyz'][geometry['index'][idx]]
            distances = np.linalg.norm(observed_xyz[:, None]-np.asarray(specification['centers'])[None], axis=-1)
            nearest = distances.argmin(1)
            nearby = distances.min(1) <= .04
            target_masses = [float(attention[local, nearby & (nearest == target)].sum())
                             for target in range(len(specification['centers']))]
            rgb_ax = fig.add_subplot(3, 3, row_number*3+1)
            rgb_ax.imshow(rgb)
            stride = config['pixel_stride']; h, w = rgb.shape[:2]
            gh, gw = len(range(stride//2, h, stride)), len(range(stride//2, w, stride))
            first_pixel = stride//2
            heat_extent = (first_pixel-stride/2, first_pixel+(gw-1)*stride+stride/2,
                           first_pixel+(gh-1)*stride+stride/2, first_pixel-stride/2)
            heat = rgb_ax.imshow(attention[local].reshape(gh, gw), extent=heat_extent,
                                 cmap='magma', alpha=.58, vmin=0, vmax=vmax, interpolation='nearest')
            for candidate, uv in enumerate(endpoint_uv):
                if endpoint_front[candidate]:
                    rgb_ax.scatter(*uv, color=COLORS[candidate], marker='x', s=65, linewidths=2)
            rgb_ax.scatter(*goal_uv, c='white', edgecolors='black', marker='*', s=170, linewidths=.7, label='Evaluation goal')
            rgb_ax.scatter(*anchor_uv, c='#00FFFF', edgecolors='black', marker='D', s=42, linewidths=.5, label='Learned anchor')
            rgb_ax.set_xlim(-.5, w-.5); rgb_ax.set_ylim(h-.5, -.5)
            rgb_ax.set_title(data['instructions'][idx]+'\nRGB + attention; crosses = predicted endpoints', fontsize=10)
            rgb_ax.axis('off')
            fig.colorbar(heat, ax=rgb_ax, shrink=.65, label='Attention mass / sampled pixel')
            xy_ax = fig.add_subplot(3, 3, row_number*3+2)
            xyz_ax = fig.add_subplot(3, 3, row_number*3+3, projection='3d')
            references = data['paths'][idx, data['path_mask'][idx]]
            for reference in references:
                xy_ax.plot(reference[:, 0], reference[:, 1], color='#BBBBBB', linewidth=1, alpha=.75)
                xyz_ax.plot(*reference.T, color='#BBBBBB', linewidth=1, alpha=.75)
            for candidate, path in enumerate(paths[local]):
                color = COLORS[candidate]
                xy_ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.7, label='Candidate '+str(candidate+1))
                xy_ax.scatter(*path[-1, :2], color=color, marker='x', s=45)
                xyz_ax.plot(*path.T, color=color, linewidth=1.7)
                xyz_ax.scatter(*path[-1], color=color, marker='x', s=30)
            xy_ax.scatter(*data['current'][idx, :2], c='black', s=30, label='Current start')
            xy_ax.scatter(*goal[:2], c='white', edgecolors='black', s=150, marker='*', label='Evaluation goal')
            xy_ax.scatter(*anchors[local, :2], c='#00FFFF', edgecolors='black', s=45, marker='D', label='Learned anchor')
            xy_ax.set_xlabel('world x (m)'); xy_ax.set_ylabel('world y (m)'); xy_ax.set_aspect('equal', adjustable='datalim')
            xy_ax.grid(alpha=.2)
            xy_ax.set_title('Top view; '+str(int(success.sum()))+'/'+str(len(success))+' strict goal hits\nendpoint error '+
                            ', '.join('%.1f' % (error*100) for error in errors)+' cm', fontsize=10)
            xyz_ax.scatter(*data['current'][idx, :3], c='black', s=25)
            xyz_ax.scatter(*goal, c='white', edgecolors='black', s=100, marker='*')
            xyz_ax.scatter(*anchors[local], c='#00FFFF', edgecolors='black', s=35, marker='D')
            xyz_ax.set_xlabel('world x (m)', fontsize=9); xyz_ax.set_ylabel('world y (m)', fontsize=9); xyz_ax.set_zlabel('world z (m)', fontsize=9, labelpad=2)
            for axis in (xyz_ax.xaxis, xyz_ax.yaxis, xyz_ax.zaxis):
                axis.set_major_locator(MaxNLocator(4))
            xyz_ax.tick_params(labelsize=8)
            xyz_ax.view_init(elev=25, azim=-65); xyz_ax.set_title('All H24 predictions; grey = recorded references', fontsize=10)
            if row_number == 2:
                xy_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, fontsize=7)
            rows.append(dict(id=str(data['scene_ids'][idx]), parent_id=str(parent), instruction=data['instructions'][idx],
                             endpoint_error_m=errors.tolist(), semantic_goal_accuracy=float(success.mean()),
                             learned_anchor_xyz=anchors[local].tolist(), evaluation_goal_xyz=goal.tolist(),
                             reference_count=len(references), attention_max=float(attention[local].max()),
                             diagnostic_attention_mass_within_4cm_nearest_target=target_masses,
                             diagnostic_correct_target_mass=target_masses[specification['target_index']],
                             diagnostic_mass_outside_target_neighborhoods=1-sum(target_masses),
                             diagnostic_note='Evaluation-only target neighborhoods; never an input or success criterion'))
        fig.suptitle(str(parent)+' — all three goals, fixed first-two-DEV-parent selection\n'+
                     'Frozen Qwen + observed RGB-D ordinary set regression; goal labels shown only for evaluation', fontsize=13)
        filename = args.output/(str(parent)+'.png')
        fig.savefig(filename, dpi=160, bbox_inches='tight'); plt.close(fig)
        figure_paths.append(dict(path=str(filename), sha256=sha256(filename)))
    np.savez_compressed(args.output/'six_fixed_predictions.npz', scene_ids=data['scene_ids'][ids], paths=paths,
                        gripper_open=opened.numpy(), attention=attention, learned_anchor=anchors)
    write_json(args.output/'index.json', dict(selection='lexicographically first two DEV_MODEL parent IDs, all three target instructions per parent; no outcome-based selection',
        evaluation_protocol='observation_eval_v2', source_run=str(args.run), checkpoint=str(args.run/'best.pt'),
        checkpoint_sha256=sha256(args.run/'best.pt'), checkpoint_step=checkpoint['step'],
        source_script_sha256=sha256(Path(__file__)), attention_color_scale_max=vmax,
        figures=figure_paths, per_scene=rows, limitation='Predicted task-level paths; no full robot execution or collision validity claimed'))
    print(json.dumps(dict(parent_ids=parents, examples=len(rows), figures=figure_paths)), flush=True)


if __name__ == '__main__':
    main()
