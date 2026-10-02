"""Plot captured RGB and physically executed collector paths (not model predictions)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parent', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    observations = [json.loads(line) for line in (a.data / 'observations.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (a.data / 'supervision.jsonl').read_text().splitlines()]
    observations = {row['id']: row for row in observations if row['parent_id'] == a.parent}
    rows = [row for row in rows if row['parent_id'] == a.parent]
    fig = plt.figure(figsize=(11, 4.6), constrained_layout=True)
    rgb_ax = fig.add_subplot(1, 2, 1)
    rgb_ax.imshow(Image.open(a.data / next(iter(observations.values()))['image']))
    rgb_ax.set_title('Shared initial RGB; three language goals')
    rgb_ax.axis('off')
    path_ax = fig.add_subplot(1, 2, 2, projection='3d')
    colors = ['#c43d35', '#df9a21', '#397f41']
    for row, color in zip(rows, colors):
        instruction = observations[row['id']]['instruction']
        for index, path in enumerate(row['routes']):
            xyz = np.load(a.data / path)['gripper_pose'][:, :3]
            path_ax.plot(*xyz.T, color=color, alpha=.8, lw=1.8,
                         linestyle=['-', '--', ':'][index % 3],
                         label=instruction if index == 0 else None)
        target = np.asarray(row['semantic_targets']['centers'][row['semantic_targets']['target_index']])
        path_ax.scatter(*target, color=color, marker='o', s=50, edgecolors='white')
    path_ax.set_xlabel('world x (m)')
    path_ax.set_ylabel('world y (m)')
    path_ax.set_zlabel('world z (m)')
    path_ax.set_title('Executed collector trajectories, 3 attempts per goal')
    path_ax.view_init(elev=24, azim=-72)
    path_ax.legend(loc='lower left', bbox_to_anchor=(-0.05, -.17), fontsize=8)
    fig.suptitle('RLBench-derived pilot: collection evidence, not learned-model performance', fontsize=12)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
