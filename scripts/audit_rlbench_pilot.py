"""Read-only audit and all-parent figures for an explicitly DEV_COLLECTION pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(data, output):
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['split'] != 'DEV_COLLECTION':
        raise ValueError('This audit must not open formal or locked parent data')
    if output.exists():
        raise FileExistsError('Preserve existing audit output')
    observations, attempts = rows(data/'observations.jsonl'), rows(data/'attempts.jsonl')
    assert all(row['split'] == 'DEV_COLLECTION' for row in observations)
    output.mkdir(parents=True)
    parents = sorted({row['parent_id'] for row in observations})
    index, audits, images = [], [], []
    for parent in parents:
        folder = data/parent
        records = [row for row in attempts if row['parent_id'] == parent and 'attempt' in row]
        assert len({row['attempt'] for row in records}) == len(records), 'duplicate attempt IDs'
        descriptions = [row['instruction'] for row in observations if row['parent_id'] == parent]
        for filename in sorted(folder.iterdir()):
            if filename.is_file():
                index.append(dict(parent_id=parent, path=str(filename), bytes=filename.stat().st_size,
                                  sha256=digest(filename), scope='DEV_COLLECTION only'))
        with np.load(folder/'observation.npz', allow_pickle=False) as archive:
            assert {'depth','gripper_pose','gripper_open','camera_intrinsics','camera_extrinsics'} == set(archive.files)
            observation = {key: archive[key].copy() for key in archive.files}
        route_records, routes_for_plot = [], []
        for record in records:
            if not record['success']:
                continue
            path = folder/('route_%02d.npz' % record['attempt'])
            with np.load(path, allow_pickle=False) as archive:
                poses, opened = archive['gripper_pose'], archive['gripper_open']
            assert len(poses) == len(opened) == record['steps']
            assert np.isfinite(poses).all() and np.isfinite(opened).all()
            actual = hashlib.sha256(poses.tobytes()).hexdigest()
            assert actual == record['trajectory_sha256'], 'trajectory hash mismatch'
            transitions = np.flatnonzero(np.diff(opened > .5)).astype(int).tolist()
            route_records.append(dict(attempt=record['attempt'], path=str(path), sha256=digest(path),
                trajectory_sha256=actual, steps=len(poses), observed_open_close_transition_indices=transitions,
                event_values_unique=np.unique(opened).tolist(), start_xyz=poses[0,:3].tolist(),
                end_xyz=poses[-1,:3].tolist(), path_length_m=float(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=-1).sum()),
                first_pose_distance_from_reference_m=float(np.linalg.norm(poses[0,:3]-observation['gripper_pose'][:3]))))
            routes_for_plot.append((record['attempt'],poses,opened))
        restored = [record.get('restore',{}) for record in records]
        audit_row = dict(parent_id=parent, task=parent.rsplit('_',1)[0], instructions=descriptions,
            attempts=len(records), successes=sum(row['success'] for row in records),
            failed_attempts=[row for row in records if not row['success']],
            all_recorded_state_exact=all(row.get('max_abs') == 0 for row in restored),
            all_recorded_rgb_exact=all(row.get('rgb_max_difference') == 0 for row in restored),
            all_recorded_inventory_equal=all(row.get('same_object_inventory') is True for row in restored),
            all_recorded_language_equal=all(row.get('language_equal') is True for row in restored),
            total_attempt_seconds=sum(row.get('seconds',0) for row in records),
            near_duplicate_successes=sum(row.get('near_duplicate',False) for row in records),
            routes=route_records, full_motion_collision_checked=all(row.get('full_motion_collision_checked',False) for row in records),
            unique_valid_route_types=None, camera_intrinsics=observation['camera_intrinsics'].tolist(),
            camera_extrinsics=observation['camera_extrinsics'].tolist())
        audits.append(audit_row)
        images.append((parent,folder/'front.png',descriptions[0],routes_for_plot))
        shutil.copy2(folder/'front.png',output/(parent+'_front.png'))
    setup = [row for row in attempts if row.get('phase') == 'initial_reset']
    result = dict(data=str(data),source_sha256=digest(__file__),manifest_sha256=digest(data/'manifest.json'),
        split='DEV_COLLECTION',collection_summary=json.loads((data/'summary.json').read_text()),
        parents_requested=len(manifest['requested_tasks'])*manifest['parents_per_task'],parents_with_observations=len(parents),
        all_attempt_records=len(attempts),initial_setup_failure_records=setup,route_attempt_records=sum(row['attempts'] for row in audits),
        total_route_successes=sum(row['successes'] for row in audits),per_parent=audits,
        limitations=['Each original simulator task success flag checked by collector; no learned model evaluated',
                    'State/RGB equality only for collector recorded fields, not a complete physics certificate',
                    'Free-prefix arm collision checks do not certify full demonstrated motion',
                    'Near-duplicate distance checks do not define route types; route type counts remain null'])
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    (output/'artifact_index.json').write_text(json.dumps(dict(files=index),indent=2))
    for name in ['manifest.json','summary.json','attempts.jsonl','observations.jsonl']:
        shutil.copy2(data/name,output/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(images),3,figsize=(13,3.5*len(images)),squeeze=False)
    for row,(parent,image,instruction,routes) in enumerate(images):
        axes[row,0].imshow(plt.imread(image));axes[row,0].axis('off')
        axes[row,0].set_title(parent+'\n'+instruction,fontsize=9)
        for attempt,poses,opened in routes:
            axes[row,1].plot(poses[:,0],poses[:,2],label='attempt '+str(attempt))
            axes[row,2].step(np.arange(len(opened)),opened,where='post',label='attempt '+str(attempt))
        axes[row,1].set(xlabel='world x (m)',ylabel='world z (m)',title='Full recorded trajectories, x-z projection')
        axes[row,2].set(xlabel='recorded step',ylabel='gripper open',title='Original event sequence retained')
        axes[row,1].legend(fontsize=7);axes[row,2].legend(fontsize=7)
    fig.suptitle('Every requested DEV_COLLECTION parent and all successful attempts\nCollection feasibility only; route types and full-motion collision certification unverified',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.955));fig.savefig(output/'all_parent_routes_events.png',dpi=135);plt.close(fig)
    print(json.dumps(dict(parents=len(parents),attempts=result['route_attempt_records'],successes=result['total_route_successes'],setup_failures=len(setup))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.data,args.output)
