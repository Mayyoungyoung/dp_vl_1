"""Read research job records without restarting jobs or trusting stale running flags."""
import argparse
import datetime
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root)
    jobs = []
    for folder in sorted((root/'runs').iterdir()):
        if not folder.is_dir() or not folder.name.startswith(('v2_', 'observ', 'obstacle_', 'constraint_', 'multigate_', 'vlm_', 'multitask_')):
            continue
        paths = set(folder.rglob('*.status.json')) | set(folder.glob('*status.json'))
        for path in sorted(paths):
            record = json.loads(path.read_text())
            record['record_path'] = str(path)
            pid = record.get('pid')
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    record['recorded_pid_exists_at_snapshot'] = True
                except ProcessLookupError:
                    record['recorded_pid_exists_at_snapshot'] = False
                except PermissionError:
                    record['recorded_pid_exists_at_snapshot'] = 'permission_denied'
            jobs.append(record)
        if (folder/'status').is_file():
            record = dict(run_id=folder.name, record_path=str(folder/'status'))
            for name in ('status', 'pid', 'exit_code', 'started_at', 'finished_at', 'source_commit',
                         'code_release', 'source_sha256.txt', 'lifecycle_error.txt'):
                path = folder/name
                if path.is_file():
                    record[name] = path.read_text().strip()
            record['logs'] = [str(path) for path in sorted(folder.glob('*.log'))]
            record['resume_policy'] = ('Read the frozen launcher and collected parent manifests before restart; '
                                       'do not blindly rerun completed collection or append --resume to a cache command.')
            for launcher in ('frozen_job.sh', 'launcher.sh', 'launch.sh'):
                if (folder/launcher).is_file():
                    record['frozen_launcher'] = str(folder/launcher)
                    break
            if str(record.get('pid','')).isdigit():
                record['pid'] = int(record['pid'])
                try:
                    os.kill(record['pid'], 0)
                    record['recorded_pid_exists_at_snapshot'] = True
                except ProcessLookupError:
                    record['recorded_pid_exists_at_snapshot'] = False
                except PermissionError:
                    record['recorded_pid_exists_at_snapshot'] = 'permission_denied'
            jobs.append(record)
    result = dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host='wzy3090', root=str(root), records=jobs,
                  warning='PID existence alone does not verify ownership or completion; inspect each log/exit and source hash before restart.')
    Path(a.output).write_text(json.dumps(result, indent=2))
    print('Saved %d recorded jobs.' % len(jobs))


if __name__ == '__main__':
    main()
