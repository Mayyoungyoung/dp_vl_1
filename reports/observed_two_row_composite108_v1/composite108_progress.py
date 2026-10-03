"""Read-only compact status; no torch, checkpoint reads, search or model calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read_json(path):
    if not path.is_file():return None
    try:return json.loads(path.read_text())
    except (ValueError,OSError):return None


def log_rows(path):
    if not path.is_file():return []
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-262144))
        tail=stream.read().decode('utf-8',errors='replace')
    result=[]
    for line in tail.splitlines():
        try:
            row=json.loads(line)
            if isinstance(row,dict):result.append(row)
        except ValueError:pass
    return result


def compact(run):
    run=Path(run);status=read_json(run/'train.status.json') or {};model=run/'peak_seed0'
    history=read_json(model/'history.json') or []
    logged=[r for r in log_rows(run/'train.log') if type(r.get('step')) is int]
    latest=max(logged,key=lambda r:r['step']) if logged else {}
    evaluation=history[-1] if history else {}
    m=evaluation.get('dev_model',{})
    result=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
        recorded_status=status.get('status'),exit_code=status.get('exit_code'),
        record_pid=status.get('pid'),child_pid=status.get('child_pid'),
        start_utc=status.get('start_utc'),end_utc=status.get('end_utc'),
        code_commit=status.get('code_commit'),last_logged_step=latest.get('step'),
        target_steps=12000,last_logged_loss=latest.get('loss'),logged_elapsed_seconds=latest.get('elapsed_s'),
        completed_selection_records=len(history),last_selection_step=evaluation.get('step'),
        latest_dev={k:m.get(k) for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK',
                                       'semantic_goal_accuracy','selection_score')},
        no_checkpoint_or_raw_reads=True,status_is_recorded_not_process_liveness=True)
    summary=read_json(model/'summary.json')
    if summary is not None:
        result['actual_summary']={k:summary.get(k) for k in ('last_step','best_step','trajectory_exposures','elapsed_s','gpu_hours_reserved')}
    diagnostic=read_json(run/'fixed-last-train.status.json')
    if diagnostic is not None:result['fixed_last_train']={k:diagnostic.get(k) for k in ('status','exit_code','child_pid','start_utc','end_utc')}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1'))
    print(json.dumps(compact(p.parse_args().run_root),indent=2,allow_nan=False))
