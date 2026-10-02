"""Match actual online Qwen requests to cached-feature predictions, by ID.

This is an artifact/latency audit, not a new metric or model selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    args=parser.parse_args();root=args.root.resolve()
    reports=root/'reports/observed_obstacle_reserved96_full_inference_v1'
    measured=root/'runs/observed_obstacle_reserved96_full_inference_v1'
    original=root/'runs/observed_obstacle_reserved96_v1'
    results={}
    for arm in ('soft','peak'):
        report=json.loads((reports/arm/'summary.json').read_text())
        status=json.loads((reports/(arm+'.status.json')).read_text())
        if status['exit_code']!=0:raise ValueError('online job did not complete')
        checkpoint=original/(arm+'_seed0')/'best.pt'
        online=measured/arm/'predictions.npz'
        cached=original/(arm+'_seed0')/'dev_model/predictions.npz'
        if digest(checkpoint)!=report['checkpoint_sha256'] or digest(online)!=report['prediction_sha256']:
            raise ValueError('checkpoint or online output changed')
        with np.load(online,allow_pickle=False) as a,np.load(cached,allow_pickle=False) as b:
            left,right=list(map(str,a['scene_ids'])),list(map(str,b['scene_ids']))
            if len(left)!=24 or len(set(left))!=24 or set(left)!=set(right):
                raise ValueError('all24 DEV inputs required on both sides')
            order=[right.index(identifier) for identifier in left]
            drift={key:float(np.abs(a[key]-b[key][order]).max()) for key in ('paths','gripper_open')}
        values={key:report[key] for key in ('model_loading_seconds','first_request_ms',
            'all_requests_ms_median','all_requests_ms_p95','after_first_two_requests_ms_median',
            'after_first_two_requests_ms_p95','cuda_peak_allocated_bytes','loaded_checkpoint_step')}
        results[arm]=dict(**values,examples=24,maximum_absolute_drift=drift,
            predictions_match_at_1e_5=all(value<=1e-5 for value in drift.values()),
            hashes={str(p.relative_to(root)):digest(p) for p in (checkpoint,online,cached,reports/arm/'summary.json')})
    audit=dict(arms=results,source_sha256=digest(__file__),
        scope='real online frozen Qwen, input reads/processor/RGB-D/head/output transfer; no score or collision check in timed requests',
        model_revision='89644892e4d85e24eaac8bacfd4f463576704203',
        note='Same24 DEV parents/instructions, no merged candidate pools. File pages may be warm; first request and model load separately retained.')
    destination=reports/'cached_vs_online_audit.json'
    if destination.exists():raise FileExistsError('preserve previous audit')
    destination.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:{k:v for k,v in value.items() if k!='hashes'} for key,value in results.items()},indent=2))
    if not all(value['predictions_match_at_1e_5'] for value in results.values()):
        raise RuntimeError('online/cached mismatch recorded; do not attach cached quality to these timings')


if __name__=='__main__':main()
