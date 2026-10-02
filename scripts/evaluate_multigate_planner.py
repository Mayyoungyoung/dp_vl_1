"""Transparent analytical tier-A planner: exactly K generated full paths.

Enumerate inexpensive opening-pair intents, order them by geometric length,
then instantiate K paths. Intent enumeration is included in timing, and no
generated full path is discarded. Only applicable to this special wall family.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from routeset.common import write_json, sha256
from routeset.multigate import load_dataset, unpack_scene, route_from_openings, route_metrics, path_validity


def plan(scene,k,horizon=24):
    start,goal,walls = unpack_scene(scene)
    intent = []
    for first in np.flatnonzero(walls[0,10:14]>.5):
        for second in np.flatnonzero(walls[1,10:14]>.5):
            y0 = walls[0,2:10].reshape(4,2)[first].mean()
            y1 = walls[1,2:10].reshape(4,2)[second].mean()
            # Rank intents from a few scalar distances; no full path generated.
            cost = abs(y0-start[1])+abs(y1-y0)+abs(goal[1]-y1)
            intent.append((cost,int(first),int(second)))
    intent.sort()
    return np.stack([route_from_openings(scene,*intent[index%len(intent)][1:],horizon=horizon)
                     for index in range(k)])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    data=load_dataset(a.data,'DEV_MODEL'); rows=[]
    for k in [1,2,4,8]:
        paths=[]; times=[]; scores=[]
        for scene in data['scenes']:
            tic=time.perf_counter(); result=plan(scene,k,data['paths'].shape[-2])
            check=path_validity(result,scene)
            # Deterministic deployment-available checker, not an oracle score.
            score=check['valid'].astype(float)*100-check['lengths']
            times.append((time.perf_counter()-tic)*1000); paths.append(result); scores.append(score)
        metrics=route_metrics(np.stack(paths),data['scenes'],data['modes'],data['path_mask'],scores=np.stack(scores))
        row={key:value for key,value in metrics.items() if np.isscalar(value)}
        row.update(K=k,method='analytical_opening_pair_planner',tier='controlled_geometry',split='DEV_MODEL',
                   generation_and_check_ms_median=float(np.median(times)),full_paths_generated=k,
                   discarded_full_paths=0,information='same true geometry and true endpoints',
                   intent_enumeration='all opening pairs; included in time',dataset_sha256=sha256(a.data))
        rows.append(row)
        np.savez_compressed(out/('predictions_k%d.npz'%k),paths=np.stack(paths),scene_ids=data['scene_ids'])
    write_json(out/'results.json',rows); print(json.dumps(rows))


if __name__=='__main__':main()
