"""Ordinary TRAIN-fitted observation-only localization, no generated routes."""
import argparse
from pathlib import Path
import time
import numpy as np
from scripts import observation_prototype_grounding as prototype
from scripts.paired_modes_data import DATA
from scripts.research_v3_frequency import RUN
from scripts.run_observed_probability import read,write,sha,lines


def strided_loader(original):
    def load(*args):
        grid,paths=original(*args)
        return tuple(x[::2,::2] for x in grid),paths
    return load


def main(output):
    out=RUN/output;out.mkdir(exist_ok=False);tic=time.monotonic()
    folder=DATA/'export';obs=lines(folder/'observations.jsonl')
    labels={r['id']:r for r in lines(folder/'supervision.jsonl')}
    train=[r for r in obs if r['split']=='TRAIN'];dev=[r for r in obs if r['split']=='DEV_MODEL']
    assert len(train)==1152 and len(dev)==288
    assert not {r['parent_id'].rsplit('_',1)[0] for r in train}&{r['parent_id'].rsplit('_',1)[0] for r in dev}
    original=prototype.load_observation
    prototype.load_observation=strided_loader(original)
    try:
        model,parents=prototype.fit_prototypes(folder,train,labels)
        model['observed_grid_stride']=2;write(out/'model.json',model)
        predictions=[];hashes={}
        for r in dev:
            grid,files=prototype.load_observation(folder,r,labels[r['id']]['observation'])
            endpoint,details=prototype.predict(*grid,r['instruction'],model)
            predictions.append(dict(id=r['id'],family=r['parent_id'].rsplit('_',1)[0],
                endpoint=None if endpoint is None else endpoint.tolist(),details=details))
            for f in files:hashes[str(f)]=sha(f)
        write(out/'predictions.json',predictions)
    finally:prototype.load_observation=original
    # Only after sealing every prediction, open DEV task labels for evaluation.
    baseline={r['id']:r for r in read(RUN/'anchor_support_audit_v1/rows.json')['mean'] if r['split']=='DEV_MODEL'}
    rows=[]
    for pred in predictions:
        checked=prototype.evaluate_endpoint(pred['endpoint'],labels[pred['id']]['semantic_targets'])
        row=dict(pred,**checked,mean_any_endpoint_correct=int(not baseline[pred['id']]['all_endpoints_wrong']))
        rows.append(row)
    families=sorted({r['family'] for r in rows});assert len(families)==32
    diff=np.array([np.mean([r['semantic_goal_accuracy']-r['mean_any_endpoint_correct'] for r in rows if r['family']==f]) for f in families])
    rng=np.random.default_rng(610091);ci=np.quantile(diff[rng.integers(32,size=(10000,32))].mean(1),[.025,.975])
    result=dict(requests=len(rows),prototype_correct=sum(r['semantic_goal_accuracy'] for r in rows),
        mean_any_endpoint_correct=sum(r['mean_any_endpoint_correct'] for r in rows),
        abstentions=sum(r['endpoint'] is None for r in rows),difference=float(diff.mean()),CI95=ci.tolist(),
        gate=bool(sum(r['semantic_goal_accuracy']-r['mean_any_endpoint_correct'] for r in rows)>=6 and ci[0]>0),model_sha256=sha(out/'model.json'),
        input_sha256=hashes,prototype_source_sha256=sha(Path(prototype.__file__)),
        train_parent_count=len(parents),train_family_count=len({r['parent_id'].rsplit('_',1)[0] for r in train}),elapsed_seconds=time.monotonic()-tic,
        scope='Closed-exact-instruction RGB-D prototype, all original TRAIN only; no new routes, no goal/geometry labels in prediction; no novelty claim')
    write(out/'rows.json',rows);write(out/'RESULTS.json',result)
    print({k:v for k,v in result.items() if k!='input_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
