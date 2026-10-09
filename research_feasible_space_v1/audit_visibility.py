"""Observed ray probes vs actual failures; no truth input to ray classification."""
import argparse,itertools,json
import numpy as np
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.geometry import node_radii_numpy
from research_feasible_space_v1.observation_evidence import classify_probes
from research_realized_coverage_v1.core import DATA,lines,read,write,sha

def main(name,names):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'};results={};allrows={};hashes={}
    corners=np.array(list(itertools.product((-1,1),repeat=3)))
    for n in names:
        folder=RUN/n/'eval_adaptive';rows=read(folder/'rows.json');byid={r['id']:r for r in rows}
        with np.load(folder/'pool.npz') as z:pool={k:z[k] for k in z.files}
        records=[]
        for i,ident in enumerate(pool['ids']):
            file=labels[str(ident)]['observation'];hashes[file]=sha(file)
            # Only current depth/camera whitelist; no hidden object/config fields.
            with np.load(file) as a:depth=a['depth'];intrinsics=a['camera_intrinsics'];pose=a['camera_extrinsics']
            c=pool['centers'][i];rho=node_radii_numpy(pool['radii'][i]);vertices=c[...,None,:]+rho[...,None,None]*corners
            status=classify_probes(vertices,depth,intrinsics,pose)
            for slot in range(8):
                fractions={v:float((status[slot]==v).mean()) for v in ('unknown','free_at_probe','observed_surface')}
                records.append(dict(id=str(ident),slot=slot,**fractions,actual_valid=byid[str(ident)]['valid'][slot],cell_eval_certified=byid[str(ident)]['corridor_certified_under_eval_truth'][slot]))
        results[n]=dict(probe_fractions={v:float(np.mean([r[v] for r in records])) for v in ('unknown','free_at_probe','observed_surface')},
            unknown_mean_valid=float(np.mean([r['unknown'] for r in records if r['actual_valid']])),unknown_mean_invalid=float(np.mean([r['unknown'] for r in records if not r['actual_valid']])))
        allrows[n]=records
    report=dict(models=results,scope='Calibrated current-depth probes only; not whole-region free/occupied certificate; no learned uncertainty claim',observation_sha256=hashes,locked_access=False)
    write(out/'RESULTS.json',report);write(out/'rows.json',allrows);print(json.dumps({k:v for k,v in report.items() if k!='observation_sha256'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--names',nargs='+',required=True);main(**vars(p.parse_args()))
