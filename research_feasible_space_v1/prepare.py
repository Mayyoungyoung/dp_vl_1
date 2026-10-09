"""Stage A2/A3: legal TRAIN labels, full witness and prototype coverage."""
import json
import numpy as np
from collections import Counter
from scripts.mode_geometry_experiment import ROOT,POOL,PREP,DATA
from routeset.mode_geometry import VOCAB
from scripts.run_observed_probability import read,write,sha,lines
from scripts.train_verified_set import checker
from scripts.evaluate_paired_modes import references
from research_feasible_space_v1.geometry import segment_radii,node_radii_numpy,prototypes

RUN=ROOT/'runs/feasible_space_v1'

def main():
    out=RUN/'prepared';out.mkdir(parents=True,exist_ok=False)
    with np.load(POOL) as z:s={k:z[k] for k in z.files}
    with np.load(PREP/'train.npz') as z:d={k:z[k] for k in z.files}
    assert np.array_equal(d['ids'],s['ids']) and set(s['splits'])=={'TRAIN'} and len(s['ids'])==1152
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    cp=np.zeros((len(s['ids']),16,2,24,3),np.float32);sr=np.zeros((len(s['ids']),16,2,23),np.float32)
    mask=np.zeros((len(s['ids']),16),bool);assignment=np.zeros(s['mask'].shape,np.int64)
    counts=Counter();radii=[];widths=[];rows=[];hashes={}
    for i,ident in enumerate(s['ids']):
        ref=references(labels[str(ident)]);check=checker(ref);cs=ref['truth']['obstacle_centers'];hs=ref['truth']['obstacle_halfsizes'];floor=ref['config']['post_base_z']+.02
        ids=np.flatnonzero(s['mask'][i]);paths=s['paths'][i,ids];events=s['events'][i,ids]
        valid,words=check(paths,events);assert valid.all() and words==s['modes'][i,ids].tolist()
        rr,slack=segment_radii(paths,cs,hs,floor)
        counts['witnesses']+=len(ids);counts['certified_witnesses']+=int((slack>0).all(1).sum());radii.extend(rr.flatten())
        for w in sorted(set(words)):
            mi=VOCAB.index(w);jj=np.flatnonzero(s['mask'][i]&(s['modes'][i]==w));p=s['paths'][i,jj];e=s['events'][i,jj]
            c,a,fall=prototypes(p,check,e);ok,ww=check(c,np.repeat(e[:1],2,0))
            # A mean can change a first-crossing mode even if geometrically clear.
            for v in range(2):
                if not ok[v] or ww[v]!=w:
                    local=np.flatnonzero(a==v)
                    if not len(local):local=np.arange(len(p))
                    c[v]=p[local[((p[local]-c[v])**2).mean((1,2)).argmin()]];fall+=1
            ok,ww=check(c,np.repeat(e[:1],2,0));assert ok.all() and ww==[w,w]
            r,_=segment_radii(c,cs,hs,floor);cp[i,mi]=c;sr[i,mi]=r;mask[i,mi]=True;assignment[i,jj]=a
            nr=node_radii_numpy(r)
            inside=(np.abs(p-c[a]) <= nr[a,:,None]+1e-7).all((1,2))
            counts['prototype_contained_witnesses']+=int(inside.sum());counts['prototype_pairs']+=1;counts['prototype_fallbacks']+=fall
            counts['prototype_centerlines_valid']+=int(ok.sum());widths.extend(r.flatten())
        for k in ('route_config','verification_only','observation'):hashes[ref['label'][k]]=sha(ref['label'][k])
        rows.append(dict(id=str(ident),witnesses=len(ids),modes=int(mask[i].sum())))
        if i%128==0:print(dict(requests=i,counts=dict(counts)),flush=True)
    np.savez_compressed(out/'corridors.npz',ids=s['ids'],centers=cp,radii=sr,mask=mask,assignment=assignment)
    report=dict(counts=dict(counts),all_witness_certificate_coverage=counts['certified_witnesses']/counts['witnesses'],
        two_prototype_coverage=counts['prototype_contained_witnesses']/counts['witnesses'],
        radius_quantiles_m=np.quantile(radii,[0,.1,.5,.9,1]).tolist(),prototype_radius_quantiles_m=np.quantile(widths,[0,.1,.5,.9,1]).tolist(),
        rows=rows,support_sha256=sha(POOL),prepared_sha256=sha(PREP/'train.npz'),corridors_sha256=sha(out/'corridors.npz'),
        oracle_label_input_hashes=hashes,locked_access=False,role='TRAIN',note='Reference corridor diagnostic only; centerline is already a verified witness. No neural advantage inferred.')
    write(out/'MANIFEST.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('rows','oracle_label_input_hashes')}),flush=True)

if __name__=='__main__':main()
