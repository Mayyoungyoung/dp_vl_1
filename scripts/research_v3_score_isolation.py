"""Mechanical cross-corpus isolation audit of explicitly permitted exports."""
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from scripts.run_observed_probability import OLD,read,write,sha,lines,verify_role
from scripts.paired_modes_data import DATA as PAIRED
from scripts.research_v3_score_data import DATA as NEW
from scripts.research_v3_frequency import RUN


def geometry_digest(centers,halves,goals):
    c,h,g=(np.asarray(x,dtype=np.float64) for x in (centers,halves,goals))
    assert c.ndim==2 and c.shape[1]==3 and c.shape==h.shape and g.ndim==2 and g.shape[1]==3
    assert all(np.isfinite(a).all() for a in (c,h,g))
    boxes=np.rint(np.rint(np.c_[c,h]*1e6)/1000).astype(np.int64)
    targets=np.rint(np.rint(g*1e6)/1000).astype(np.int64)
    value=dict(boxes=sorted(map(tuple,boxes.tolist())),goals=sorted(map(tuple,targets.tolist())))
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main(output):
    out=RUN/output;out.mkdir(exist_ok=False)
    folders={'old_generator':OLD,'paired_generator':PAIRED/'export'}
    for role in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION','FUTURE_GENERATOR_TRAIN'):
        folders['old_'+role]=verify_role(role)
    for role in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION'):folders['new_'+role]=NEW/'exports'/role
    records=[];hashes={}
    for domain,folder in folders.items():
        obs=lines(folder/'observations.jsonl')
        assert all(r['split'] in ('TRAIN','DEV_MODEL','SCORE_TRAIN','DEV_SCORE','CALIBRATION','FUTURE_GENERATOR_TRAIN') for r in obs)
        labels={r['id']:r for r in lines(folder/'supervision.jsonl')};seen={}
        hashes[str(folder/'observations.jsonl')]=sha(folder/'observations.jsonl')
        hashes[str(folder/'supervision.jsonl')]=sha(folder/'supervision.jsonl')
        for r in obs:
            label=labels[r['id']];ident=r['parent_id'];image=folder/r['image'];observation=folder/label['observation']
            pointer=(str(image),str(observation),r['split'])
            if ident in seen:assert seen[ident]==pointer;continue
            seen[ident]=pointer;truth=folder/label['verification_only']
            with np.load(truth) as z:geo=geometry_digest(z['obstacle_centers'],z['obstacle_halfsizes'],label['semantic_targets']['centers'])
            d=hashlib.sha256()
            fields={'rgb':np.array(Image.open(image).convert('RGB'))}
            with np.load(observation) as z:
                for key in ('depth','camera_intrinsics','camera_extrinsics','gripper_pose','gripper_open'):fields[key]=z[key]
            for key,value in sorted(fields.items()):
                d.update(key.encode());d.update(str((value.shape,str(value.dtype))).encode());d.update(value.tobytes())
            records.append(dict(domain=domain,parent=ident,role=r['split'],geometry_1mm_unordered=geo,observed_input_sha256=d.hexdigest()))
            for p in (image,observation,truth):hashes[str(p)]=sha(p)
    conflicts=[]
    for key in ('geometry_1mm_unordered','observed_input_sha256'):
        groups={}
        for r in records:groups.setdefault(r[key],[]).append(r)
        for digest,rr in groups.items():
            if any(r['domain'].startswith('new_') for r in rr) and len({(r['domain'],r['parent']) for r in rr})>1:
                conflicts.append(dict(kind=key,digest=digest,parents=rr))
    result=dict(parents_by_domain={k:sum(r['domain']==k for r in records) for k in folders},
        conflicts=conflicts,passed=not conflicts,input_sha256=hashes,
        scope='Only registered nonlocked exports. New-versus-old and within-new identity check; historical old-old duplication is not reclassified. Geometry includes unordered boxes and target centers at1mm, observations include RGB-D/camera/current state. No TEST_LOCKED access.')
    write(out/'RESULTS.json',result);write(out/'parents.json',records)
    print(dict(parents=result['parents_by_domain'],conflicts=len(conflicts),passed=result['passed']),flush=True)
    if conflicts:raise RuntimeError('New score-data isolation gate fails; preserve artifacts and invalidate comparison')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
