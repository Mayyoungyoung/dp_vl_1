"""Cache only current-camera evidence to refresh features during center updates."""
import argparse
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import *
from scripts.train_observed_geometry import read_geometry
from research_selective_repair_v1.local import features

def build(name,source,data=None):
    data=data or DATA;out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    with np.load(RUN/source/'samples.npz') as z:d={k:z[k] for k in z.files}
    identifiers=set(d['ids']);rows={r['id']:r for r in lines(data/'export/observations.jsonl') if r['id'] in identifiers}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['id'] in rows}
    arrays={k:[] for k in ('visible_points','visible_mask','depth','intrinsics','extrinsics')}
    for i,ident in enumerate(d['ids']):
        ident=str(ident);row=rows[ident]
        with np.load(labels[ident]['observation']) as z:
            depth=z['depth'];intr=z['camera_intrinsics'];extr=z['camera_extrinsics']
        geo=read_geometry(row['image'],labels[ident]['observation'],2)
        color,xyz,valid=geo['rgb'],geo['world_xyz'],geo['valid_mask'].astype(bool)
        keep=valid&(color.max(-1)-color.min(-1)<.08)&(color.mean(-1)>.15)&(color.mean(-1)<.8)&(xyz[...,2]>.78)&(xyz[...,2]<1.1)&(xyz[...,0]>.07)&(xyz[...,0]<.42)
        points=xyz[keep]
        if len(points)>768:points=points[np.linspace(0,len(points)-1,768).astype(int)]
        if i<8:
            actual=features(d['drafts'][i],points.astype(np.float32),dict(depth=depth,camera_intrinsics=intr,camera_extrinsics=extr))
            np.testing.assert_allclose(actual,d['local'][i,...,:29],atol=2e-5,rtol=0)
        padded=np.zeros((768,3),np.float32);mask=np.zeros(768,bool);padded[:len(points)]=points;mask[:len(points)]=True
        for k,v in dict(visible_points=padded,visible_mask=mask,depth=depth.astype(np.float32),intrinsics=intr.astype(np.float32),extrinsics=extr.astype(np.float32)).items():arrays[k].append(v)
    d.update({k:np.asarray(v) for k,v in arrays.items()});np.savez_compressed(out/'samples.npz',**d)
    write(out/'MANIFEST.json',dict(source_sha256=sha(RUN/source/'samples.npz'),samples_sha256=sha(out/'samples.npz'),requests=len(d['ids']),contract='Current RGB-D/camera only; supervision record used solely as observation-file pointer; no mask or truth arrays read',locked_access=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--source',required=True);p.add_argument('--data',type=Path);build(**vars(p.parse_args()))
