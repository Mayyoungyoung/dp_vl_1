"""Physically separate development, scoring and locked arrays without evaluating them."""
import argparse
import json
from pathlib import Path
import numpy as np
from routeset.common import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    with np.load(a.data,allow_pickle=False) as f: data = {key:f[key] for key in f.files}
    groups = {'development':['TRAIN','DEV_MODEL'],'scoring':['DEV_SCORE','CALIBRATION'],
              'locked':['TEST_LOCKED','OOD_LOCKED']}
    manifest = {'source_sha256':sha256(a.data),'artifacts':{}}
    parents = []
    for name,splits in groups.items():
        mask = np.isin(data['splits'],splits)
        path = out/(name+'.npz')
        if path.exists(): raise FileExistsError(str(path))
        np.savez_compressed(path,**{key:value[mask] for key,value in data.items()})
        parents.append(set(data['parent_ids'][mask]))
        manifest['artifacts'][name]={'sha256':sha256(path),'splits':splits,'examples':int(mask.sum())}
    assert all(not (parents[i]&parents[j]) for i in range(3) for j in range(i+1,3))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__': main()
