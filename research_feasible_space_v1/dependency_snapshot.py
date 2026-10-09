"""Copy fixed legal dependencies into the dedicated auditable result archive."""
import argparse,json,shutil
from scripts.mode_geometry_experiment import ROOT,Q,POOL,PREP
from scripts.run_observed_probability import sha,write
from research_feasible_space_v1.prepare import RUN


def main(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    deps=[('C0_pretrained.pt',ROOT/'runs/mode_geometry_v1/canonical_C_seed0/last.pt','6090df425d0ac237f275620f5de5e9453ce8e5c9638fbda1afed2e39c4ea6718'),
          ('Q_frozen_scorer.pt',Q,'2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'),
          ('TRAIN_verified_support.npz',POOL,'101c2a4acff9ff08a34ac6b5c0b3138f04ffd5deaea04c2ac8daf0bf7512ca33'),
          ('TRAIN_observed_context.npz',PREP/'train.npz','ef2e3a75bc1c56aea8254bd9c6563f750e097807a64e7948b4f4447cd4d5cccd')]
    records={}
    for target,source,expected in deps:
        assert sha(source)==expected,(source,'fixed dependency changed')
        shutil.copy2(source,out/target);assert sha(out/target)==expected
        records[target]=dict(source=str(source),sha256=expected,bytes=source.stat().st_size)
    write(out/'MANIFEST.json',dict(files=records,scope='Existing immutable pretrained/scorer weights and TRAIN-only derived support/context; no reserved payload',locked_access=False))
    print(json.dumps({k:v['bytes'] for k,v in records.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
