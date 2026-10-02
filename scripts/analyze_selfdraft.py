"""Paired development analysis of the measured model-draft training repair."""
import json
from pathlib import Path
import numpy as np


def rows(folder, mechanism, mode):
    source=folder/(mechanism+'_'+mode)/'per_scene.json'
    return {x['scene_id']:x for x in json.loads(source.read_text())}


def difference(a,b,key):
    if set(a)!=set(b): raise ValueError('parents differ')
    ids=sorted(a)
    values=np.array([b[i][key]-a[i][key] for i in ids])
    rng=np.random.default_rng(20261002)
    boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),parent_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
                wins=int((values>0).sum()),ties=int((values==0).sum()),losses=int((values<0).sum()))


def main():
    root=Path(__file__).resolve().parents[1]
    old=root/'reports/v2_completion/rollout_seed0'
    new=root/'reports/v2_completion_selfdraft/rollout_seed0'
    result=dict(scope='same 128 development parents, seed0 only; descriptive, not confirmatory',
        training='400000 supervised target slots each; mixed adds576 draft forward batches; original threads4 vs mixed2',
        decision='repair improves own-draft rollout but does not justify two passes over joint4 or max pooling as a core mechanism',
        comparisons={})
    for method in ('attention','coverage'):
        before=rows(old,method,'self2_then2'); after=rows(new,method,'self2_then2'); joint=rows(new,method,'joint4')
        result['comparisons'][method]=dict(
            repair_vs_reference_training={key:difference(before,after,key) for key in ('unique_valid','valid_rate')},
            repaired_rollout_vs_repaired_joint4={key:difference(joint,after,key) for key in ('unique_valid','valid_rate')})
    output=root/'reports/v2_completion_selfdraft/analysis.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
