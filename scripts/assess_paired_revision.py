"""Apply the frozen REVISION1_PROTOCOL gate to all three final checkpoints."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha


def main(source,output):
    source,output=Path(source),Path(output);d=read(source)
    means={}
    for arm in ('R0','R1','R2','R3'):
        keys=['paired_dev/%s/seed%d'%(arm,s) for s in range(3)]
        means[arm]={k:float(np.mean([d['results'][s][k] for s in keys])) for k in ('valid','modes','coverage')}
        means[arm]['shared_recall']=float(np.mean([d['paired'][s]['shared_recall'] for s in keys]))
    conditions={
        'mean_modes_above_R1':means['R3']['modes']>means['R1']['modes'],
        'mean_shared_recall_above_R1':means['R3']['shared_recall']>means['R1']['shared_recall'],
        'mean_validity_loss_at_most_02':means['R3']['valid']>=means['R1']['valid']-.02}
    for seed in range(3):conditions['seed%d_validity_loss_at_most_02'%seed]=d['results']['paired_dev/R3/seed%d'%seed]['valid']>=d['results']['paired_dev/R1/seed%d'%seed]['valid']-.02
    if output.exists():raise FileExistsError(output)
    write(output,dict(passed=all(conditions.values()),conditions=conditions,means=means,input_sha256=sha(source),
        scope='Apply the pre-fit revision1 development gate; not independent final-test evidence or statistical significance.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
